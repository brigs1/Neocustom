# Lint as: python3
import json
import logging
import os

import datasets

from PIL import Image
import numpy as np

from transformers import AutoTokenizer

def load_image(image_path, img_size=224):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    # resize image to 224x224
    image = image.resize((img_size, img_size))
    image = np.asarray(image)
    image = image.transpose(2, 0, 1) # move channels to first dimension
    return image, (w, h)


def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


def simplify_bbox(bbox):
    return [
        min(bbox[0::2]),
        min(bbox[1::2]),
        max(bbox[2::2]),
        max(bbox[3::2]),
    ]


def merge_bbox(bbox_list):
    x0, y0, x1, y1 = list(zip(*bbox_list))
    return [min(x0), min(y0), max(x1), max(y1)]


_URL = "https://github.com/brigs1/Neocustom/blob/master/dataset.zip"

logger = logging.getLogger(__name__)


class NeocustomConfig(datasets.BuilderConfig):
    """BuilderConfig for Neocustom"""

    def __init__(self, **kwargs):
        """BuilderConfig for Neocustom.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(NeocustomConfig, self).__init__(**kwargs)


def Neocustom_to_xfun(Neocustom_path):
    ret = {}
    train_path = f"{Neocustom_path}/dataset/training_data!!!!!!/"
    ann_dir = os.path.join(train_path, "annotations!!!!/")
    img_dir = os.path.join(train_path, "images/")
    documents = []
    for guid, file in enumerate(sorted(os.listdir(ann_dir))):
        with open(os.path.join(ann_dir, file), "r", encoding="utf8") as f:
            data = json.load(f)
        documents.append({'id': file, 'document': data["form"], 'img': {'fname': file.replace("json", "png")}})
    with open('train.json', 'w') as outfile:
        json.dump({'documents': documents}, outfile)

    ret['train'] = ['train.json', img_dir]

    testing_path = f"{Neocustom_path}/dataset/testing_data/"
    ann_dir = os.path.join(testing_path, "annotations")
    img_dir = os.path.join(testing_path, "images")
    documents = []
    for guid, file in enumerate(sorted(os.listdir(ann_dir))):
        with open(os.path.join(ann_dir, file), "r", encoding="utf8") as f:
            data = json.load(f)
        documents.append({'id': file, 'document': data["form"], 'img': {'fname': file.replace("json", "png")}})
    with open('val.json', 'w') as outfile:
        json.dump({'documents': documents}, outfile)

    ret['val'] = ['val.json', img_dir]

    return ret


class Neocustom(datasets.GeneratorBasedBuilder):
    """Neocustom dataset."""

    BUILDER_CONFIGS = [NeocustomConfig(name="Neocustom", version=datasets.Version("1.0.0"), description="Neocustom dataset"), ]

    tokenizer = AutoTokenizer.from_pretrained('microsoft/layoutxlm-base', pad_token='<pad>')

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "input_ids": datasets.Sequence(datasets.Value("int64")),
                    "bbox": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "labels": datasets.Sequence(
                        datasets.ClassLabel(
                            names=["가", "나", "다", "라", "마"]
                        )
                    ),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                    "hd_image": datasets.Array3D(shape=(3, 1024, 1024), dtype="uint8"),
                    "entities": datasets.Sequence(
                        {
                            "start": datasets.Value("int64"),
                            "end": datasets.Value("int64"),
                            "label": datasets.ClassLabel(names=["HEADER", "QUESTION", "ANSWER"]),
                        }
                    ),
                    "relations": datasets.Sequence(
                        {
                            "head": datasets.Value("int64"),
                            "tail": datasets.Value("int64"),
                            "start_index": datasets.Value("int64"),
                            "end_index": datasets.Value("int64"),
                        }
                    ),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_file = dl_manager.download_and_extract(_URL)
        downloaded_files = Neocustom_to_xfun(downloaded_file)
        train_files_for_many_langs = [downloaded_files["train"]]
        val_files_for_many_langs = [downloaded_files["val"]]

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": train_files_for_many_langs}),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepaths": val_files_for_many_langs}
            ),
            # datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepaths": test_files_for_many_langs}),
        ]

    def _generate_examples(self, filepaths):
        for filepath in filepaths:
            logger.info("Generating examples from = %s", filepath)
            with open(filepath[0], "r") as f:
                data = json.load(f)

            for doc in data["documents"]:
                doc["img"]["fpath"] = os.path.join(filepath[1], doc["img"]["fname"])
                if not os.path.exists(doc["img"]["fpath"]):
                    continue
                image, size = load_image(doc["img"]["fpath"])
                hd_image, _ = load_image(doc["img"]["fpath"], 1024)
                document = doc["document"]
                tokenized_doc = {"input_ids": [], "bbox": [], "labels": []}
                entities = []
                relations = []
                id2label = {}
                entity_id_to_index_map = {}
                empty_entity = set()
                for line in document:
                    if len(line["text"]) == 0:
                        empty_entity.add(line["id"])
                        continue
                    id2label[line["id"]] = line["label"]
                    relations.extend([tuple(sorted(l)) for l in line["linking"]])
                    tokenized_inputs = self.tokenizer(
                        [line["text"]],
                        boxes=[line["box"]],
                        add_special_tokens=False,
                        return_offsets_mapping=True,
                        return_attention_mask=False,
                    )
                    text_length = 0
                    ocr_length = 0
                    bbox = []
                    for token_id, offset in zip(tokenized_inputs["input_ids"], tokenized_inputs["offset_mapping"]):
                        if token_id == 6:
                            bbox.append(None)
                            continue
                        text_length += offset[1] - offset[0]
                        tmp_box = []
                        while ocr_length < text_length:
                            if len(line["words"]) == 0:
                                break
                            ocr_word = line["words"].pop(0)
                            ocr_length += len(
                                self.tokenizer._tokenizer.normalizer.normalize_str(ocr_word["text"].strip())
                            )
                            tmp_box.append(simplify_bbox(ocr_word["box"]))
                        if len(tmp_box) == 0:
                            tmp_box = last_box
                        bbox.append(normalize_bbox(merge_bbox(tmp_box), size))
                        last_box = tmp_box  # noqa
                    bbox = [
                        [bbox[i + 1][0], bbox[i + 1][1], bbox[i + 1][0], bbox[i + 1][1]] if b is None else b
                        for i, b in enumerate(bbox)
                    ]
                    if line["label"] == "other":
                        label = ["O"] * len(bbox)
                    else:
                        label = [f"I-{line['label'].upper()}"] * len(bbox)
                        label[0] = f"B-{line['label'].upper()}"
                    tokenized_inputs.update({"bbox": bbox, "labels": label})
                    if label[0] != "O":
                        entity_id_to_index_map[line["id"]] = len(entities)
                        entities.append(
                            {
                                "start": len(tokenized_doc["input_ids"]),
                                "end": len(tokenized_doc["input_ids"]) + len(tokenized_inputs["input_ids"]),
                                "label": line["label"].upper(),
                            }
                        )
                    for i in tokenized_doc:
                        tokenized_doc[i] = tokenized_doc[i] + tokenized_inputs[i]
                relations = list(set(relations))
                relations = [rel for rel in relations if rel[0] not in empty_entity and rel[1] not in empty_entity]
                kvrelations = []
                for rel in relations:
                    pair = [id2label[rel[0]], id2label[rel[1]]]
                    if pair == ["question", "answer"]:
                        kvrelations.append(
                            {"head": entity_id_to_index_map[rel[0]], "tail": entity_id_to_index_map[rel[1]]}
                        )
                    elif pair == ["answer", "question"]:
                        kvrelations.append(
                            {"head": entity_id_to_index_map[rel[1]], "tail": entity_id_to_index_map[rel[0]]}
                        )
                    else:
                        continue

                def get_relation_span(rel):
                    bound = []
                    for entity_index in [rel["head"], rel["tail"]]:
                        bound.append(entities[entity_index]["start"])
                        bound.append(entities[entity_index]["end"])
                    return min(bound), max(bound)

                relations = sorted(
                    [
                        {
                            "head": rel["head"],
                            "tail": rel["tail"],
                            "start_index": get_relation_span(rel)[0],
                            "end_index": get_relation_span(rel)[1],
                        }
                        for rel in kvrelations
                    ],
                    key=lambda x: x["head"],
                )
                chunk_size = 512
                for chunk_id, index in enumerate(range(0, len(tokenized_doc["input_ids"]), chunk_size)):
                    item = {}
                    for k in tokenized_doc:
                        item[k] = tokenized_doc[k][index : index + chunk_size]
                    entities_in_this_span = []
                    global_to_local_map = {}
                    for entity_id, entity in enumerate(entities):
                        if (
                            index <= entity["start"] < index + chunk_size
                            and index <= entity["end"] < index + chunk_size
                        ):
                            entity["start"] = entity["start"] - index
                            entity["end"] = entity["end"] - index
                            global_to_local_map[entity_id] = len(entities_in_this_span)
                            entities_in_this_span.append(entity)
                    relations_in_this_span = []
                    for relation in relations:
                        if (
                            index <= relation["start_index"] < index + chunk_size
                            and index <= relation["end_index"] < index + chunk_size
                        ):
                            relations_in_this_span.append(
                                {
                                    "head": global_to_local_map[relation["head"]],
                                    "tail": global_to_local_map[relation["tail"]],
                                    "start_index": relation["start_index"] - index,
                                    "end_index": relation["end_index"] - index,
                                }
                            )
                    item.update(
                        {
                            "id": f"{doc['id']}_{chunk_id}",
                            "image": image,
                            "hd_image": hd_image,
                            "entities": entities_in_this_span,
                            "relations": relations_in_this_span,
                        }
                    )
                    yield f"{doc['id']}_{chunk_id}", item
