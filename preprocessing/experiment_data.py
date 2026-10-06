"""Strict dataset/text preparation for reproducible experiments."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import urllib.request

import torch
from transformers import AutoTokenizer

from preprocessing.dataset import load_fb15k237, load_synthetic_toy_graph
from preprocessing.entity_text import align_entity_texts, download_entity_text_files, read_text_mapping
from preprocessing.graph_builder import assert_no_leakage, build_train_graph


@dataclass
class EntityTokens:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    has_text: torch.Tensor
    relations: EntityTokens | None = None

    def validate(self, num_entities, num_relations=None):
        if self.input_ids.ndim != 2 or self.input_ids.shape != self.attention_mask.shape:
            raise ValueError("Entity token/mask shapes must match")
        if len(self.input_ids) != num_entities or self.has_text.shape != (num_entities,):
            raise ValueError("Text rows must align exactly with dataset entity IDs")
        if self.input_ids.device.type != "cpu" or self.attention_mask.device.type != "cpu":
            raise ValueError("Store tokens on CPU; transfer only LM micro-batches")
        if self.has_text.device.type != "cpu" or self.has_text.dtype != torch.bool:
            raise ValueError("Text presence mask must be a CPU boolean tensor")
        if num_relations is not None:
            if self.relations is None:
                raise ValueError("Relation text tokens are required")
            self.relations.validate(num_relations)
        return self


def load_experiment_dataset(cfg):
    if cfg.dataset == "synthetic":
        return load_synthetic_toy_graph(seed=cfg.seed)
    dataset = load_fb15k237(cfg.raw_dir, download_if_missing=cfg.download_if_missing)
    expected = (14541, 237, 272115, 17535, 20466)
    actual = (dataset.num_entities, dataset.num_relations, len(dataset.train),
              len(dataset.valid), len(dataset.test))
    if actual != expected:
        raise ValueError(f"FB15k-237 statistics mismatch: {actual}, expected {expected}")
    return dataset


def validate_dataset(dataset):
    graph = build_train_graph(dataset.train, dataset.num_entities, dataset.num_relations)
    assert_no_leakage(graph, dataset.train, dataset.valid, dataset.test)
    for split in (dataset.train, dataset.valid, dataset.test):
        if not split:
            raise ValueError("Train, validation and test splits must all be non-empty")
        if len(set(split)) != len(split):
            raise ValueError("Duplicate triples in a split")
        if any(not (0 <= h < dataset.num_entities and 0 <= t < dataset.num_entities
                    and 0 <= r < dataset.num_relations) for h, r, t in split):
            raise ValueError("Triple ID outside vocabulary")
    if set(dataset.valid) & set(dataset.test):
        raise ValueError("Validation/test splits overlap")
    for mapping in (dataset.entity2id, dataset.relation2id):
        if sorted(mapping.values()) != list(range(len(mapping))):
            raise ValueError("Vocabulary IDs must be unique and contiguous")


RELATION_TEXT_URL = (
    "https://raw.githubusercontent.com/yao8839836/kg-bert/master/data/FB15k-237/relation2text.txt"
)


def align_relation_texts(relation2id, path):
    """Strict relation-key alignment; never infer rows from file order."""
    mapping = read_text_mapping(path)
    descriptions = [""] * len(relation2id)
    for key, index in relation2id.items():
        descriptions[index] = mapping.get(key, "").strip()
    missing = [key for key, index in relation2id.items() if not descriptions[index]]
    if missing:
        raise ValueError(f"Missing relation descriptions for {len(missing)} relations: {missing[:3]}")
    return descriptions


def prepare_text_tokens(cfg, dataset):
    if cfg.dataset == "synthetic":
        descriptions = [f"Synthetic entity {i}." for i in range(dataset.num_entities)]
        present = [True] * dataset.num_entities
        relation_descriptions = [f"Synthetic relation {i}." for i in range(dataset.num_relations)]
    else:
        if cfg.download_if_missing:
            long_path, short_path = download_entity_text_files(cfg.text_dir)
        else:
            long_path = Path(cfg.text_dir) / "FB15k_mid2description.txt"
            short_path = Path(cfg.text_dir) / "entity2text.txt"
        aligned = align_entity_texts(dataset.entity2id, long_path, short_path)
        descriptions, present = aligned.texts_by_id, aligned.has_text
        relation_path = Path(cfg.text_dir) / "relation2text.txt"
        if cfg.download_if_missing and not relation_path.exists():
            temporary = relation_path.with_suffix(".txt.tmp")
            urllib.request.urlretrieve(RELATION_TEXT_URL, temporary)
            # Validate the resource before making it the reusable local file.
            align_relation_texts(dataset.relation2id, temporary)
            temporary.replace(relation_path)
        relation_descriptions = align_relation_texts(dataset.relation2id, relation_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.lm_name)
    encoded = tokenizer(descriptions, padding="max_length", truncation=True,
                        max_length=cfg.max_text_length, return_tensors="pt")
    relation_encoded = tokenizer(relation_descriptions, padding="max_length", truncation=True,
                                 max_length=cfg.max_text_length, return_tensors="pt")
    relations = EntityTokens(relation_encoded["input_ids"], relation_encoded["attention_mask"],
                             torch.ones(dataset.num_relations, dtype=torch.bool))
    return EntityTokens(encoded["input_ids"], encoded["attention_mask"],
                        torch.tensor(present, dtype=torch.bool), relations).validate(
                            dataset.num_entities, dataset.num_relations)


def dataset_manifest(cfg, dataset, tokens=None):
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    result = {
        "dataset": cfg.dataset,
        "synthetic": cfg.dataset == "synthetic",
        "num_entities": dataset.num_entities, "num_relations": dataset.num_relations,
        "split_sizes": {s: len(getattr(dataset, s)) for s in ("train", "valid", "test")},
        "split_sha256": {s: digest(getattr(dataset, s)) for s in ("train", "valid", "test")},
        "entity_ids_sha256": digest(dataset.entity2id),
        "relation_ids_sha256": digest(dataset.relation2id),
    }
    if tokens is not None:
        result["entities_with_text"] = int(tokens.has_text.sum())
        result["entities_without_text"] = int((~tokens.has_text).sum())
        result["tokens_sha256"] = hashlib.sha256(b"".join(
            t.numpy().tobytes() for t in (tokens.input_ids, tokens.attention_mask, tokens.has_text)
        )).hexdigest()
        if tokens.relations is None:
            raise ValueError("Manifest requires relation tokens")
        result["relations_with_text"] = int(tokens.relations.has_text.sum())
        result["relation_tokens_sha256"] = hashlib.sha256(b"".join(
            t.numpy().tobytes() for t in (tokens.relations.input_ids,
                                         tokens.relations.attention_mask, tokens.relations.has_text)
        )).hexdigest()
    return result
