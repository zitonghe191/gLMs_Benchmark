import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .adapters import build_adapter
from .env import set_seed
from .readers import build_reader


class SeqDataset(Dataset):
    def __init__(self, labels, encodings, uses_mask):
        self.encodings = encodings
        self.labels = labels
        self.uses_mask = uses_mask

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {"input_ids": self.encodings["input_ids"][idx], "y": self.labels[idx]}
        if self.uses_mask:
            item["attention_mask"] = self.encodings["attention_mask"][idx]
        return item


def _extract_matrix(adapter, tokenizer, model, seqs, labels, max_length, batch_size, device):
    enc = tokenizer(
        seqs,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    ds = SeqDataset(labels, enc, adapter.uses_attention_mask)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    chunks = []
    for batch in loader:
        chunks.append(adapter.pool(model, batch, device))
    return np.concatenate(chunks, axis=0)


def extract_main(plan):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = build_adapter(plan.model, plan.model_cfg)
    reader = build_reader(plan.task_cfg["reader"])
    units = list(reader.units(plan.data_dir, plan.task_cfg))
    if not units:
        raise RuntimeError(f"no data units found under {plan.data_dir}")

    print(f"[extract] {plan.task}/{plan.model} | device={device} | units={len(units)}")

    for num_layers in plan.layers:
        out_dir = os.path.join(plan.base_dir, "features", f"{adapter.tag}_layer{num_layers}")
        tokenizer = adapter.build_tokenizer(plan.max_length)
        model = adapter.build_model(num_layers, device)
        for unit in units:
            feats = _extract_matrix(
                adapter, tokenizer, model, unit.seq_a, unit.labels,
                plan.max_length, plan.batch_size, device,
            )
            if unit.seq_b is not None:
                feats_b = _extract_matrix(
                    adapter, tokenizer, model, unit.seq_b, unit.labels,
                    plan.max_length, plan.batch_size, device,
                )
                feats = np.concatenate([feats, feats_b], axis=1)
            save_dir = os.path.join(out_dir, unit.name)
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, "all_features.npy"), feats)
            np.save(os.path.join(save_dir, "all_labels.npy"), np.array(unit.labels))
            print(f"  layer{num_layers} | {unit.name} | features={feats.shape}")
        del model
        torch.cuda.empty_cache()
