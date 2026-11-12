import glob
import os
from dataclasses import dataclass

import pandas as pd


@dataclass
class Unit:
    name: str
    seq_a: list
    labels: list
    seq_b: list = None


def load_single_csv(path):
    df = pd.read_csv(path, names=["sequence", "label"], header=0)
    labels = [int(float(v)) for v in df["label"].tolist()]
    return df["sequence"].tolist(), labels


def load_paired_csv(path):
    df = pd.read_csv(path)
    labels = df["label"].astype(int).tolist()
    return df["enhancer"].tolist(), df["promoter"].tolist(), labels


def load_paired_fasta(left_files, right_files):
    seqs_left, seqs_right, labels = [], [], []
    for left_file, right_file in zip(left_files, right_files):
        with open(left_file) as f:
            for line in f:
                if line[0] == ">":
                    labels.append(1 if line[1] == "+" else 0)
                else:
                    seqs_left.append(line.strip())
        with open(right_file) as f:
            for line in f:
                if line[0] != ">":
                    seqs_right.append(line.strip())
    return seqs_left, seqs_right, labels


class SingleCsvReader:
    def units(self, data_dir, task_cfg):
        pattern = os.path.join(data_dir, "**", "*.csv")
        for path in sorted(glob.glob(pattern, recursive=True)):
            seqs, labels = load_single_csv(path)
            yield Unit(os.path.splitext(os.path.basename(path))[0], seqs, labels)


class PairedCsvReader:
    def units(self, data_dir, task_cfg):
        for cell in task_cfg.get("cell_lines", []):
            cell_dir = os.path.join(data_dir, cell)
            if not os.path.isdir(cell_dir):
                continue
            enh, pro, labels = [], [], []
            for split in ("train", "dev", "test"):
                path = os.path.join(cell_dir, f"{split}.csv")
                if os.path.exists(path):
                    e, p, l = load_paired_csv(path)
                    enh += e
                    pro += p
                    labels += l
            if labels:
                yield Unit(f"{cell}_all", enh, labels, seq_b=pro)


class PairedFastaReader:
    def units(self, data_dir, task_cfg):
        mapping = task_cfg.get("file_mapping", {})
        for cell in task_cfg.get("cell_lines", []):
            for pair in task_cfg.get("pair_types", []):
                spec = mapping.get(cell, {}).get(pair)
                if not spec:
                    continue
                left = [os.path.join(data_dir, cell, f) for f in spec["left"]]
                right = [os.path.join(data_dir, cell, f) for f in spec["right"]]
                left = [f for f in left if os.path.exists(f)]
                right = [f for f in right if os.path.exists(f)]
                if not left or not right:
                    continue
                seqs_left, seqs_right, labels = load_paired_fasta(left, right)
                yield Unit(f"{cell}_{pair}_all", seqs_left, labels, seq_b=seqs_right)


READERS = {
    "single_csv": SingleCsvReader,
    "paired_csv": PairedCsvReader,
    "paired_fasta": PairedFastaReader,
}


def build_reader(key):
    if key not in READERS:
        raise KeyError(f"unknown reader: {key}")
    return READERS[key]()
