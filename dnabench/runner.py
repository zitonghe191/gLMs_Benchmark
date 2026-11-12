import argparse
import glob
import os

import pandas as pd

from .config import resolve
from .env import set_seed
from .extract import extract_main
from .trainers import run_benchmark

METRIC_ORDER = [
    "accuracy", "roc_auc", "pr_auc", "f1", "f1_macro", "f1_weighted", "mcc",
    "test_mean_auc", "test_std_auc",
    "cv_mean_auc", "cv_std_auc", "best_val_auc",
    "cv_mean_mcc", "cv_std_mcc", "best_val_mcc",
]


def train_main(plan):
    set_seed()
    layer_dirs = sorted(
        glob.glob(os.path.join(plan.base_dir, "features", f"{plan.tag}_layer*")))
    rows = []
    for layer_dir in layer_dirs:
        layer = os.path.basename(layer_dir)
        for unit_dir in sorted(glob.glob(os.path.join(layer_dir, "*"))):
            if not os.path.isfile(os.path.join(unit_dir, "all_features.npy")):
                continue
            unit = os.path.basename(unit_dir)
            print(f"[train] {plan.task}/{plan.model} | {layer} | {unit}")
            result = run_benchmark(unit_dir, plan)
            result.pop("dataset", None)
            result.update({"task": plan.task, "model": plan.model, "layer": layer, "unit": unit})
            rows.append(result)

    if not rows:
        raise RuntimeError(f"no features found under {plan.base_dir}/features")

    out_path = os.path.join(plan.base_dir, "results", "mean_results.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = pd.DataFrame(rows)
    cols = ["task", "model", "layer", "unit"] + [c for c in METRIC_ORDER if c in df.columns]
    df = df[cols]
    df.to_csv(out_path, index=False)
    print(f"[train] wrote {out_path}")
    return df


def run(task, model, max_length, batch_size, train_batch_size, stage="all", base_dir=None):
    plan = resolve(task, model, max_length, batch_size, train_batch_size, base_dir)
    print(f"[run] task={task} model={model} stage={stage} base={plan.base_dir}")
    if stage in ("extract", "all"):
        extract_main(plan)
    if stage in ("train", "all"):
        train_main(plan)


def _parse_args(task, model, stage, max_length, batch_size, train_batch_size):
    parser = argparse.ArgumentParser(prog="dnabench")
    parser.add_argument("--task", required=task is None)
    parser.add_argument("--model", required=model is None)
    parser.add_argument("--stage", default=stage, choices=["extract", "train", "all"])
    parser.add_argument("--max_length", type=int, required=max_length is None)
    parser.add_argument("--batch_size", type=int, required=batch_size is None)
    parser.add_argument("--train_batch_size", type=int, required=train_batch_size is None)
    return parser.parse_known_args()[0]


def main(task=None, model=None, stage="all", base_dir=None,
         max_length=None, batch_size=None, train_batch_size=None):
    args = _parse_args(task, model, stage, max_length, batch_size, train_batch_size)
    task = task if task is not None else args.task
    model = model if model is not None else args.model
    max_length = max_length if max_length is not None else args.max_length
    batch_size = batch_size if batch_size is not None else args.batch_size
    train_batch_size = (
        train_batch_size if train_batch_size is not None else args.train_batch_size)
    run(task, model, max_length, batch_size, train_batch_size, args.stage, base_dir)
