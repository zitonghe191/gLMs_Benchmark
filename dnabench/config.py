import os
from dataclasses import dataclass, field

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(ROOT, "configs")


@dataclass
class Plan:
    task: str
    model: str
    base_dir: str
    task_cfg: dict
    model_cfg: dict
    tag: str
    data_dir: str
    layers: list
    max_length: int
    batch_size: int
    train_batch_size: int
    classifier: str
    num_classes: int
    hidden_dims: list


def _load(name):
    with open(os.path.join(CONFIG_DIR, name)) as f:
        return yaml.safe_load(f)


def load_all():
    return _load("models.yaml"), _load("tasks.yaml")


def resolve(task, model, max_length, batch_size, train_batch_size, base_dir=None):
    models, tasks = load_all()
    if model not in models:
        raise KeyError(f"unknown model: {model}")
    if task not in tasks:
        raise KeyError(f"unknown task: {task}")

    model_cfg = models[model]
    task_cfg = tasks[task]
    overrides = (task_cfg.get("model_overrides") or {}).get(model, {})

    def pick(key, default=None):
        for source in (overrides, task_cfg, model_cfg):
            if key in source:
                return source[key]
        return default

    base_dir = base_dir or os.path.join(ROOT, task, model)
    return Plan(
        task=task,
        model=model,
        base_dir=os.path.abspath(base_dir),
        task_cfg=task_cfg,
        model_cfg=model_cfg,
        tag=model_cfg["tag"],
        data_dir=os.path.join(os.path.abspath(base_dir), task_cfg["data_dir"]),
        layers=pick("layers"),
        max_length=max_length,
        batch_size=batch_size,
        train_batch_size=train_batch_size,
        classifier=task_cfg.get("classifier", "binary"),
        num_classes=task_cfg.get("num_classes", 1),
        hidden_dims=pick("hidden_dims"),
    )
