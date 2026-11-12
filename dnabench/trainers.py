import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
CV_SPLITS = 5
TEST_SIZE = 0.2
EPOCHS = 20
PATIENCE = 3
LR = 1e-3


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes=1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        if num_classes == 1:
            layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


def load_features(features_dir):
    x = np.load(os.path.join(features_dir, "all_features.npy"))
    y = np.load(os.path.join(features_dir, "all_labels.npy"))
    return x, y


def _binary_fold(train_X, train_y, val_X, val_y, hidden_dims, device, batch_size):
    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_X, dtype=torch.float32),
                      torch.tensor(train_y, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(val_X, dtype=torch.float32),
                      torch.tensor(val_y, dtype=torch.float32)),
        batch_size=batch_size)

    model = MLPClassifier(train_X.shape[1], hidden_dims, 1).to(device)
    optimizer = optim.Adam(model.mlp.parameters(), lr=LR)
    criterion = nn.BCELoss()

    best_auc, best_weights, counter = 0, None, 0
    for _ in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x.to(device)).view(-1), y.to(device))
            loss.backward()
            optimizer.step()

        model.eval()
        probs, targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                probs.extend(model(x.to(device)).view(-1).cpu().numpy())
                targets.extend(y.numpy())
        val_auc = roc_auc_score(targets, probs)

        if val_auc > best_auc:
            best_auc, counter = val_auc, 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            counter += 1
        if counter >= PATIENCE:
            break

    del model
    torch.cuda.empty_cache()
    return best_weights, best_auc


def _binary_eval(weights, test_X, test_y, hidden_dims, device, batch_size):
    loader = DataLoader(
        TensorDataset(torch.tensor(test_X, dtype=torch.float32),
                      torch.tensor(test_y, dtype=torch.float32)),
        batch_size=batch_size)
    model = MLPClassifier(test_X.shape[1], hidden_dims, 1).to(device)
    model.load_state_dict(weights)
    model.eval()

    probs, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            probs.extend(model(x.to(device)).view(-1).cpu().numpy())
            targets.extend(y.numpy())
    probs, targets = np.array(probs), np.array(targets)
    preds = (probs > 0.5).astype(int)
    return {
        "accuracy": accuracy_score(targets, preds),
        "roc_auc": roc_auc_score(targets, probs),
        "pr_auc": average_precision_score(targets, probs),
        "mcc": matthews_corrcoef(targets, preds),
        "f1": f1_score(targets, preds),
    }


def _multi_fold(train_X, train_y, val_X, val_y, hidden_dims, num_classes, device, batch_size):
    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_X, dtype=torch.float32),
                      torch.tensor(train_y, dtype=torch.long)),
        batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(val_X, dtype=torch.float32),
                      torch.tensor(val_y, dtype=torch.long)),
        batch_size=batch_size)

    model = MLPClassifier(train_X.shape[1], hidden_dims, num_classes).to(device)
    optimizer = optim.Adam(model.mlp.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_mcc, best_weights, counter = -1, None, 0
    for _ in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()

        model.eval()
        probs, targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                out = model(x.to(device)).cpu().numpy()
                probs.extend(torch.softmax(torch.tensor(out), dim=1).numpy())
                targets.extend(y.numpy())
        probs, targets = np.array(probs), np.array(targets)
        val_mcc = matthews_corrcoef(targets, np.argmax(probs, axis=1))

        if val_mcc > best_mcc:
            best_mcc, counter = val_mcc, 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            counter += 1
        if counter >= PATIENCE:
            break

    del model
    torch.cuda.empty_cache()
    return best_weights, best_mcc


def _macro_auc(targets, probs):
    present = np.unique(targets)
    binary = label_binarize(targets, classes=present)
    try:
        return roc_auc_score(binary, probs[:, present], multi_class="ovr", average="macro")
    except ValueError:
        return 0.0


def _multi_eval(weights, test_X, test_y, hidden_dims, num_classes, device, batch_size):
    loader = DataLoader(
        TensorDataset(torch.tensor(test_X, dtype=torch.float32),
                      torch.tensor(test_y, dtype=torch.long)),
        batch_size=batch_size)
    model = MLPClassifier(test_X.shape[1], hidden_dims, num_classes).to(device)
    model.load_state_dict(weights)
    model.eval()

    probs, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device)).cpu().numpy()
            probs.extend(torch.softmax(torch.tensor(out), dim=1).numpy())
            targets.extend(y.numpy())
    probs, targets = np.array(probs), np.array(targets)
    preds = np.argmax(probs, axis=1)
    return {
        "accuracy": accuracy_score(targets, preds),
        "roc_auc": _macro_auc(targets, probs),
        "f1_macro": f1_score(targets, preds, average="macro"),
        "f1_weighted": f1_score(targets, preds, average="weighted"),
        "mcc": matthews_corrcoef(targets, preds),
    }


def run_benchmark(features_dir, plan):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_X, all_y = load_features(features_dir)
    hidden_dims = plan.hidden_dims
    batch_size = plan.train_batch_size
    multi = plan.classifier == "multiclass"

    train_val_X, test_X, train_val_y, test_y = train_test_split(
        all_X, all_y, test_size=TEST_SIZE, random_state=SEED, stratify=all_y)

    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
    fold_weights, fold_scores = [], []
    for train_idx, val_idx in skf.split(train_val_X, train_val_y):
        if multi:
            weights, score = _multi_fold(
                train_val_X[train_idx], train_val_y[train_idx],
                train_val_X[val_idx], train_val_y[val_idx],
                hidden_dims, plan.num_classes, device, batch_size)
        else:
            weights, score = _binary_fold(
                train_val_X[train_idx], train_val_y[train_idx],
                train_val_X[val_idx], train_val_y[val_idx],
                hidden_dims, device, batch_size)
        fold_weights.append(weights)
        fold_scores.append(score)

    test_metrics = []
    for weights in fold_weights:
        if multi:
            test_metrics.append(_multi_eval(
                weights, test_X, test_y, hidden_dims, plan.num_classes, device, batch_size))
        else:
            test_metrics.append(_binary_eval(
                weights, test_X, test_y, hidden_dims, device, batch_size))

    result = {"dataset": os.path.basename(features_dir)}
    for key in test_metrics[0]:
        result[key] = float(np.mean([m[key] for m in test_metrics]))
    result["test_mean_auc"] = result["roc_auc"]
    result["test_std_auc"] = float(np.std([m["roc_auc"] for m in test_metrics]))
    if multi:
        result["cv_mean_mcc"] = float(np.mean(fold_scores))
        result["cv_std_mcc"] = float(np.std(fold_scores))
        result["best_val_mcc"] = float(max(fold_scores))
    else:
        result["cv_mean_auc"] = float(np.mean(fold_scores))
        result["cv_std_auc"] = float(np.std(fold_scores))
        result["best_val_auc"] = float(max(fold_scores))
    return result
