import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Layout:
    feat_tmpl: str
    label_tmpl: str = ""

    def feature_path(self, cfg, layer, dataset, base):
        return self.feat_tmpl.format(base=base, dir=cfg["dir"], feat=cfg["feat"],
                                     prefix=cfg["prefix"], layer=layer,
                                     dataset=dataset)

    def label_path(self, cfg, layer, dataset, base):
        assert self.label_tmpl, "layout has no label template"
        return self.label_tmpl.format(base=base, dir=cfg["dir"], feat=cfg["feat"],
                                      prefix=cfg["prefix"], layer=layer,
                                      dataset=dataset)


def cache_key(model, ds):
    return f"{model}__{ds}".replace("-", "_").replace(".", "_")


def cos_sim(A, B):
    An = np.linalg.norm(A, axis=1)
    Bn = np.linalg.norm(B, axis=1)
    return (A * B).sum(axis=1) / (An * Bn + 1e-12)


def linear_cka(X, Y):
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xty = X.T @ Y
    return float(np.sum(xty ** 2)
                 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y) + 1e-12))


def subsample(n, max_samples, seed):
    if max_samples is None or n <= max_samples:
        return np.arange(n)
    return np.random.RandomState(seed).choice(n, max_samples, replace=False)


def load_features(layout, cfg, layer, dataset, base):
    return np.load(layout.feature_path(cfg, layer, dataset, base))


def load_layer_features(layout, cfg, dataset, base, *, max_samples=None, seed=0):
    layers = cfg["layers"]
    arr0 = load_features(layout, cfg, layers[0], dataset, base)
    idx = subsample(len(arr0), max_samples, seed)
    return {L: load_features(layout, cfg, L, dataset, base)[idx] for L in layers}


def layer_sim_matrix(feats):
    layers = list(feats)
    n = len(layers)
    M = np.eye(n)
    for i, L1 in enumerate(layers):
        for j, L2 in enumerate(layers):
            if i < j:
                v = float(cos_sim(feats[L1], feats[L2]).mean())
                M[i, j] = M[j, i] = v
    return M


def build_matrix(layout, cfg, dataset, base, *, max_samples=None, seed=0):
    feats = load_layer_features(layout, cfg, dataset, base,
                                max_samples=max_samples, seed=seed)
    return layer_sim_matrix(feats)


def load_positive_features(layout, cfg, layer, dataset, base, *, positive=1):
    lab = np.load(layout.label_path(cfg, layer, dataset, base))
    feats = load_features(layout, cfg, layer, dataset, base)[lab == positive]
    assert len(feats), f"{cfg['dir']} {dataset} L{layer}: no positive samples"
    return feats


def load_positive_centroid(layout, cfg, layer, dataset, base, *, positive=1):
    return load_positive_features(layout, cfg, layer, dataset, base,
                                  positive=positive).mean(axis=0).astype(np.float64)


def fingerprint(M):
    n = M.shape[0]
    return M[np.triu_indices(n, 1)]


def corr_rows(V):
    V = np.asarray(V, dtype=float)
    Vz = (V - V.mean(0)) / (V.std(0) + 1e-12)
    return np.corrcoef(Vz)


def read_cache(path, meta):
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=True)
    if z["meta"].item() != meta:
        return None
    return {k: z[k] for k in z.files if k != "meta"}


def write_cache(path, meta, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, meta=np.array(meta, dtype=object), **arrays)
