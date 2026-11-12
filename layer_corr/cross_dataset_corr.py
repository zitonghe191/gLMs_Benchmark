import functools
import os

import numpy as np
import pandas as pd

import config as C
import core


def metas(*, cw_cache=C.CW_CACHE, prom_cache=C.PROM_CACHE):
    return (np.load(cw_cache, allow_pickle=True)["meta"].item(),
            np.load(prom_cache, allow_pickle=True)["meta"].item())


def plan(*, models=C.MODEL_NAMES, cw_cache=C.CW_CACHE, prom_cache=C.PROM_CACHE):
    cw, pr = metas(cw_cache=cw_cache, prom_cache=prom_cache)
    out = []
    for m in models:
        if m not in cw["models"]:
            continue
        L = cw["models"][m]["layers"]
        dim = len(L) * (len(L) - 1) // 2
        if m not in pr["models"]:
            continue
        if pr["models"][m] != L:
            continue
        if dim < 2:
            continue
        out.append(m)
    return out


@functools.lru_cache(maxsize=1)
def prom_lengths(prom_datasets, *, csv_dir=C.PROM_CSV_DIR):
    bp = {}
    for d in prom_datasets:
        L = pd.read_csv(os.path.join(csv_dir, f"{d}.csv"),
                        usecols=["sequence"])["sequence"].str.len()
        assert L.nunique() == 1, f"{d}: length not unique {sorted(L.unique())}"
        bp[d] = int(L.iloc[0])
    return bp


def load(model, *, cw_cache=C.CW_CACHE, prom_cache=C.PROM_CACHE,
         csv_dir=C.PROM_CSV_DIR):
    cw, pr = metas(cw_cache=cw_cache, prom_cache=prom_cache)
    cw_datasets = list(cw["datasets"])
    prom_datasets = list(pr["datasets"])
    layers = cw["models"][model]["layers"]
    n_layers = len(layers)
    assert model in pr["models"], f"{model}: not in promoter cache"
    assert pr["models"][model] == layers, f"{model}: layer lists disagree"
    dim = n_layers * (n_layers - 1) // 2
    assert dim >= 2, f"{model}: fingerprint only {dim}-dim, correlation undefined"

    cw_bp = {d: C.cw_len(d) for d in cw_datasets}
    prom_bp = prom_lengths(tuple(prom_datasets), csv_dir=csv_dir)

    prom_order = sorted(prom_datasets, key=lambda d: (prom_bp[d], d))
    cw_order = sorted(cw_datasets, key=lambda d: (cw_bp[d], d))

    z = np.load(cw_cache, allow_pickle=True)
    zp = np.load(prom_cache, allow_pickle=True)
    mats = {d: zp[core.cache_key(model, d)] for d in prom_order}
    mats.update({d: z[core.cache_key(model, d)] for d in cw_order})
    V = np.array([core.fingerprint(mats[d]) for d in prom_order + cw_order])
    R = core.corr_rows(V)
    R = R[np.ix_(np.arange(len(prom_order)),
                 np.arange(len(prom_order), len(prom_order) + len(cw_order)))]
    return R, prom_order, cw_order, prom_bp, cw_bp, n_layers


def load_self(model, *, prom_cache=C.PROM_CACHE, csv_dir=C.PROM_CSV_DIR,
              pairs=C.PAIRS):
    z = np.load(prom_cache, allow_pickle=True)
    pr = z["meta"].item()
    datasets = list(pr["datasets"])
    assert model in pr["models"], f"{model}: not in promoter cache"
    n_layers = len(pr["models"][model])
    dim = n_layers * (n_layers - 1) // 2
    assert dim >= 2, f"{model}: fingerprint only {dim}-dim, correlation undefined"

    bp = prom_lengths(tuple(datasets), csv_dir=csv_dir)
    order = sorted(datasets, key=lambda d: (bp[d], d))
    for a, b in pairs:
        assert a in datasets and b in datasets, f"PAIRS entry {a} / {b} missing from cache"

    V = np.array([core.fingerprint(z[core.cache_key(model, d)]) for d in order])
    R = core.corr_rows(V)
    np.fill_diagonal(R, 1.0)
    return R, order, bp, n_layers
