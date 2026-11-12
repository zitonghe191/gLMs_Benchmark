import os

import numpy as np

import config as C
import core


def centroid_path(model, *, cache_dir=C.CACHE_DIR):
    return os.path.join(cache_dir, f"centroids_{model}.npz".replace("-", "_"))


def centroid_meta(model, *, models=C.MODELS_PROM):
    cfg = models[model]
    return {"model": model, "dir": cfg["dir"], "prefix": cfg["prefix"],
            "layers": cfg["layers"], "samples": "pos"}


def get_centroids(model, datasets, refresh=False, *, models=C.MODELS_PROM,
                  base=C.BASE_PROMOTER, layout=C.LAYOUT, cache_dir=C.CACHE_DIR):
    cfg = models[model]
    path = centroid_path(model, cache_dir=cache_dir)
    meta = centroid_meta(model, models=models)
    cached = {} if refresh else (core.read_cache(path, meta) or {})

    missing = [(ds, L) for ds in datasets for L in cfg["layers"]
               if f"{ds}__L{L}" not in cached]
    if not missing:
        print(f"loaded centroid cache: {path}  ({len(cached)} vectors)")
        return {(ds, L): cached[f"{ds}__L{L}"]
                for ds in datasets for L in cfg["layers"]}

    for ds, L in missing:
        cached[f"{ds}__L{L}"] = core.load_positive_centroid(
            layout, cfg, L, ds, base)
    core.write_cache(path, meta, **cached)
    print(f"centroid cache: computed {len(missing)} new -> {path}  ({len(cached)} total)")
    return {(ds, L): cached[f"{ds}__L{L}"]
            for ds in datasets for L in cfg["layers"]}


def same_layer_cos(cents, a, b, layers):
    A = np.array([cents[(a, L)] for L in layers])
    B = np.array([cents[(b, L)] for L in layers])
    return core.cos_sim(A, B)


def centered_same_layer(cents, a, b, layers):
    A = np.array([cents[(a, L)] for L in layers])
    B = np.array([cents[(b, L)] for L in layers])
    return core.cos_sim(A - A.mean(0), B - B.mean(0))
