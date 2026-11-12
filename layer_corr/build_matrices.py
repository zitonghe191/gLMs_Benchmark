import os

import config as C
import core


def cache_meta(*, models=C.MODELS, datasets=C.DATASETS, base=C.BASE_COLDWATER,
               layout=C.LAYOUT, seed=C.SEED, max_samples=C.MAX_SAMPLES):
    return {"models": {k: {kk: v[kk] for kk in ("dir", "feat", "layers", "prefix")}
                       for k, v in models.items()},
            "datasets": list(datasets), "base": base,
            "layout": [layout.feat_tmpl, layout.label_tmpl],
            "seed": seed, "max_samples": max_samples}


def build_cache(*, models=C.MODELS, datasets=C.DATASETS, base=C.BASE_COLDWATER,
                layout=C.LAYOUT, max_samples=C.MAX_SAMPLES, seed=C.SEED,
                cache_path=C.CW_CACHE):
    mats = {}
    for model, cfg in models.items():
        for ds in datasets:
            print(f"computing {model} | {ds} ...", flush=True)
            mats[core.cache_key(model, ds)] = core.build_matrix(
                layout, cfg, ds, base, max_samples=max_samples, seed=seed)
    meta = cache_meta(models=models, datasets=datasets, base=base, layout=layout,
                      seed=seed, max_samples=max_samples)
    core.write_cache(cache_path, meta, **mats)
    return mats


def get_matrices(refresh=False, *, models=C.MODELS, datasets=C.DATASETS,
                 base=C.BASE_COLDWATER, layout=C.LAYOUT,
                 max_samples=C.MAX_SAMPLES, seed=C.SEED, cache_path=C.CW_CACHE):
    meta = cache_meta(models=models, datasets=datasets, base=base, layout=layout,
                      seed=seed, max_samples=max_samples)
    if not refresh:
        cached = core.read_cache(cache_path, meta)
        if cached is not None:
            print("loaded cache:", cache_path)
            return cached
        if os.path.exists(cache_path):
            print("cache meta changed -> recomputing")
    mats = build_cache(models=models, datasets=datasets, base=base, layout=layout,
                       max_samples=max_samples, seed=seed, cache_path=cache_path)
    print("saved cache:", cache_path)
    return mats
