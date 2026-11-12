import numpy as np

import config as C
import core


def mid_layer(model, *, models=C.MODELS):
    layers = models[model]["layers"]
    return layers[len(layers) // 2]


def mid_layers(*, models=C.MODELS):
    return {m: mid_layer(m, models=models) for m in models}


def load_pos(model, ds, *, models=C.MODELS, base=C.BASE_COLDWATER, layout=C.LAYOUT):
    cfg = models[model]
    L = mid_layer(model, models=models)
    return core.load_positive_features(layout, cfg, L, ds, base)


def cka_matrix_for_dataset(ds, *, models=C.MODELS, base=C.BASE_COLDWATER,
                           layout=C.LAYOUT):
    names = list(models)
    feats = {m: load_pos(m, ds, models=models, base=base, layout=layout)
             for m in names}
    counts = {len(feats[m]) for m in names}
    assert len(counts) == 1, f"{ds}: models disagree on positive count {sorted(counts)}"
    n = len(names)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = core.linear_cka(feats[names[i]], feats[names[j]])
    return M


def cache_meta(*, models=C.MODELS, datasets=C.DATASETS, base=C.BASE_COLDWATER,
               layout=C.LAYOUT):
    return {"models": {m: {k: models[m][k] for k in ("dir", "feat", "layers", "prefix")}
                       for m in models},
            "mid_layer": mid_layers(models=models),
            "datasets": list(datasets),
            "base": base,
            "layout": [layout.feat_tmpl, layout.label_tmpl],
            "metric": "linear_cka", "samples": "pos"}


def get_matrices(refresh=False, *, models=C.MODELS, datasets=C.DATASETS,
                 base=C.BASE_COLDWATER, layout=C.LAYOUT, cache_path=C.CKA_CACHE):
    meta = cache_meta(models=models, datasets=datasets, base=base, layout=layout)
    if not refresh:
        cached = core.read_cache(cache_path, meta)
        if cached is not None:
            print("loaded cache:", cache_path)
            return cached
    out = {}
    for ds in datasets:
        print(f"computing CKA | {ds} ...", flush=True)
        out[ds] = cka_matrix_for_dataset(ds, models=models, base=base, layout=layout)
    core.write_cache(cache_path, meta, **out)
    print("saved cache:", cache_path)
    return out
