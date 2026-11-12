import numpy as np

import config as C
import core


def load(model, *, prom_cache=C.PROM_CACHE, blocks=C.HUMAN_BLOCKS):
    z = np.load(prom_cache, allow_pickle=True)
    pr = z["meta"].item()
    assert model in pr["models"], f"{model}: not in promoter cache"
    fp = {d: core.fingerprint(z[core.cache_key(model, d)]).astype(float)
          for d in pr["datasets"]}

    panels = []
    for title, ids in blocks:
        R = np.corrcoef(np.array([fp[d] for d in ids]))
        assert np.isfinite(R).all(), f"{title}: has NaN"
        panels.append((title, [C.short_prom(d) for d in ids], R))
    return panels
