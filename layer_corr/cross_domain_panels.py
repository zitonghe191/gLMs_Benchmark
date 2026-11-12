import numpy as np

import config as C
import core


def load(model, *, prom_cache=C.PROM_CACHE, cw_cache=C.CW_CACHE,
         blocks=C.MIXED_BLOCKS, clim=C.CLIM, vmax=C.VMAX):
    z = np.load(prom_cache, allow_pickle=True)
    zp = np.load(cw_cache, allow_pickle=True)
    pr, cw = z["meta"].item(), zp["meta"].item()
    assert model in pr["models"], f"{model}: not in promoter cache"
    layers = cw["models"][model]["layers"]
    assert pr["models"][model] == layers, f"{model}: layer lists disagree"

    fp = {}
    for _, rows, cols in blocks:
        for d in rows:
            assert d in pr["datasets"], f"{d}: not in promoter cache"
            fp[d] = core.fingerprint(z[core.cache_key(model, d)]).astype(float)
        for d in cols:
            assert d in cw["datasets"], f"{d}: not in cold_water cache"
            fp[d] = core.fingerprint(zp[core.cache_key(model, d)]).astype(float)

    panels = []
    for title, rows, cols in blocks:
        V = np.array([fp[d] for d in rows + cols])
        R = np.corrcoef(V)[:len(rows), len(rows):]
        assert np.isfinite(R).all(), f"{title}: has NaN"
        rl = [C.short_mixed(d) for d in rows]
        cl = [C.cw_display(d) for d in cols]
        panels.append((title, rl, cl, R))

    off = np.concatenate([R.ravel() for _, _, _, R in panels])
    vmin = clim[model][0]
    assert off.min() >= vmin and off.max() <= vmax, \
        f"{model}: measured {off.min():.6f}..{off.max():.6f} outside clim [{vmin}, {vmax}]"
    return panels
