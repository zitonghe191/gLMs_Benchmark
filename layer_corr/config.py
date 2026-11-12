import os
import re

import core

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cache")
CW_CACHE = os.path.join(CACHE_DIR, "cross_model_grid.npz")
PROM_CACHE = os.path.join(CACHE_DIR, "prom_cross_model_grid.npz")
CKA_CACHE = os.path.join(CACHE_DIR, "cross_network_cka.npz")

BASE_COLDWATER = "/path/to/your/data"
BASE_PROMOTER = "/path/to/your/promoter"
PROM_CSV_DIR = "/path/to/your/promoter_csv"

LAYOUT = core.Layout(
    feat_tmpl="{base}/{dir}/{feat}/{prefix}{layer}/{dataset}/all_features.npy",
    label_tmpl="{base}/{dir}/{feat}/{prefix}{layer}/{dataset}/all_labels.npy")

MODELS = {
    "DNABERT2":   {"dir": "DNABERT2",   "feat": "features-dnabert2", "label": "DNABERT2",       "layers": [2, 4, 6, 8, 10, 12], "prefix": "dnabert2_layer"},
    "GENA":       {"dir": "GENA",       "feat": "features-GENA",     "label": "GENA",           "layers": [2, 4, 6, 8, 10, 12], "prefix": "gena_layer"},
    "GROVER":     {"dir": "GROVER",     "feat": "features-grover",   "label": "GROVER",         "layers": [2, 4, 6, 8, 10, 12], "prefix": "grover_layer"},
    "NTv2-50m":   {"dir": "NTv2-50m",   "feat": "features-50m",      "label": "NTv2-50m",       "layers": [2, 4, 6, 8, 10, 12], "prefix": "ntv2_layer"},
    "NTv2-250m":  {"dir": "NTv2-250m",  "feat": "features-250m",     "label": "NTv2-250m",
                   "layers": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24], "prefix": "ntv2250m_layer"},
    "hyena32k":   {"dir": "hyena32k",   "feat": "features",          "label": "HyenaDNA-small", "layers": [1, 2, 3, 4],         "prefix": "hyenasmall_layer"},
    "hyena1k256": {"dir": "hyena1k256", "feat": "features",          "label": "HyenaDNA-256",   "layers": [1, 2],               "prefix": "hyenadna256_layer"},
    "hyena1k128": {"dir": "hyena1k128", "feat": "features",          "label": "HyenaDNA-128",   "layers": [1, 2],               "prefix": "hyenadna_layer"},
}

MODELS_PROM = {
    "DNABERT2":       {"dir": "DNABERT2",       "feat": "features", "label": "DNABERT2",       "layers": [2, 4, 6, 8, 10, 12], "prefix": "dnabert2_layer"},
    "GENA":           {"dir": "GENA",           "feat": "features", "label": "GENA-LM",        "layers": [2, 4, 6, 8, 10, 12], "prefix": "gena_layer"},
    "GROVER":         {"dir": "GROVER",         "feat": "features", "label": "GROVER",         "layers": [2, 4, 6, 8, 10, 12], "prefix": "grover_layer"},
    "NTv2-50m":       {"dir": "NTv2-50m",       "feat": "features", "label": "NTv2-50m",       "layers": [2, 4, 6, 8, 10, 12], "prefix": "ntv2_layer"},
    "NTv2-250m":      {"dir": "ntv2250m",       "feat": "features", "label": "NTv2-250m",
                       "layers": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24], "prefix": "ntv2250m_layer"},
    "hyena32k":       {"dir": "hyena32k",       "feat": "features", "label": "HyenaDNA-small", "layers": [1, 2, 3, 4],         "prefix": "hyenasmall_layer"},
    "hyena1k256":     {"dir": "hyena1k256",     "feat": "features", "label": "HyenaDNA-256",   "layers": [1, 2],               "prefix": "hyenadna256_layer"},
    "hyenaDNAtiny1k": {"dir": "hyenaDNAtiny1k", "feat": "features", "label": "HyenaDNA-128",   "layers": [1, 2],               "prefix": "hyenadna_layer"},
}

BUILD_PRESETS = {
    "coldwater": {"models": MODELS, "base": BASE_COLDWATER, "cache_path": CW_CACHE},
    "promoter": {"models": MODELS_PROM, "base": BASE_PROMOTER, "cache_path": PROM_CACHE},
}

MODEL_NAMES = ["DNABERT2", "GENA", "GROVER", "NTv2-50m", "NTv2-250m",
               "hyena32k", "hyena1k256", "hyena1k128"]

DATASETS = ["A_40bp", "A_70bp", "random_40bp_1", "random_40bp_2",
            "random_70bp_1", "random_70bp_2"]

HUMAN_BLOCKS = [
    ("Human 70 bp", ["prom_human_promoter_all_70bps",
                     "prom_human_promoter_notata_70bps",
                     "prom_human_promoter_tata_70bps"]),
    ("Human 300 bp", ["prom_human_promoter_all_300bps",
                      "prom_human_promoter_notata_300bps",
                      "prom_human_promoter_tata_300bps"]),
]

MIXED_BLOCKS = [
    ("40 bp", ["Promoter_B_amyloliquefaciens",
               "Promoter_R_capsulatus"],
              ["A_40bp", "random_40bp_1", "random_40bp_2"]),
    ("70 bp", ["prom_human_promoter_all_70bps",
               "prom_human_promoter_notata_70bps",
               "prom_human_promoter_tata_70bps"],
              ["A_70bp", "random_70bp_1", "random_70bp_2"]),
]

PAIRS = [("prom_human_promoter_all_70bps", "prom_human_promoter_all_300bps"),
         ("prom_human_promoter_notata_70bps", "prom_human_promoter_notata_300bps"),
         ("prom_human_promoter_tata_70bps", "prom_human_promoter_tata_300bps")]

MAX_SAMPLES = 50_000
SEED = 42
CLIM = {"GROVER": (0.985, 1.0), "hyena32k": (0.935, 1.0)}
VMAX = 1.0


def cw_display(name):
    return name.replace("random", "Random")


def cw_len(name):
    return int(re.search(r"(\d+)bp", name).group(1))


def clean_ds_name(name):
    if name.startswith("prom_"):
        name = name[5:]
    name = name.replace("_promoter", "")
    name = name.replace("_notata", "_nonTATA").replace("_tata", "_TATA")
    return name[0].upper() + name[1:] if name else name


def prom_short(name):
    n = clean_ds_name(name)
    return n[len("Promoter_"):] if n.startswith("Promoter_") else n


def short_prom(d):
    s = prom_short(d).replace("Human_", "")
    for suf in ("_70bps", "_300bps"):
        s = s.replace(suf, "")
    return "All" if s == "all" else s


def short_mixed(d):
    if d.startswith("prom_"):
        s = prom_short(d).replace("Human_", "")
        for suf in ("_70bps", "_300bps"):
            s = s.replace(suf, "")
        return "All" if s == "all" else s
    return prom_short(d)
