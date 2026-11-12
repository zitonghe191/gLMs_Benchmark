#!/usr/bin/env python3

import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(ROOT, "result")
FIG_DIR = os.path.join(ROOT, "figures")

ORDER = ["hyena128", "hyena256", "hyenasmall", "gena", "grover",
         "ntv2-50m", "ntv2-250m", "dnabert2"]
STYLE = {
    "hyena128": dict(label="HyenaDNA-128", short="HyenaDNA-128", color="#d62728", marker="o"),
    "hyena256": dict(label="HyenaDNA-256", short="HyenaDNA-256", color="#ff7f0e", marker="s"),
    "hyenasmall": dict(label="HyenaDNA-small", short="HyenaDNA-small", color="#8c564b", marker="^"),
    "gena": dict(label="GENA-LM", short="GENA-LM", color="#2ca02c", marker="D"),
    "grover": dict(label="GROVER", short="GROVER", color="#bcbd22", marker="v"),
    "ntv2-50m": dict(label="NTv2-50M", short="NTv2-50M", color="#1f77b4", marker=">"),
    "ntv2-250m": dict(label="NTv2-250M", short="NTv2-250M", color="#17becf", marker="<"),
    "dnabert2": dict(label="DNABERT-2", short="DNABERT-2", color="#9467bd", marker="P"),
}
NT_TICKS = [64, 256, 1024, 4096, 16384, 65536]
LABEL_OFFSET = {
    "HyenaDNA-128": (4, 4), "HyenaDNA-256": (2, 9), "HyenaDNA-small": (4, -13),
    "GENA-LM": (4, 4), "GROVER": (4, 2), "NTv2-50M": (4, 5),
    "NTv2-250M": (-6, 6), "DNABERT-2": (4, 4),
}
LABEL_HA = {"NTv2-250M": "right"}
PANEL_OFFSETS = {
    (512, "HyenaDNA-128"): (4, 10),
    (512, "NTv2-50M"): (4, -11),
}
PANEL_LABELS = {
    (512, "GROVER"): (0.3, 2.8),
    (512, "GENA-LM"): (0.3, 3.6),
}


def nt_label(t):
    return f"{int(t):,}"


def setup():
    plt.rcParams.update({
        "font.family": "Arial", "font.sans-serif": ["Arial"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 120, "savefig.bbox": "tight",
    })
    os.makedirs(FIG_DIR, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name + ".pdf"))
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=300)
    plt.close(fig)
    print("wrote", os.path.join(FIG_DIR, name + ".{pdf,png}"))


def load():
    frames = []
    for path in sorted(glob.glob(os.path.join(RESULT_DIR, "*.csv"))):
        df = pd.read_csv(path)
        if len(df) and df["model"].iloc[0] in STYLE:
            frames.append(df)
    if not frames:
        raise SystemExit(f"no result CSVs in {RESULT_DIR}")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["model"].isin(ORDER)]
    return df


def aggregate(df):
    fwd = df[(df["stage"] == "forward") & (df["status"] == "ok")].copy()
    fwd["n_nt"] = fwd["n_nt"].astype(int)
    agg = fwd.groupby(["model", "n_nt"]).agg(
        lat=("latency_ms", "median"),
        act=("act_peak_mb", "median"),
        peak=("peak_alloc_mb", "median"),
        toks=("n_tokens", "median"),
    ).reset_index()
    return fwd, agg


def end_notes(df):
    out = {}
    for model in ORDER:
        fwd = df[(df["model"] == model) & (df["stage"] == "forward")
                 & (df["status"] == "ok")]
        if not len(fwd):
            continue
        end = df[(df["model"] == model)
                 & ((df["stage"] == "stop") | (df["status"] == "oom"))]
        kind, at, cap = "", None, None
        if len(end):
            row = end.iloc[0]
            at = int(row["n_nt"])
            kind = "OOM" if row["status"] == "oom" else "length cap"
            m = re.search(r"nt_cap=(\d+)", str(row.get("error", "")))
            cap = int(m.group(1)) if m else None
        out[model] = dict(max_nt=int(fwd["n_nt"].max()), end_kind=kind,
                          end_nt=at, cap=cap)
    return out


def plot_latency(agg):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for model in ORDER:
        g = agg[agg["model"] == model].sort_values("n_nt")
        if g.empty:
            continue
        s = STYLE[model]
        ax.plot(g["n_nt"], g["lat"], marker=s["marker"], ms=4, lw=1.4,
                color=s["color"], label=s["label"], alpha=.9, ls=s.get("ls", "-"))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(NT_TICKS)
    ax.set_xticklabels([nt_label(t) for t in NT_TICKS])
    ax.set_xlabel("Sequence length (nt)")
    ax.set_ylabel("Median latency (ms, batch=1)")
    ax.grid(True, which="major", alpha=.3, lw=.5)
    ax.legend(ncol=2, frameon=False, loc="upper left")
    save(fig, "fig1_latency_vs_length")


def plot_memory(agg):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharex=True)
    for ax, col, title in ((axes[0], "act", "Activation peak"),
                           (axes[1], "peak", "Total CUDA peak (weights+inputs+activations)")):
        for model in ORDER:
            g = agg[agg["model"] == model].sort_values("n_nt")
            if g.empty:
                continue
            s = STYLE[model]
            ax.plot(g["n_nt"], g[col], marker=s["marker"], ms=4, lw=1.4,
                    color=s["color"], label=s["label"], alpha=.9,
                    ls=s.get("ls", "-"))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(NT_TICKS)
        ax.set_xticklabels([nt_label(t) for t in NT_TICKS])
        ax.set_xlabel("Sequence length (nt)")
        ax.set_ylabel("GPU memory (MB)")
        ax.set_title(title, fontsize=9)
        ax.grid(True, which="major", alpha=.3, lw=.5)
    axes[0].legend(ncol=1, frameon=False, loc="upper left")
    save(fig, "fig2_memory_vs_length")


def plot_coverage(df, agg, notes):
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ys = []
    for i, model in enumerate(ORDER):
        if model not in notes:
            continue
        g = agg[agg["model"] == model]
        lo, hi = int(g["n_nt"].min()), notes[model]["max_nt"]
        s = STYLE[model]
        ax.hlines(i, lo, hi, color=s["color"], lw=3, alpha=.85)
        ax.plot([lo, hi], [i, i], marker=s["marker"], ms=5,
                color=s["color"], ls="none")
        txt = f"{hi:,} nt"
        if notes[model]["end_kind"] == "OOM":
            txt += f"  (OOM at {notes[model]['end_nt']:,})"
        elif notes[model]["cap"]:
            txt += f"  (cap {notes[model]['cap']:,})"
        elif notes[model]["end_kind"] == "length cap":
            txt += f"  (cap {notes[model]['end_nt']:,})"
        ax.annotate(txt, (hi, i), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7)
        ys.append(i)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([STYLE[m]["label"] for m in ORDER if m in notes])
    ax.set_xscale("log", base=2)
    ax.set_xticks(NT_TICKS)
    ax.set_xticklabels([nt_label(t) for t in NT_TICKS])
    ax.set_xlabel("Sequence length (nt)")
    ax.set_ylim(-0.7, len(ORDER) - 0.3)
    ax.invert_yaxis()
    ax.grid(True, axis="x", which="major", alpha=.3, lw=.5)
    ax.set_title("Evaluated span per model (dots = first/last tested length)", fontsize=9)
    save(fig, "figS0_coverage")


def plot_pareto(agg):
    lengths = [64, 512, 2048, 8192]
    fig, axes = plt.subplots(1, 4, figsize=(18.0, 4.2))
    for ax, nt in zip(axes, lengths):
        g = agg[agg["n_nt"] == nt]
        for _, row in g.iterrows():
            s = STYLE[row["model"]]
            ax.scatter(row["lat"], row["act"], s=34, color=s["color"],
                       marker=s["marker"], zorder=3)
            ovr = PANEL_LABELS.get((nt, s["short"]))
            if ovr:
                ax.annotate(s["short"], (row["lat"], row["act"]),
                            xytext=ovr, textcoords="data", fontsize=8.5,
                            va="center",
                            arrowprops=dict(arrowstyle="-", lw=.6,
                                            color="0.45", shrinkA=0, shrinkB=3))
            else:
                off = PANEL_OFFSETS.get((nt, s["short"]),
                                        LABEL_OFFSET.get(s["short"], (4, 4)))
                ax.annotate(s["short"], (row["lat"], row["act"]),
                            xytext=off, textcoords="offset points",
                            fontsize=8.5,
                            ha=LABEL_HA.get(s["short"], "left"))
        ax.set_title(f"{nt_label(nt)} nt", fontsize=10.5)
        ax.margins(0.18)
        ax.grid(True, which="major", alpha=.3, lw=.5)
    for ax in axes:
        ax.set_xlabel("Median latency (ms)")
    axes[0].set_ylabel("Activation peak (MB)")
    save(fig, "figS3_pareto_64_512_2048_8192")


def plot_token_ratio(df, agg):
    rows = []
    for model in ORDER:
        g = agg[agg["model"] == model]
        if g.empty:
            continue
        r = g.loc[g["n_nt"].idxmax()]
        rows.append((model, r["n_nt"] / r["toks"], r["toks"], r["n_nt"]))
    if not rows:
        return
    models = [r[0] for r in rows]
    ratios = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    y = np.arange(len(models))
    ax.barh(y, ratios, color=[STYLE[m]["color"] for m in models], alpha=.85)
    for yi, (m, ratio, toks, nt) in zip(y, rows):
        ax.text(ratio + .05, yi, f"{ratio:.2f} ({int(nt):,} nt, {int(toks):,} tokens)",
                va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels([STYLE[m]["short"] for m in models])
    ax.invert_yaxis()
    ax.set_xlabel("Nucleotides per token (at longest tested length)")
    ax.set_xlim(0, max(ratios) * 2.1)
    save(fig, "figS4_tokenizer_ratio")


def main():
    setup()
    df = load()
    _, agg = aggregate(df)
    notes = end_notes(df)
    plot_latency(agg)
    plot_memory(agg)
    plot_coverage(df, agg, notes)
    plot_pareto(agg)
    plot_token_ratio(df, agg)


if __name__ == "__main__":
    main()
