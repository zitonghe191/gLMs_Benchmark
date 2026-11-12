#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import os
import random
import re
import resource
import sys
import time

import torch
import transformers
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SEED = 42
DEFAULT_LENGTHS = [2 ** k for k in range(6, 20)]
DEFAULT_MAX_NT = 2 ** 17  # 131072 nt: hard ceiling for every model

MODELS = {
    "hyena128": dict(kind="hyena", dir="HyenaDNA-128", package="hyena",
                     model_id="LongSafari/hyenadna-tiny-1k-seqlen-hf",
                     module="extract_hyenadna_layers.py",
                     extractor="HyenaDNA_FeatureExtractor", backbone="model"),
    "hyena256": dict(kind="hyena", dir="HyenaDNA-256", package="hyena",
                     model_id="LongSafari/hyenadna-tiny-1k-seqlen-d256-hf",
                     module="extract_hyenadna_layers.py",
                     extractor="HyenaDNA_FeatureExtractor", backbone="model"),
    "hyenasmall": dict(kind="hyena", dir="HyenaDNA-small", package="hyenasmall",
                       model_id="LongSafari/hyenadna-small-32k-seqlen-hf",
                       module="extract_hyenadna_layers.py",
                       extractor="HyenaDNA_FeatureExtractor", backbone="model"),
    "dnabert2": dict(kind="dnabert2", dir="DNABERT-2",
                     module="extract_dnabert2_layers.py",
                     extractor="DNABERT2_FeatureExtractor", backbone="bert",
                     model_id="zhihan1996/DNABERT-2-117M", layers=12,
                     trust_remote=True, pad_to_unk=False, unlimited=True),
    "gena": dict(kind="gena", dir="GENA-LM",
                 module="extract_gena2023_layers.py",
                 extractor="GENA_FeatureExtractor", backbone="backbone",
                 model_id="AIRI-Institute/gena-lm-bert-base", layers=12,
                 trust_remote=False, pad_to_unk=True),
    "grover": dict(kind="grover", dir="GROVER",
                   module="extract_grover_layers.py",
                   extractor="NTV2_FeatureExtractor", backbone="backbone.bert",
                   model_id="PoetschLab/GROVER", layers=12,
                   trust_remote=True, pad_to_unk=False),
    "ntv2-50m": dict(kind="ntv2", dir="NTv2-50m",
                     module="extract_ntv2-50m_layers.py",
                     extractor="NTV2_FeatureExtractor", backbone="backbone.esm",
                     model_id="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
                     layers=12, trust_remote=True, pad_to_unk=False),
    "ntv2-250m": dict(kind="ntv2", dir="NTv2-250m",
                      module="extract_ntv2_layers.py",
                      extractor="NTV2_FeatureExtractor", backbone="backbone.esm",
                      model_id="InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
                      layers=24, trust_remote=True, pad_to_unk=False),
}

FIELDS = [
    "run_id", "timestamp", "seed", "model", "stage", "repeat", "status",
    "n_nt", "n_tokens", "batch_size", "latency_ms",
    "base_alloc_mb", "peak_alloc_mb", "act_peak_mb", "peak_reserved_mb",
    "weight_mb", "host_rss_mb", "n_params", "load_s",
    "device", "gpu_index", "gpu_name", "vram_total_mb",
    "model_revision", "transformers_version", "dtype", "attn_impl",
    "tf32_cudnn", "tf32_matmul", "cudnn_benchmark", "matmul_precision",
    "torch_version", "cuda_version", "error",
]


def load_module(path, name):
    name = re.sub(r"\W", "_", name)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve_attr(obj, path):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def host_rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024.0 if sys.platform != "darwin" else rss / (1024.0 * 1024.0)


def random_seq(nt, seed):
    return "".join(random.Random(seed + nt).choices("ACGT", k=nt))


def build(name, spec, device):
    model_dir = os.path.join(ROOT, spec["dir"])
    if spec["kind"] == "hyena":
        sys.path.insert(0, model_dir)
    if spec["kind"] == "gena":
        # The extract script resolves its custom modeling_bert.py from this repo.
        os.environ["GENA_MODEL_ID"] = spec["model_id"]
    # The extract scripts force the pure-PyTorch attention path; mirror that so
    # DNABERT-2 does not switch to flash-attn/triton on a machine that has it.
    sys.modules["triton"] = None
    sys.modules["triton.language"] = None
    mod = load_module(os.path.join(model_dir, spec["module"]), f"_bench_{name}")

    cls = getattr(mod, spec["extractor"])
    if spec["kind"] == "hyena":
        # The local dir only carries the `hyena` package code; weights come from
        # the HF repo (same ids as the extract scripts' __main__).
        wrapper = cls(spec["model_id"])
        tokenizer = mod.HyenaDNATokenizer(
            model_max_length=wrapper.model.config.max_seq_len)
    else:
        wrapper = cls(spec["model_id"], num_layers=spec["layers"])
        tokenizer = AutoTokenizer.from_pretrained(
            spec["model_id"],
            trust_remote_code=spec["trust_remote"],
        )
        if spec["pad_to_unk"]:
            tokenizer.pad_token = tokenizer.unk_token

    # Only the encoder backbone is moved to the device; task heads (NTv2/GROVER)
    # stay on CPU so they neither run nor occupy VRAM.
    backbone = resolve_attr(wrapper, spec["backbone"])
    backbone.to(device).eval()

    if spec["kind"] == "hyena":
        max_tokens = backbone.config.max_seq_len - 1  # tokenizer appends [SEP]
    else:
        max_tokens = getattr(backbone.config, "max_position_embeddings", None)
    return backbone, tokenizer, max_tokens


def configure_backend(allow_tf32):
    # The extract scripts set deterministic/cudnn flags at import; override them
    # after loading so every model runs under the same backend policy.
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def backend_meta(backbone):
    param = next(backbone.parameters())
    attn = getattr(backbone.config, "_attn_implementation", None)
    if attn is None:
        attn = getattr(backbone, "_attn_implementation", "")
    return dict(
        dtype=str(param.dtype).replace("torch.", ""),
        attn_impl=attn or "",
        model_revision=getattr(backbone.config, "_commit_hash", "") or "",
        transformers_version=transformers.__version__,
        tf32_cudnn=int(torch.backends.cudnn.allow_tf32),
        tf32_matmul=int(torch.backends.cuda.matmul.allow_tf32),
        cudnn_benchmark=int(torch.backends.cudnn.benchmark),
        matmul_precision=torch.get_float32_matmul_precision(),
    )


def forward_fn(kind):
    if kind == "hyena":
        return lambda backbone, enc: backbone(
            input_ids=enc["input_ids"], return_dict=False)
    return lambda backbone, enc: backbone(
        input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])


def encode(tokenizer, seq):
    enc = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
    return {k: enc[k] for k in ("input_ids", "attention_mask") if k in enc}


def estimate_nt_cap(tokenizer, max_tokens, seed):
    if not max_tokens:
        return None, None
    probe = random_seq(2048, seed)
    n_probe = len(tokenizer(probe, add_special_tokens=True)["input_ids"])
    tok_per_nt = n_probe / 2048.0
    return (int(max_tokens / tok_per_nt) if tok_per_nt > 0 else None), tok_per_nt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--lengths", type=str, default="",
                        help="comma separated nt lengths; default 2^6..2^19, "
                             "loop stops at the ceiling")
    parser.add_argument("--allow-extrapolation", action="store_true",
                        help="keep going past the recommended max length")
    parser.add_argument("--max-nt", type=int, default=DEFAULT_MAX_NT,
                        help="hard ceiling on sequence length in nt "
                             f"(default {DEFAULT_MAX_NT})")
    parser.add_argument("--tok-reps", type=int, default=20,
                        help="tokenization timings per length (median reported)")
    parser.add_argument("--allow-tf32", action="store_true",
                        help="keep TF32 on (default: strict fp32)")
    parser.add_argument("--require-cuda", action="store_true",
                        help="abort instead of falling back to CPU")
    parser.add_argument("--out-dir", type=str,
                        default=os.path.join(ROOT, "bench"))
    args = parser.parse_args()
    if args.reps < 1:
        parser.error("--reps must be >= 1")
    if args.tok_reps < 1:
        parser.error("--tok-reps must be >= 1")

    lengths = ([int(x) for x in args.lengths.split(",") if x]
               if args.lengths else DEFAULT_LENGTHS)
    spec = MODELS[args.model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        print("CUDA requested but not available")
        return 2

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
        gpu_index = torch.cuda.current_device()
    else:
        gpu_name, vram_total, gpu_index = "cpu", "", ""

    run_id = f"{args.model}-{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.model}.csv")
    common = dict(
        run_id=run_id, seed=args.seed, model=args.model,
        batch_size=1, device=str(device), gpu_index=gpu_index,
        gpu_name=gpu_name,
        vram_total_mb=round(vram_total, 1) if vram_total != "" else "",
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda or "",
    )

    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()

        def write(**kw):
            row = {k: "" for k in FIELDS}
            row.update(common)
            row.update(kw)
            row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            row["host_rss_mb"] = round(host_rss_mb(), 1)
            writer.writerow(row)
            fh.flush()

        t0 = time.perf_counter()
        try:
            backbone, tokenizer, max_tokens = build(
                args.model, spec, device)
        except Exception as e:
            write(stage="load", status="load_error", error=str(e)[:500])
            print(f"[load error] {e}")
            return 1
        load_s = time.perf_counter() - t0

        # Must run after build(): the extract scripts set backend flags at import.
        configure_backend(args.allow_tf32)
        common.update(backend_meta(backbone))

        n_params = sum(p.numel() for p in backbone.parameters())
        weight_mb = sum(p.numel() * p.element_size()
                        for p in backbone.parameters()) / 2 ** 20
        write(stage="load", status="ok", load_s=round(load_s, 3),
              n_params=n_params, weight_mb=round(weight_mb, 1),
              latency_ms=round(load_s * 1000, 3))

        nt_cap, tok_per_nt = estimate_nt_cap(tokenizer, max_tokens, args.seed)
        unlimited = spec.get("unlimited", False)
        if spec.get("max_nt"):
            nt_cap = spec["max_nt"] if nt_cap is None else min(nt_cap, spec["max_nt"])
        # The global ceiling caps every model; ALiBi models have no
        # positional limit, so it is their only bound.
        if unlimited or nt_cap is None:
            nt_cap = args.max_nt
        else:
            nt_cap = min(nt_cap, args.max_nt)
        run = forward_fn(spec["kind"])
        use_cuda = device.type == "cuda"
        start = torch.cuda.Event(enable_timing=True) if use_cuda else None
        end = torch.cuda.Event(enable_timing=True) if use_cuda else None

        ratio_txt = ("n/a" if tok_per_nt is None
                     else f"{1 / tok_per_nt:.2f} nt/token")
        print(f"{args.model}: device={device} params={n_params/1e6:.1f}M "
              f"weights={weight_mb:.0f}MB load={load_s:.1f}s")
        print(f"  tokenizer: {ratio_txt} max_tokens={max_tokens} "
              f"ceiling={args.max_nt} cap={nt_cap}"
              f"{' (no positional limit)' if unlimited else ''}")

        for nt in lengths:
            if not args.allow_extrapolation and nt_cap and nt > nt_cap:
                write(stage="stop", status="exceed_max_tokens", n_nt=nt,
                      error=f"nt_cap={nt_cap} max_tokens={max_tokens}")
                print(f"  {nt} nt: stop (cap {nt_cap})")
                break

            seq = random_seq(nt, args.seed)
            try:
                tok_times = []
                for _ in range(args.tok_reps):
                    t0 = time.perf_counter()
                    enc = encode(tokenizer, seq)
                    tok_times.append((time.perf_counter() - t0) * 1000)
                tok_ms = sorted(tok_times)[len(tok_times) // 2]
                n_tokens = int(enc["input_ids"].shape[1])
            except Exception as e:
                write(stage="tokenize", status="error", n_nt=nt,
                      error=str(e)[:500])
                print(f"  {nt} nt: tokenize error ({str(e)[:100]})")
                break

            if (not args.allow_extrapolation and not unlimited and max_tokens
                    and n_tokens > max_tokens):
                write(stage="stop", status="exceed_max_tokens", n_nt=nt,
                      n_tokens=n_tokens, error=f"max_tokens={max_tokens}")
                print(f"  {nt} nt ({n_tokens} tok): stop (max {max_tokens})")
                break

            write(stage="tokenize", repeat=0, status="ok", n_nt=nt,
                  n_tokens=n_tokens, latency_ms=round(tok_ms, 3))

            latencies = []
            peak_alloc = peak_res = ""
            try:
                inputs = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    for _ in range(args.warmup):
                        run(backbone, inputs)
                if use_cuda:
                    torch.cuda.synchronize()

                for rep in range(args.reps):
                    if use_cuda:
                        torch.cuda.reset_peak_memory_stats()
                        base_alloc = torch.cuda.memory_allocated() / 2 ** 20
                        start.record()
                        with torch.no_grad():
                            out = run(backbone, inputs)
                        end.record()
                        torch.cuda.synchronize()
                        latency = start.elapsed_time(end)
                        peak_alloc = torch.cuda.max_memory_allocated() / 2 ** 20
                        peak_res = torch.cuda.max_memory_reserved() / 2 ** 20
                        act_peak = peak_alloc - base_alloc
                    else:
                        base_alloc = act_peak = ""
                        t1 = time.perf_counter()
                        with torch.no_grad():
                            out = run(backbone, inputs)
                        latency = (time.perf_counter() - t1) * 1000
                        peak_alloc = peak_res = ""
                    del out
                    latencies.append(latency)
                    write(stage="forward", repeat=rep, status="ok", n_nt=nt,
                          n_tokens=n_tokens, latency_ms=round(latency, 4),
                          base_alloc_mb=round(base_alloc, 1)
                          if base_alloc != "" else "",
                          peak_alloc_mb=round(peak_alloc, 1)
                          if peak_alloc != "" else "",
                          act_peak_mb=round(act_peak, 1)
                          if act_peak != "" else "",
                          peak_reserved_mb=round(peak_res, 1)
                          if peak_res != "" else "")
            except Exception as e:
                status = "oom" if "out of memory" in str(e).lower() else "error"
                write(stage="forward", repeat=-1, status=status, n_nt=nt,
                      n_tokens=n_tokens, error=str(e)[:500])
                print(f"  {nt} nt: {status} ({str(e)[:100]})")
                if use_cuda:
                    torch.cuda.empty_cache()
                break

            med = sorted(latencies)[len(latencies) // 2]
            peak_txt = "n/a" if peak_alloc == "" else f"{peak_alloc:.0f} MB"
            print(f"  {nt:>6} nt / {n_tokens:>6} tok: median={med:.2f} ms, "
                  f"peak_alloc={peak_txt}")
            del inputs, enc
            if use_cuda:
                torch.cuda.empty_cache()

    print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
