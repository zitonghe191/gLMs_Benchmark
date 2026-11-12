import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import glob
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM


MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species"


def extract_attn_one(model, tok, seq, device):
    enc = tok(seq, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )
    attn = out.attentions
    if attn is None:
        raise RuntimeError("model returned no attentions — verify output_attentions=True")
    stacked = torch.stack([a.squeeze(0) for a in attn], dim=0).mean(dim=1)
    return stacked.cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--out_dir", default="./results")
    p.add_argument("--model",    default=MODEL_NAME)
    p.add_argument("--max_seqs", type=int, default=None,
                   help="cap per dataset (any label, applied before label filter); default None = use all CSV rows")
    p.add_argument("--max_pos", type=int, default=5000,
                   help="cap positive-class sequences per dataset; if positives > this, randomly sample this many; default 5000")
    args = p.parse_args()

    csv_files = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))
    if not csv_files:
        print(f"  [error] no csv files in {args.data_dir}")
        return
    print(f"found {len(csv_files)} datasets in {args.data_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        output_attentions=True,
    ).to(device)
    model.eval()
    for q in model.parameters():
        q.requires_grad_(False)
    print(f"layers: {model.config.num_hidden_layers}  "
          f"heads: {model.config.num_attention_heads}")

    for path in csv_files:
        ds_name = os.path.basename(path)
        df = pd.read_csv(path)
        if args.max_seqs is not None:
            df = df.iloc[:args.max_seqs]
        df = df[df["label"] == 1].reset_index(drop=True)
        n_pos = len(df)
        if args.max_pos is not None and n_pos > args.max_pos:
            df = df.sample(n=args.max_pos, random_state=42).reset_index(drop=True)
        n = len(df)
        print(f"\n=== {ds_name}  ({n}/{n_pos} positive sequences used) ===")

        attn_arr, seqs_arr = [], []
        for i in tqdm(range(n), desc=ds_name):
            seq = str(df.iloc[i]["sequence"]).upper().replace("U", "T")
            if not set(seq) <= {"A", "C", "G", "T"} or len(seq) == 0:
                continue
            try:
                a = extract_attn_one(model, tok, seq, device)
            except Exception as e:
                if i < 3 or i % 50 == 0:
                    print(f"  [skip seq {i}] {e}")
                continue
            attn_arr.append(a)
            seqs_arr.append(seq)

        if not attn_arr:
            print(f"  [warn] no valid sequences")
            continue
        out_attn = os.path.join(args.out_dir, ds_name.replace(".csv", "_attn.npy"))
        out_seq  = os.path.join(args.out_dir, ds_name.replace(".csv", "_seqs.npy"))
        np.save(out_attn, np.stack(attn_arr))
        np.save(out_seq,  np.array(seqs_arr))
        print(f"  saved: {out_attn}  shape={np.stack(attn_arr).shape}  "
              f"dtype=float32")
        print(f"  saved: {out_seq}")


if __name__ == "__main__":
    main()