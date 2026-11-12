import csv
import random
from pathlib import Path

N_ROWS = 5000
SEQ_LENS = [70, 70, 70, 70]
LABEL = 1
BASES = "ACGT"
OUT_DIR = Path(__file__).resolve().parent


def random_seq(rng: random.Random, length: int) -> str:
    seq = [rng.choice(BASES) for _ in range(length)]
    rng.shuffle(seq)
    return "".join(seq)


def main() -> None:
    seen: dict[int, int] = {}
    for i, seq_len in enumerate(SEQ_LENS):
        seen[seq_len] = seen.get(seq_len, 0) + 1
        rng = random.Random(20260920 + i + 1)
        out_path = OUT_DIR / f"random_{seq_len}bp_{seen[seq_len]}.csv"
        with out_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sequence", "label"])
            for _ in range(N_ROWS):
                w.writerow([random_seq(rng, seq_len), LABEL])
        print(f"[ok] {out_path.name}  ({N_ROWS} rows, {seq_len} bp)")


if __name__ == "__main__":
    main()
