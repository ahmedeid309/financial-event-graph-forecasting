"""Summarize a fusion-ablation result file into the slide-11 style table.

Reads the append-only result file written by run.py, pairs every arm against
the ABL_P price-only baseline within the same seed, and prints one row per arm
sorted by mean MSE.

Usage:
    python summarize_ablation.py --result_file result_ablation_T.txt
"""

from __future__ import annotations

import argparse
import re
import statistics as st
from pathlib import Path
from typing import Dict, List

# arm label -> which of the three components were ON
COMPONENTS: Dict[str, tuple] = {
    "A": (1, 0, 0),
    "AG": (1, 1, 0),
    "F": (1, 1, 1),
    "AR": (1, 0, 1),
    "GR": (0, 1, 1),
    "R": (0, 0, 1),
    "G": (0, 1, 0),
    "C2": (0, 0, 0),
}

RUN = re.compile(
    r"long_term_forecast_ABL_([A-Z0-9]+)_s(\d+)_.*?\n\s*mse:([0-9.eE+-]+), mae:([0-9.eE+-]+)",
    re.S,
)


def parse(path: Path) -> Dict[str, Dict[str, float]]:
    """arm -> {seed: mse}. A repeated (arm, seed) keeps the last value written."""
    out: Dict[str, Dict[str, float]] = {}
    for arm, seed, mse, _mae in RUN.findall(path.read_text()):
        out.setdefault(arm, {})[seed] = float(mse)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", default="result_ablation_T.txt")
    args = parser.parse_args()

    path = Path(args.result_file)
    if not path.exists():
        raise SystemExit(f"No such result file: {path}")

    by_arm = parse(path)
    if "P" not in by_arm:
        raise SystemExit("No ABL_P price-only baseline rows found; cannot pair.")

    baseline = by_arm.pop("P")
    seeds = sorted(baseline, key=int)
    base_mean = st.fmean(baseline[s] for s in seeds)

    print(f"\nPrice-only baseline: MSE {base_mean:.6f}   (seeds {', '.join(seeds)})\n")
    header = f"{'cell':<5}{'gate':<6}{'attention':<11}{'reduction':<11}{'mean MSE':<12}{'sd':<11}{'vs price-only':<15}{'wins'}"
    print(header)
    print("-" * len(header))

    rows: List[tuple] = []
    for arm, per_seed in by_arm.items():
        shared = [s for s in seeds if s in per_seed]
        if not shared:
            continue
        vals = [per_seed[s] for s in shared]
        mean = st.fmean(vals)
        sd = st.stdev(vals) if len(vals) > 1 else 0.0
        wins = sum(1 for s in shared if per_seed[s] < baseline[s])
        paired_base = st.fmean(baseline[s] for s in shared)
        rows.append((mean, arm, sd, 100.0 * (mean - paired_base) / paired_base, wins, len(shared)))

    for mean, arm, sd, pct, wins, n in sorted(rows):
        a, g, r = COMPONENTS.get(arm, ("?", "?", "?"))
        yn = lambda v: "yes" if v == 1 else ("no" if v == 0 else "?")
        on_off = lambda v: "on" if v == 1 else ("off" if v == 0 else "?")
        print(
            f"{arm:<5}{on_off(g):<6}{yn(a):<11}{yn(r):<11}"
            f"{mean:<12.6f}{sd:<11.6f}{pct:>+8.2f}%      {wins}/{n}"
        )

    print("\nCompare cells within this table only. Every arm is paired against the")
    print("ABL_P baseline of the same seed, so the percentages are within-seed.")


if __name__ == "__main__":
    main()
