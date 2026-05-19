#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate FPDE result folders.

Run from the artifact root:
    python scripts/aggregate_precomputed_results.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate FPDE per-instance result folders.")
    parser.add_argument("--results-dir", type=Path, default=Path("results_precomputed"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--glob-pattern",
        default="experimental_outputs_aime_full_n*_seed*",
        help="Directory glob under results-dir to aggregate.",
    )
    args = parser.parse_args()

    results = args.results_dir if args.results_dir.is_absolute() else ROOT / args.results_dir
    out = args.out_dir if args.out_dir is not None else results / "aggregate_all_seeds"
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    parts = []
    configs = []
    for d in sorted(results.glob(args.glob_pattern)):
        cfg_path = d / "run_config.json"
        seed = None
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            seed = cfg.get("seed")
            configs.append(cfg)
        p = d / "per_instance_results.csv"
        if not p.exists():
            print(f"[skip] missing {p}")
            continue
        df = pd.read_csv(p)
        df["seed_from_dir"] = seed
        parts.append(df)

    if not parts:
        raise SystemExit(f"No per_instance_results.csv files found under {results}.")

    all_per = pd.concat(parts, ignore_index=True)
    summary = all_per.groupby("method").agg(
        mean_deletion_drop_auc=("deletion_drop_auc", "mean"),
        std_deletion_drop_auc=("deletion_drop_auc", "std"),
        mean_insertion_auc=("insertion_auc", "mean"),
        std_insertion_auc=("insertion_auc", "std"),
        mean_combined_score=("combined_score", "mean"),
        std_combined_score=("combined_score", "std"),
        mean_runtime_seconds=("runtime_seconds", "mean"),
        std_runtime_seconds=("runtime_seconds", "std"),
        completed_instances=("combined_score", "count"),
        completed_tasks=("task_id", lambda s: int(pd.Series(s).nunique())),
        completed_seeds=("seed_from_dir", lambda s: int(pd.Series(s).nunique())),
        mean_test_accuracy=("test_accuracy", "mean"),
        mean_test_balanced_accuracy=("test_balanced_accuracy", "mean"),
    ).reset_index().sort_values("mean_combined_score", ascending=False)
    summary.to_csv(out / "summary_by_method_all_seeds.csv", index=False)
    print(f"Wrote {out / 'summary_by_method_all_seeds.csv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
