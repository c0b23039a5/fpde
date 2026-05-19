#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one full UCI-24 FPDE experiment seed or selected UCI datasets."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one UCI-24 FPDE seed.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("results_uci24"))
    parser.add_argument("--uci-ids", default=None, help="Comma-separated UCI dataset IDs to run.")
    parser.add_argument("--max-tasks", type=int, default=0, help="Maximum datasets to run; 0 means all selected datasets.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-explain", type=int, default=20)
    parser.add_argument("--n-val-select", type=int, default=20)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-test-rows", type=int, default=0)
    parser.add_argument(
        "--methods",
        default="hyb_fpde_grid,aime,shap,lime",
        help="Comma-separated explanation methods to evaluate.",
    )
    parser.add_argument("--aime-local-y", default="onehot")
    parser.add_argument("--force", action="store_true", help="Run even if summary_by_method.csv already exists.")
    args = parser.parse_args()

    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    uci_suffix = ""
    if args.uci_ids:
        normalized = "_".join(part.strip() for part in args.uci_ids.split(",") if part.strip())
        if normalized:
            uci_suffix = f"_uci{normalized}"

    out = output_root / f"experimental_outputs_uci24_full_n{args.n_explain}_seed{args.seed}{uci_suffix}"
    summary = out / "summary_by_method.csv"
    if summary.exists() and not args.force:
        dataset_label = f", uci_ids={args.uci_ids}" if args.uci_ids else ""
        print(f"[skip] seed={args.seed}{dataset_label}: {summary} already exists")
        return

    cmd = [
        sys.executable,
        str(ROOT / "code" / "run_fpde_uci24_experimental_aime_clean.py"),
        "--output-dir",
        str(out),
        "--seed",
        str(args.seed),
        "--test-size",
        str(args.test_size),
        "--n-explain",
        str(args.n_explain),
        "--n-val-select",
        str(args.n_val_select),
        "--max-train-rows",
        str(args.max_train_rows),
        "--max-test-rows",
        str(args.max_test_rows),
        "--max-tasks",
        str(args.max_tasks),
        "--methods",
        args.methods,
        "--aime-local-y",
        args.aime_local_y,
    ]
    if args.uci_ids:
        cmd.extend(["--uci-ids", args.uci_ids])

    print(" ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    main()
