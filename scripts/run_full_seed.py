#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one full OpenML-CC18 FPDE experiment seed or seed/task pair.

This is intended for GitHub Actions matrix jobs, where each seed should run in
its own worker and upload its result folder as an artifact.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one full OpenML-CC18 FPDE seed.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("results_recomputed"))
    parser.add_argument("--suite-id", type=int, default=99)
    parser.add_argument("--task-id", type=int, default=None, help="Single OpenML task ID to run.")
    parser.add_argument("--task-ids", default=None, help="Comma-separated OpenML task IDs to run.")
    parser.add_argument("--max-tasks", type=int, default=72)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--n-explain", type=int, default=200)
    parser.add_argument("--n-val-select", type=int, default=200)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-test-rows", type=int, default=0)
    parser.add_argument(
        "--methods",
        default="diff_fpde,cos_fpde,hyb_fpde_grid,shap,lime,aime",
        help="Comma-separated explanation methods to evaluate.",
    )
    parser.add_argument("--aime-local-y", default="onehot")
    parser.add_argument("--force", action="store_true", help="Run even if summary_by_method.csv already exists.")
    args = parser.parse_args()

    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    task_ids = args.task_ids
    task_suffix = ""
    if args.task_id is not None:
        task_ids = str(args.task_id)
        task_suffix = f"_task{args.task_id}"
    elif task_ids:
        normalized = "_".join(part.strip() for part in task_ids.split(",") if part.strip())
        if normalized:
            task_suffix = f"_tasks{normalized}"

    out = output_root / f"experimental_outputs_aime_full_n{args.n_explain}_seed{args.seed}{task_suffix}"
    summary = out / "summary_by_method.csv"
    if summary.exists() and not args.force:
        task_label = f", task_ids={task_ids}" if task_ids else ""
        print(f"[skip] seed={args.seed}{task_label}: {summary} already exists")
        return

    cmd = [
        sys.executable,
        str(ROOT / "code" / "run_fpde_openml_cc18_experimental_aime_clean.py"),
        "--fpde-path",
        str(ROOT / "code" / "FPDE.py"),
        "--output-dir",
        str(out),
        "--suite-id",
        str(args.suite_id),
        "--fold",
        str(args.fold),
        "--repeat",
        str(args.repeat),
        "--sample",
        str(args.sample),
        "--seed",
        str(args.seed),
        "--n-explain",
        str(args.n_explain),
        "--n-val-select",
        str(args.n_val_select),
        "--max-train-rows",
        str(args.max_train_rows),
        "--max-test-rows",
        str(args.max_test_rows),
        "--methods",
        args.methods,
        "--aime-local-y",
        args.aime_local_y,
    ]
    if task_ids:
        cmd.extend(["--task-ids", task_ids])
        cmd.extend(["--max-tasks", "0"])
    else:
        cmd.extend(["--max-tasks", str(args.max_tasks)])
    print(" ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    main()
