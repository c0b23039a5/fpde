#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_fpde_uci24_experimental_aime_clean.py

Experimental runner for the 24 retained UCI CC18-like datasets after removing
OpenML-CC18 duplicates.

This runner intentionally reuses the same helper functions and output format as
`run_fpde_openml_cc18_experimental_aime_clean.py`, but replaces the OpenML task
loader/split with a UCI/ucimlrepo loader and a deterministic stratified split.

Expected outputs in output_dir/:
    per_instance_results.csv
    summary_by_task.csv
    summary_by_method.csv
    lambda_distribution.csv
    main_results_table.tex
    lambda_distribution_table.tex
    run_config.json
    candidate_metadata_used.csv
    task_failures.csv                 # only when failures occur

Default methods are the final-comparison set:
    hyb_fpde_grid,aime,shap,lime

For smoke tests, reduce --max-tasks, --n-explain, --n-val-select, and row caps.
For final runs, set --max-tasks 0, --max-train-rows 0, and --max-test-rows 0.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DEFAULT_UCI_IDS = [
    602, 73, 850, 545, 848, 350, 563, 468, 529, 722, 582, 379,
    890, 329, 544, 759, 503, 863, 887, 936, 372, 159, 75, 697,
]

METHODS_DEFAULT = ("hyb_fpde_grid", "aime", "shap", "lime")
FRACTIONS_DEFAULT = "0,0.05,0.1,0.2,0.3,0.5,0.7,1.0"
LAMBDA_GRID_DEFAULT = "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
FPDE_WHEEL_DEFAULT = "fpde_xai-0.1.0-py3-none-any.whl"


@dataclass
class UciRunConfig:
    candidate_csv: str
    uci_ids: Optional[List[int]]
    max_tasks: Optional[int]
    test_size: float
    seed: int
    n_explain: int
    n_val_select: int
    max_train_rows: Optional[int]
    max_test_rows: Optional[int]
    lambda_grid: List[float]
    fractions: List[float]
    methods: List[str]
    output_dir: str
    fpde_path: str
    base_runner_path: str
    n_estimators: int
    learning_rate: float
    num_leaves: int
    n_jobs: int
    lime_num_samples: int
    aime_local_y: str


def parse_float_list(text: str) -> List[float]:
    values: List[float] = []
    for part in text.split(','):
        part = part.strip()
        if part:
            values.append(float(part))
    if not values:
        raise ValueError("float list must contain at least one value")
    if not all(math.isfinite(v) for v in values):
        raise ValueError("float list contains NaN or infinity")
    return values


def parse_int_list_or_none(text: Optional[str]) -> Optional[List[int]]:
    if text is None or text.strip() == "":
        return None
    out: List[int] = []
    for part in text.split(','):
        part = part.strip()
        if part:
            out.append(int(part))
    return out or None


def import_module_from_path(path: Path, module_name: str):
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"module not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_neighbor_path(path_text: str, *, script_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    neighbor = script_dir / path
    if neighbor.exists():
        return neighbor
    return path


def unique_feature_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Ensure feature columns are unique strings before ColumnTransformer.

    Some UCI tables contain duplicate or blank column labels.  The original
    OpenML runner's preprocessor indexes columns by label, so duplicated names
    can make X[col] return a DataFrame instead of a Series.  We avoid that here
    by renaming features deterministically.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    counts: Dict[str, int] = {}
    new_cols: List[str] = []
    for j, col in enumerate(X.columns):
        base = str(col).strip() if str(col).strip() else f"feature_{j}"
        if base in counts:
            counts[base] += 1
            name = f"{base}__dup{counts[base]}"
        else:
            counts[base] = 0
            name = base
        new_cols.append(name)
    out = X.copy()
    out.columns = new_cols
    return out


def normalize_uci_feature_dtypes(X: pd.DataFrame) -> pd.DataFrame:
    """Normalize UCI extension/string dtypes for the shared OpenML preprocessor."""
    out = X.copy()
    for col in out.columns:
        if pd.api.types.is_bool_dtype(out[col]) or pd.api.types.is_numeric_dtype(out[col]):
            continue
        out[col] = out[col].astype(object)
    return out


def find_column_case_insensitive(columns: Sequence[Any], target: Any) -> Optional[Any]:
    if target is None or (isinstance(target, float) and np.isnan(target)):
        return None
    target_s = str(target).strip()
    if not target_s:
        return None
    for c in columns:
        if str(c) == target_s:
            return c
    for c in columns:
        if str(c).strip() == target_s:
            return c
    lower = target_s.lower()
    for c in columns:
        if str(c).strip().lower() == lower:
            return c
    return None


def load_candidate_table(candidate_csv: Path, uci_ids: Optional[List[int]]) -> pd.DataFrame:
    if candidate_csv.exists():
        df = pd.read_csv(candidate_csv)
        if "uci_id" not in df.columns:
            raise ValueError(f"candidate CSV must contain uci_id column: {candidate_csv}")
    else:
        # Fallback: use the embedded 24-ID list when the CSV is not present.
        df = pd.DataFrame({"uci_id": DEFAULT_UCI_IDS})

    df = df.copy()
    df["uci_id"] = df["uci_id"].astype(int)
    if uci_ids is not None:
        wanted = [int(x) for x in uci_ids]
        df = df[df["uci_id"].isin(wanted)].copy()
        # Preserve the user-specified order.
        order = {uci_id: i for i, uci_id in enumerate(wanted)}
        df["__order"] = df["uci_id"].map(order)
        df = df.sort_values("__order").drop(columns=["__order"])
    else:
        # If CSV exists, keep its order; otherwise DEFAULT_UCI_IDS order.
        if not candidate_csv.exists():
            order = {uci_id: i for i, uci_id in enumerate(DEFAULT_UCI_IDS)}
            df["__order"] = df["uci_id"].map(order)
            df = df.sort_values("__order").drop(columns=["__order"])
    if df.empty:
        raise ValueError("no UCI datasets selected")
    return df.reset_index(drop=True)


def fetch_uci_dataframe(candidate_row: pd.Series) -> Tuple[pd.DataFrame, pd.Series, str, Dict[str, Any]]:
    try:
        from ucimlrepo import fetch_ucirepo
    except Exception as exc:
        raise RuntimeError("ucimlrepo is required; install with `pip install ucimlrepo`.") from exc

    uci_id = int(candidate_row["uci_id"])
    ds = fetch_ucirepo(id=uci_id)
    X = ds.data.features
    y = ds.data.targets
    if X is None or y is None:
        raise ValueError("features or targets missing from ucimlrepo result")
    X_df = unique_feature_columns(X if isinstance(X, pd.DataFrame) else pd.DataFrame(X))
    X_df = normalize_uci_feature_dtypes(X_df)
    y_df = y if isinstance(y, pd.DataFrame) else pd.DataFrame(y)
    if y_df.shape[1] == 0:
        raise ValueError("target table has no columns")

    selected_target = None
    if "selected_target" in candidate_row.index:
        selected_target = find_column_case_insensitive(list(y_df.columns), candidate_row.get("selected_target"))
    if selected_target is None and "target_cols" in candidate_row.index:
        first_target = str(candidate_row.get("target_cols", "")).split(",")[0].strip()
        selected_target = find_column_case_insensitive(list(y_df.columns), first_target)
    if selected_target is None:
        selected_target = y_df.columns[0]

    y_ser = y_df[selected_target]
    md = getattr(ds, "metadata", None)
    md_name = None
    try:
        md_name = md.get("name") if isinstance(md, dict) else getattr(md, "name", None)
    except Exception:
        md_name = None
    name = str(candidate_row.get("name", "") or md_name or f"uci_{uci_id}")

    metadata = {str(k): candidate_row.get(k) for k in candidate_row.index}
    metadata.update(
        {
            "uci_id": uci_id,
            "dataset_name": name,
            "selected_target_used": str(selected_target),
            "target_columns_available": ", ".join(map(str, y_df.columns)),
            "n_rows_fetched": int(X_df.shape[0]),
            "n_features_fetched": int(X_df.shape[1]),
        }
    )
    return X_df, y_ser, name, metadata


def run_one_uci_dataset(
    candidate_row: pd.Series,
    *,
    cfg: UciRunConfig,
    fpde: Any,
    base: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    uci_id = int(candidate_row["uci_id"])
    X_df, y_ser, dataset_name, metadata = fetch_uci_dataframe(candidate_row)

    mask = ~pd.isna(y_ser)
    X_df = X_df.loc[mask].reset_index(drop=True)
    y_ser = y_ser.loc[mask].reset_index(drop=True)

    le = LabelEncoder()
    y_all = le.fit_transform(y_ser.astype(str))
    if len(np.unique(y_all)) < 2:
        raise ValueError("dataset has fewer than two classes after target filtering")

    all_idx = np.arange(len(y_all))
    train_idx, test_idx = train_test_split(
        all_idx,
        test_size=float(cfg.test_size),
        random_state=int(cfg.seed),
        stratify=base.safe_stratify(y_all),
    )
    split_name = f"stratified_{1.0 - cfg.test_size:.2f}_{cfg.test_size:.2f}_seed{cfg.seed}"

    X_train_raw = X_df.iloc[train_idx].reset_index(drop=True)
    y_train_full = y_all[train_idx]
    X_test_raw = X_df.iloc[test_idx].reset_index(drop=True)
    y_test_full = y_all[test_idx]

    tr_keep = (
        base.stratified_subsample_indices(y_train_full, cfg.max_train_rows, cfg.seed)
        if cfg.max_train_rows
        else np.arange(len(y_train_full))
    )
    te_keep = (
        base.stratified_subsample_indices(y_test_full, cfg.max_test_rows, cfg.seed + 1)
        if cfg.max_test_rows
        else np.arange(len(y_test_full))
    )
    X_train_raw = X_train_raw.iloc[tr_keep].reset_index(drop=True)
    y_train_full = y_train_full[tr_keep]
    X_test_raw = X_test_raw.iloc[te_keep].reset_index(drop=True)
    y_test_full = y_test_full[te_keep]

    train_inner_idx, val_inner_idx = train_test_split(
        np.arange(len(y_train_full)),
        test_size=0.2,
        random_state=int(cfg.seed),
        stratify=base.safe_stratify(y_train_full),
    )
    X_fit_raw = X_train_raw.iloc[train_inner_idx].reset_index(drop=True)
    y_fit = y_train_full[train_inner_idx]
    X_val_raw = X_train_raw.iloc[val_inner_idx].reset_index(drop=True)
    y_val = y_train_full[val_inner_idx]

    preprocessor = base.build_preprocessor(X_fit_raw)
    X_fit = base.to_dense_float_array(preprocessor.fit_transform(X_fit_raw))
    X_val = base.to_dense_float_array(preprocessor.transform(X_val_raw))
    X_test = base.to_dense_float_array(preprocessor.transform(X_test_raw))

    feature_names = base.get_feature_names(preprocessor, X_fit.shape[1])
    model = base.fit_lightgbm(
        X_fit,
        y_fit,
        seed=int(cfg.seed),
        n_estimators=int(cfg.n_estimators),
        learning_rate=float(cfg.learning_rate),
        num_leaves=int(cfg.num_leaves),
        n_jobs=int(cfg.n_jobs),
        feature_names=feature_names,
    )
    perf = base.model_metrics(model, X_test, y_test_full)

    baseline = np.mean(X_fit, axis=0)
    prototypes, prototype_labels = fpde.class_mean_prototypes(X_fit, y_fit)
    anchor = np.mean(X_fit, axis=0)

    val_sel_idx = base.stratified_subsample_indices(y_val, min(cfg.n_val_select, len(y_val)), cfg.seed + 2)
    selection = None
    lambda_rows: List[Dict[str, Any]] = []
    if "hyb_fpde_grid" in cfg.methods:
        selection = fpde.select_lambda_by_deletion_insertion_validation(
            X_fit,
            y_fit,
            X_val[val_sel_idx],
            model,
            lambda_hyb_grid=cfg.lambda_grid,
            fractions=cfg.fractions,
            baseline=baseline,
            normalize="l1",
            anchor_strategy="mean",
        )
        for row in selection.rows:
            r = dict(row)
            r.update(
                {
                    "task_id": int(uci_id),
                    "uci_id": int(uci_id),
                    "dataset_name": dataset_name,
                    "selected_lambda": float(selection.best_lambda),
                }
            )
            lambda_rows.append(r)

    explain_idx = base.stratified_subsample_indices(y_test_full, min(cfg.n_explain, len(y_test_full)), cfg.seed + 3)

    lime_explainer = None
    if "lime" in cfg.methods:
        try:
            class_names = [str(c) for c in le.classes_]
            lime_explainer = base.build_lime_explainer(X_fit, y_fit, feature_names, class_names)
        except base.MethodUnavailable:
            lime_explainer = None

    aime_adapter = None
    if "aime" in cfg.methods:
        try:
            aime_adapter = base.build_aime_adapter(
                X_fit,
                model,
                feature_names,
                local_y_mode=cfg.aime_local_y,
            )
        except base.MethodUnavailable:
            aime_adapter = None

    per_instance_rows: List[Dict[str, Any]] = []
    for local_i, test_i in enumerate(explain_idx.tolist()):
        x = X_test[test_i]
        true_label = int(y_test_full[test_i])
        pred_proba = np.asarray(model.predict_proba(x.reshape(1, -1))[0], dtype=float)
        pred_label = int(model.classes_[int(np.argmax(pred_proba))])

        for method in cfg.methods:
            start = time.perf_counter()
            status = "ok"
            error = ""
            attr_norm_l1 = float("nan")
            evidence = float("nan")
            lambda_hyb = float("nan")
            exactness_residual = float("nan")
            metrics = {
                "p0": float("nan"),
                "deletion_auc": float("nan"),
                "deletion_drop_auc": float("nan"),
                "insertion_auc": float("nan"),
                "combined_score": float("nan"),
            }
            target_label_for_eval: Any = pred_label

            try:
                if method in ("diff_fpde", "cos_fpde", "hyb_fpde_grid"):
                    attr, target_label_for_eval, detail = base.explain_fpde(
                        fpde,
                        method,
                        x,
                        X_train=X_fit,
                        y_train=y_fit,
                        model=model,
                        prototypes=prototypes,
                        prototype_labels=prototype_labels,
                        anchor=anchor,
                        selection=selection,
                    )
                    evidence = float(detail.get("evidence", np.nan))
                    lambda_hyb = float(detail.get("lambda_hyb", np.nan))
                    exactness_residual = float(detail.get("exactness_residual", np.nan))
                elif method == "shap":
                    attr = base.explain_shap(model, x, target_label_for_eval, feature_names)
                elif method == "lime":
                    if lime_explainer is None:
                        raise base.MethodUnavailable("LIME explainer could not be initialized")
                    attr = base.explain_lime(
                        lime_explainer,
                        model,
                        x,
                        target_label_for_eval,
                        num_samples=cfg.lime_num_samples,
                    )
                elif method == "aime":
                    attr = base.explain_aime(aime_adapter, model, x, target_label_for_eval)
                else:
                    raise ValueError(f"unknown method: {method}")

                attr = np.asarray(attr, dtype=float)
                if attr.shape != x.shape:
                    raise ValueError(f"attribution shape {attr.shape} differs from x shape {x.shape}")
                if not np.all(np.isfinite(attr)):
                    raise ValueError("attribution contains NaN or infinity")
                attr_norm_l1 = float(np.sum(np.abs(attr)))
                metrics = base.evaluate_attribution(fpde, model, x, attr, target_label_for_eval, baseline, cfg.fractions)
            except Exception as exc:
                status = "error"
                error = f"{type(exc).__name__}: {exc}"

            elapsed = float(time.perf_counter() - start)
            row = {
                "task_id": int(uci_id),
                "uci_id": int(uci_id),
                "dataset_name": dataset_name,
                "source": "UCI",
                "split_name": split_name,
                "selected_target": metadata.get("selected_target_used", ""),
                "method": method,
                "status": status,
                "error": error,
                "explained_index": int(test_i),
                "explained_order": int(local_i),
                "n_features_transformed": int(X_fit.shape[1]),
                "n_train_fit": int(X_fit.shape[0]),
                "n_val": int(X_val.shape[0]),
                "n_test": int(X_test.shape[0]),
                "n_classes": int(len(le.classes_)),
                "true_label_encoded": true_label,
                "pred_label_encoded": pred_label,
                "target_label_encoded": int(target_label_for_eval),
                "pred_target_probability": float(pred_proba[int(np.where(model.classes_ == target_label_for_eval)[0][0])]),
                "runtime_seconds": elapsed,
                "attr_l1_norm": attr_norm_l1,
                "evidence": evidence,
                "lambda_hyb": lambda_hyb,
                "exactness_residual": exactness_residual,
                **metrics,
                **perf,
            }
            per_instance_rows.append(row)

    metadata.update(
        {
            "n_rows_after_target_drop": int(X_df.shape[0]),
            "n_features_raw": int(X_df.shape[1]),
            "n_features_transformed": int(X_fit.shape[1]),
            "n_classes_after_label_encoding": int(len(le.classes_)),
            "class_labels_original": json.dumps([str(c) for c in le.classes_], ensure_ascii=False),
            "split_name": split_name,
            "test_size": float(cfg.test_size),
        }
    )
    return per_instance_rows, lambda_rows, metadata


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run FPDE experimental evaluation on 24 retained UCI CC18-like datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--candidate-csv",
        type=str,
        default="uci_cc18_like_final_base_candidates_24_no_openml_cc18_duplicates.csv",
        help="CSV containing the 24 retained UCI candidates and selected_target metadata.",
    )
    p.add_argument("--uci-ids", type=str, default=None, help="Comma-separated UCI IDs. Overrides candidate CSV subset when given.")
    p.add_argument("--max-tasks", type=int, default=3, help="Maximum number of UCI datasets for a smoke run. Set 0 for all selected datasets.")
    p.add_argument("--test-size", type=float, default=0.2, help="Stratified held-out test fraction for UCI datasets.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-explain", type=int, default=20, help="Number of test instances explained per dataset.")
    p.add_argument("--n-val-select", type=int, default=20, help="Validation instances used to select Hyb-FPDE-grid lambda.")
    p.add_argument("--max-train-rows", type=int, default=5000, help="Optional train cap for smoke runs. Set 0 to disable.")
    p.add_argument("--max-test-rows", type=int, default=2000, help="Optional test cap for smoke runs. Set 0 to disable.")
    p.add_argument("--lambda-grid", type=str, default=LAMBDA_GRID_DEFAULT)
    p.add_argument("--fractions", type=str, default=FRACTIONS_DEFAULT)
    p.add_argument("--methods", type=str, default=",".join(METHODS_DEFAULT), help="Comma-separated method names.")
    p.add_argument("--output-dir", type=str, default="experimental_outputs_uci24")
    p.add_argument(
        "--fpde-path",
        type=str,
        default=FPDE_WHEEL_DEFAULT,
        help="Path to the fpde-xai wheel or a legacy FPDE.py module. Relative paths are resolved from the current directory or this runner's directory.",
    )
    p.add_argument(
        "--base-runner-path",
        type=str,
        default="run_fpde_openml_cc18_experimental_aime_clean.py",
        help="Path to the existing OpenML runner. Its preprocessing/explainer/output helpers are reused.",
    )
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--num-leaves", type=int, default=31)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--lime-num-samples", type=int, default=1000)
    p.add_argument(
        "--aime-local-y",
        type=str,
        choices=("onehot", "proba"),
        default="onehot",
        help="Output vector passed to AIME local_feature_importance: onehot target class or full predict_proba vector.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    allowed = {"diff_fpde", "cos_fpde", "hyb_fpde_grid", "shap", "lime", "aime"}
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise ValueError(f"unknown methods: {unknown}; allowed={sorted(allowed)}")
    if not (0.0 < float(args.test_size) < 1.0):
        raise ValueError("--test-size must be in (0, 1)")

    script_dir = Path(__file__).resolve().parent
    base_runner_path = resolve_neighbor_path(args.base_runner_path, script_dir=script_dir)
    fpde_path = resolve_neighbor_path(args.fpde_path, script_dir=script_dir)
    candidate_csv = resolve_neighbor_path(args.candidate_csv, script_dir=script_dir)

    base = import_module_from_path(base_runner_path, "fpde_openml_runner_helpers")
    fpde = base.import_fpde_module(fpde_path)

    cfg = UciRunConfig(
        candidate_csv=str(candidate_csv),
        uci_ids=parse_int_list_or_none(args.uci_ids),
        max_tasks=None if args.max_tasks is None or args.max_tasks <= 0 else int(args.max_tasks),
        test_size=float(args.test_size),
        seed=int(args.seed),
        n_explain=int(args.n_explain),
        n_val_select=int(args.n_val_select),
        max_train_rows=None if args.max_train_rows is None or args.max_train_rows <= 0 else int(args.max_train_rows),
        max_test_rows=None if args.max_test_rows is None or args.max_test_rows <= 0 else int(args.max_test_rows),
        lambda_grid=parse_float_list(args.lambda_grid),
        fractions=parse_float_list(args.fractions),
        methods=methods,
        output_dir=str(args.output_dir),
        fpde_path=str(fpde_path),
        base_runner_path=str(base_runner_path),
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        n_jobs=int(args.n_jobs),
        lime_num_samples=int(args.lime_num_samples),
        aime_local_y=str(args.aime_local_y),
    )

    candidates = load_candidate_table(candidate_csv, cfg.uci_ids)
    if cfg.max_tasks is not None:
        candidates = candidates.iloc[: cfg.max_tasks].reset_index(drop=True)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_instance_rows: List[Dict[str, Any]] = []
    lambda_rows: List[Dict[str, Any]] = []
    metadata_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    ids = candidates["uci_id"].astype(int).tolist()
    print(f"Running {len(ids)} UCI dataset(s): {ids}")
    for order, (_, row) in enumerate(candidates.iterrows(), start=1):
        uci_id = int(row["uci_id"])
        name = str(row.get("name", f"uci_{uci_id}"))
        print(f"[{order}/{len(candidates)}] uci_id={uci_id} dataset={name}", flush=True)
        try:
            rows, lam_rows, metadata = run_one_uci_dataset(row, cfg=cfg, fpde=fpde, base=base)
            per_instance_rows.extend(rows)
            lambda_rows.extend(lam_rows)
            metadata_rows.append(metadata)
            base.write_outputs(output_dir, cfg, per_instance_rows, lambda_rows)
            pd.DataFrame(metadata_rows).to_csv(output_dir / "candidate_metadata_used.csv", index=False)
        except Exception as exc:
            failures.append(
                {
                    "task_id": int(uci_id),
                    "uci_id": int(uci_id),
                    "dataset_name": name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            pd.DataFrame(failures).to_csv(output_dir / "task_failures.csv", index=False)
            print(f"  ERROR: {type(exc).__name__}: {exc}", flush=True)

    base.write_outputs(output_dir, cfg, per_instance_rows, lambda_rows)
    pd.DataFrame(metadata_rows).to_csv(output_dir / "candidate_metadata_used.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(output_dir / "task_failures.csv", index=False)
    print(f"Done. Outputs written to: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
