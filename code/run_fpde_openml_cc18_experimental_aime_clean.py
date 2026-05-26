#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_fpde_openml_cc18_experimental.py

Experimental runner for the FPDE MLWA manuscript.

Default target:
    - OpenML-CC18 benchmark suite, suite id 99
    - LightGBM black-box classifier
    - Diff-FPDE, Cos-FPDE, Hyb-FPDE-grid with validation-selected lambda
    - Optional SHAP and LIME baselines when installed
    - AIME baseline through the official `aime-xai` local feature-importance API
      when `aime-xai` and its plotting/embedding dependencies are installed.

Outputs:
    output_dir/
        per_instance_results.csv
        summary_by_task.csv
        summary_by_method.csv
        lambda_distribution.csv
        main_results_table.tex
        lambda_distribution_table.tex
        run_config.json

This file intentionally does not fabricate any numerical result.  It computes
all reported values from the actual run.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Keep optional TensorFlow/oneDNN logs quiet when transitive dependencies import TensorFlow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from sklearn.preprocessing import OneHotEncoder
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scikit-learn is required") from exc


METHODS_DEFAULT = (
    "diff_fpde",
    "cos_fpde",
    "hyb_fpde_grid",
    "shap",
    "lime",
    "aime",
)

FRACTIONS_DEFAULT = "0,0.05,0.1,0.2,0.3,0.5,0.7,1.0"
LAMBDA_GRID_DEFAULT = "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
FPDE_WHEEL_DEFAULT = "fpde_xai-0.1.0-py3-none-any.whl"


@dataclass
class RunConfig:
    suite_id: int
    task_ids: Optional[List[int]]
    max_tasks: Optional[int]
    seed: int
    fold: int
    repeat: int
    sample: int
    n_explain: int
    n_val_select: int
    max_train_rows: Optional[int]
    max_test_rows: Optional[int]
    lambda_grid: List[float]
    fractions: List[float]
    methods: List[str]
    output_dir: str
    fpde_path: str
    n_estimators: int
    learning_rate: float
    num_leaves: int
    n_jobs: int
    lime_num_samples: int
    aime_local_y: str


class MethodUnavailable(RuntimeError):
    """Raised when an optional explanation method cannot be evaluated."""


def parse_float_list(text: str) -> List[float]:
    values: List[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
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
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out or None


def _add_fpde_runner_compat(fpde: Any) -> Any:
    if not hasattr(fpde, "select_lambda_by_deletion_insertion_validation") and hasattr(fpde, "FPDEEngine"):
        def select_lambda_by_deletion_insertion_validation(
            X_train: np.ndarray,
            y_train: Sequence[Any],
            X_val: np.ndarray,
            model: Any,
            *,
            lambda_hyb_grid: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
            fractions: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
            baseline: Optional[np.ndarray] = None,
            normalize: str = "l1",
            anchor_strategy: str = "mean",
            eps: float = 1e-12,
        ) -> Any:
            engine = fpde.FPDEEngine.fit(X_train, y_train, model=model, baseline=baseline)
            return engine.select_lambda(
                X_val,
                lambda_hyb_grid=lambda_hyb_grid,
                fractions=fractions,
                normalize=normalize,
                anchor_strategy=anchor_strategy,
                eps=eps,
            )

        fpde.select_lambda_by_deletion_insertion_validation = select_lambda_by_deletion_insertion_validation
        fpde.select_hyb_fpde_grid_lambda = select_lambda_by_deletion_insertion_validation

    if not hasattr(fpde, "explain_with_validation_selected_lambda") and hasattr(fpde, "FPDEEngine"):
        def explain_with_validation_selected_lambda(
            x: np.ndarray,
            X_train: np.ndarray,
            y_train: Sequence[Any],
            model: Any,
            selection: Any,
            *,
            normalize: str = "l1",
            anchor_strategy: str = "mean",
            eps: float = 1e-12,
        ) -> Tuple[np.ndarray, Dict[str, Any]]:
            engine = fpde.FPDEEngine.fit(X_train, y_train, model=model)
            return engine.explain_one(
                x,
                lambda_hyb=float(selection.best_lambda),
                normalize=normalize,
                anchor_strategy=anchor_strategy,
                eps=eps,
            )

        fpde.explain_with_validation_selected_lambda = explain_with_validation_selected_lambda

    required = (
        "class_mean_prototypes",
        "top_two_labels",
        "explain_with_selected_prototypes",
        "perturbation_curves",
        "select_lambda_by_deletion_insertion_validation",
        "explain_with_validation_selected_lambda",
    )
    missing = [name for name in required if not hasattr(fpde, name)]
    if missing:
        raise ImportError(f"FPDE module is missing required runner API(s): {missing}")
    return fpde


def import_fpde_module(fpde_path: Path):
    fpde_path = fpde_path.expanduser().resolve()
    if not fpde_path.exists():
        raise FileNotFoundError(f"FPDE module not found: {fpde_path}")
    if fpde_path.suffix == ".whl":
        wheel_path = str(fpde_path)
        if wheel_path not in sys.path:
            sys.path.insert(0, wheel_path)
        for module_name in [name for name in sys.modules if name == "fpde" or name.startswith("fpde.")]:
            del sys.modules[module_name]
        return _add_fpde_runner_compat(importlib.import_module("fpde"))

    spec = importlib.util.spec_from_file_location("fpde_core", str(fpde_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load FPDE module from {fpde_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["fpde_core"] = module
    spec.loader.exec_module(module)
    return _add_fpde_runner_compat(module)


def make_one_hot_encoder():
    """Return a dense OneHotEncoder compatible with older/newer sklearn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Fit-time preprocessing for heterogeneous OpenML tabular data.

    Numeric columns: median imputation + standardization.
    Categorical columns: most-frequent imputation + one-hot encoding.

    The final transformed representation is dense because FPDE, SHAP, and LIME
    are all evaluated on a common vector representation.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    categorical_cols = [
        c
        for c in X.columns
        if str(X[c].dtype) in ("object", "category", "bool") or pd.api.types.is_bool_dtype(X[c])
    ]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical_cols,
            )
        )

    if not transformers:
        raise ValueError("no usable feature columns found")

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)


def get_feature_names(preprocessor: ColumnTransformer, fallback_dim: int) -> List[str]:
    try:
        names = preprocessor.get_feature_names_out()
        return [str(n) for n in names]
    except Exception:
        return [f"x{j}" for j in range(fallback_dim)]


def to_dense_float_array(X: Any) -> np.ndarray:
    if hasattr(X, "toarray"):
        X = X.toarray()
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D array, got shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("transformed matrix contains NaN or infinity")
    return arr


def stratified_subsample_indices(y: np.ndarray, n: Optional[int], seed: int) -> np.ndarray:
    """Return deterministic stratified subsample indices when possible."""
    n_total = int(len(y))
    if n is None or n >= n_total:
        return np.arange(n_total)
    if n <= 0:
        raise ValueError("subsample size must be positive")

    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    chosen: List[int] = []
    for cls, count in zip(classes, counts):
        cls_idx = np.where(y == cls)[0]
        k = max(1, int(round(n * (count / n_total))))
        k = min(k, len(cls_idx))
        chosen.extend(rng.choice(cls_idx, size=k, replace=False).tolist())
    if len(chosen) > n:
        chosen = rng.choice(np.asarray(chosen), size=n, replace=False).tolist()
    elif len(chosen) < n:
        remain = np.setdiff1d(np.arange(n_total), np.asarray(chosen, dtype=int), assume_unique=False)
        extra = rng.choice(remain, size=min(n - len(chosen), len(remain)), replace=False).tolist()
        chosen.extend(extra)
    return np.asarray(sorted(set(int(i) for i in chosen)), dtype=int)


def safe_stratify(y: np.ndarray) -> Optional[np.ndarray]:
    _, counts = np.unique(y, return_counts=True)
    return y if np.all(counts >= 2) else None


def load_openml_task_dataframe(task_id: int) -> Tuple[Any, pd.DataFrame, pd.Series, str]:
    import openml

    task = openml.tasks.get_task(task_id)
    dataset_name = f"task_{task_id}"

    # Preferred current API.
    try:
        X, y = task.get_X_and_y(dataset_format="dataframe")
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_ser = y if isinstance(y, pd.Series) else pd.Series(y, name="target")
        if hasattr(task, "get_dataset"):
            try:
                dataset_name = str(task.get_dataset().name)
            except Exception:
                pass
        return task, X_df, y_ser, dataset_name
    except Exception:
        pass

    # Fallback through dataset object.
    dataset = task.get_dataset()
    dataset_name = str(getattr(dataset, "name", dataset_name))
    target = getattr(task, "target_name", None) or getattr(dataset, "default_target_attribute", None)
    if target is None:
        raise ValueError(f"cannot infer target attribute for OpenML task {task_id}")
    X, y, _, _ = dataset.get_data(target=target, dataset_format="dataframe")
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    y_ser = y if isinstance(y, pd.Series) else pd.Series(y, name=str(target))
    return task, X_df, y_ser, dataset_name


def get_suite_task_ids(suite_id: int) -> List[int]:
    import openml

    suite = openml.study.get_suite(suite_id)
    return [int(t) for t in suite.tasks]


def split_task_data(
    task: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    fold: int,
    repeat: int,
    sample: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Use the OpenML task split if available; otherwise fall back to random split."""
    try:
        train_idx, test_idx = task.get_train_test_split_indices(fold=fold, repeat=repeat, sample=sample)
        train_idx = np.asarray(train_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)
        if train_idx.size > 0 and test_idx.size > 0:
            return train_idx, test_idx, f"openml_repeat{repeat}_fold{fold}_sample{sample}"
    except Exception:
        pass

    all_idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        all_idx,
        test_size=0.2,
        random_state=seed,
        stratify=safe_stratify(y),
    )
    return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int), "random_80_20_fallback"


class FeatureNameSafePredictor:
    """Prediction wrapper that preserves transformed feature names.

    LightGBM's sklearn interface may record feature names during fit.  FPDE, LIME,
    and perturbation curves usually pass NumPy arrays back into predict_proba().
    This wrapper converts those arrays back to DataFrames with the same columns,
    avoiding repeated "X does not have valid feature names" warnings.
    """

    def __init__(self, model: Any, feature_names: Sequence[str]):
        self.model = model
        self.raw_model = model
        self.feature_names = [str(name) for name in feature_names]
        self.classes_ = np.asarray(model.classes_)

    def _as_feature_frame(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.loc[:, self.feature_names]
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"expected 1D or 2D input, got shape={arr.shape}")
        if arr.shape[1] != len(self.feature_names):
            raise ValueError(
                f"feature dimension mismatch: expected {len(self.feature_names)}, got {arr.shape[1]}"
            )
        return pd.DataFrame(arr, columns=self.feature_names)

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.model.predict_proba(self._as_feature_frame(X)), dtype=float)

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model.predict(self._as_feature_frame(X)))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)


def fit_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    n_estimators: int,
    learning_rate: float,
    num_leaves: int,
    n_jobs: int,
    feature_names: Optional[Sequence[str]] = None,
):
    from lightgbm import LGBMClassifier

    names = [str(n) for n in (feature_names or [f"x{j}" for j in range(X_train.shape[1])])]
    X_train_df = pd.DataFrame(np.asarray(X_train, dtype=float), columns=names)

    model = LGBMClassifier(
        objective="multiclass" if len(np.unique(y_train)) > 2 else "binary",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=-1,
    )
    model.fit(X_train_df, y_train)
    return FeatureNameSafePredictor(model, names)


def model_metrics(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    proba = np.asarray(model.predict_proba(X_test), dtype=float)
    pred = np.asarray(model.predict(X_test))
    out = {
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
    }
    try:
        out["test_log_loss"] = float(log_loss(y_test, proba, labels=np.asarray(model.classes_)))
    except Exception:
        out["test_log_loss"] = float("nan")
    return out


def explain_fpde(
    fpde: Any,
    method: str,
    x: np.ndarray,
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model: Any,
    prototypes: np.ndarray,
    prototype_labels: np.ndarray,
    anchor: np.ndarray,
    selection: Optional[Any],
) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
    positive_label, negative_label, _ = fpde.top_two_labels(model, x)

    if method == "diff_fpde":
        exp = fpde.explain_with_selected_prototypes(
            x,
            prototypes,
            prototype_labels,
            positive_label=positive_label,
            negative_label=negative_label,
            mode="diff",
        )
        return np.asarray(exp.attributions, dtype=float), positive_label, {
            "evidence": float(exp.evidence),
            "lambda_hyb": np.nan,
            "exactness_residual": float(exp.exactness_residual),
        }

    if method == "cos_fpde":
        exp = fpde.explain_with_selected_prototypes(
            x,
            prototypes,
            prototype_labels,
            positive_label=positive_label,
            negative_label=negative_label,
            mode="cos",
            anchor=anchor,
        )
        return np.asarray(exp.attributions, dtype=float), positive_label, {
            "evidence": float(exp.evidence),
            "lambda_hyb": np.nan,
            "exactness_residual": float(exp.exactness_residual),
        }

    if method == "hyb_fpde_grid":
        if selection is None:
            raise ValueError("hyb_fpde_grid requires a validation selection result")
        attr, detail = fpde.explain_with_validation_selected_lambda(
            x,
            X_train,
            y_train,
            model,
            selection,
            normalize="l1",
            anchor_strategy="mean",
        )
        return np.asarray(attr, dtype=float), detail["target_label"], {
            "evidence": float(detail.get("evidence", np.nan)),
            "lambda_hyb": float(detail.get("lambda_hyb", np.nan)),
            "exactness_residual": float(detail.get("exactness_residual", np.nan)),
        }

    raise ValueError(f"unknown FPDE method: {method}")


def explain_shap(
    model: Any,
    x: np.ndarray,
    target_label: Any,
    feature_names: Optional[Sequence[str]] = None,
) -> np.ndarray:
    try:
        import shap
    except Exception as exc:
        raise MethodUnavailable("SHAP is not installed; install with `pip install shap`.") from exc

    classes = np.asarray(model.classes_)
    target_index = int(np.where(classes == target_label)[0][0])
    raw_model = getattr(model, "raw_model", model)
    names = list(feature_names or getattr(model, "feature_names", [f"x{j}" for j in range(x.shape[0])]))
    x_frame = pd.DataFrame(np.asarray(x, dtype=float).reshape(1, -1), columns=names)

    explainer = shap.TreeExplainer(raw_model)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="LightGBM binary classifier with TreeExplainer shap values output has changed.*",
            category=UserWarning,
        )
        raw = explainer.shap_values(x_frame)

    if hasattr(raw, "values"):
        values = np.asarray(raw.values)
    else:
        values = raw

    if isinstance(values, list):
        if len(values) == 1:
            return np.asarray(values[0][0], dtype=float)
        return np.asarray(values[target_index][0], dtype=float)

    arr = np.asarray(values, dtype=float)
    if arr.ndim == 2:
        return arr[0]
    if arr.ndim == 3:
        # Common new shape: (n_samples, n_features, n_classes)
        if arr.shape[0] == 1 and arr.shape[2] == len(classes):
            return arr[0, :, target_index]
        # Older/alternate shape: (n_classes, n_samples, n_features)
        if arr.shape[0] == len(classes) and arr.shape[1] == 1:
            return arr[target_index, 0, :]
    raise ValueError(f"unsupported SHAP value shape: {arr.shape}")


def build_lime_explainer(X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str], class_names: List[str]):
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except Exception as exc:
        raise MethodUnavailable("LIME is not installed; install with `pip install lime`.") from exc

    return LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=False,
        random_state=0,
    )


def explain_lime(
    lime_explainer: Any,
    model: Any,
    x: np.ndarray,
    target_label: Any,
    *,
    num_samples: int,
) -> np.ndarray:
    target_index = int(np.where(np.asarray(model.classes_) == target_label)[0][0])
    exp = lime_explainer.explain_instance(
        data_row=x,
        predict_fn=model.predict_proba,
        labels=(target_index,),
        num_features=x.shape[0],
        num_samples=num_samples,
    )
    attr = np.zeros(x.shape[0], dtype=float)
    local = exp.local_exp.get(target_index, [])
    for j, w in local:
        if 0 <= int(j) < attr.shape[0]:
            attr[int(j)] = float(w)
    return attr


class AIMEAdapter:
    """Thin wrapper around the official `aime-xai` AIME API.

    The package API used here is:

        from aime_xai import AIME
        AIME().create_explainer(X, Y, normalize=True)
        explainer.local_feature_importance(x, y, with_plot=False)

    For class-specific comparison with SHAP/LIME/FPDE, the default local output
    vector `y` is a one-hot vector for the model's predicted target class.  This
    asks AIME for the contribution toward that class-specific output direction.
    Use `--aime-local-y proba` to pass the model's full probability vector
    instead.
    """

    def __init__(
        self,
        explainer: Any,
        *,
        classes: Sequence[Any],
        feature_names: Sequence[str],
        local_y_mode: str = "onehot",
    ) -> None:
        self.explainer = explainer
        self.classes_ = np.asarray(classes, dtype=object)
        self.feature_names = [str(name) for name in feature_names]
        if local_y_mode not in ("onehot", "proba"):
            raise ValueError("local_y_mode must be 'onehot' or 'proba'")
        self.local_y_mode = local_y_mode

    def _local_output_vector(self, model: Any, x: np.ndarray, target_label: Any) -> np.ndarray:
        proba = np.asarray(model.predict_proba(x.reshape(1, -1))[0], dtype=float)
        if proba.ndim != 1 or proba.shape[0] != self.classes_.shape[0]:
            raise ValueError("predict_proba output shape is incompatible with model.classes_")
        if self.local_y_mode == "proba":
            return proba

        matches = np.where(self.classes_ == target_label)[0]
        if matches.size == 0:
            raise ValueError(f"target_label {target_label!r} is not in AIME classes {self.classes_.tolist()!r}")
        y_vec = np.zeros_like(proba, dtype=float)
        y_vec[int(matches[0])] = 1.0
        return y_vec

    def explain(self, model: Any, x: np.ndarray, target_label: Any) -> np.ndarray:
        y_vec = self._local_output_vector(model, x, target_label)
        df = self.explainer.local_feature_importance(
            x=np.asarray(x, dtype=float),
            y=y_vec,
            feature_names=self.feature_names,
            scale=True,
            top_k=None,
            ignore_zero_features=False,
            with_plot=False,
        )
        if hasattr(df, "loc") and "mean" in getattr(df, "index", []):
            values = df.loc["mean"].to_numpy(dtype=float)
        elif hasattr(df, "iloc"):
            values = df.iloc[0].to_numpy(dtype=float)
        else:
            values = np.asarray(df, dtype=float).reshape(-1)
        return np.asarray(values, dtype=float)


def build_aime_adapter(
    X_train: np.ndarray,
    model: Any,
    feature_names: Sequence[str],
    *,
    local_y_mode: str = "onehot",
) -> AIMEAdapter:
    """Fit official AIME on the transformed training representation.

    AIME learns an approximate inverse from black-box outputs `Y` to inputs `X`.
    In this experimental runner, `Y` is LightGBM's `predict_proba(X_train)`, so
    the same trained black-box used for perturbation evaluation defines AIME's
    output space.
    """
    try:
        from aime_xai import AIME
    except Exception as exc:
        raise MethodUnavailable(
            "AIME is not available. Install the official package and its imported "
            "dependencies, e.g. `pip install aime-xai umap-learn seaborn matplotlib`."
        ) from exc

    X_arr = np.asarray(X_train, dtype=float)
    Y_arr = np.asarray(model.predict_proba(X_arr), dtype=float)
    if X_arr.ndim != 2 or Y_arr.ndim != 2:
        raise ValueError("AIME requires 2D X_train and 2D predict_proba output")
    if X_arr.shape[0] != Y_arr.shape[0]:
        raise ValueError("AIME X/Y sample counts differ")
    if not np.all(np.isfinite(X_arr)) or not np.all(np.isfinite(Y_arr)):
        raise ValueError("AIME input contains NaN or infinity")

    explainer = AIME().create_explainer(X_arr, Y_arr, normalize=True)
    return AIMEAdapter(
        explainer,
        classes=np.asarray(model.classes_, dtype=object),
        feature_names=feature_names,
        local_y_mode=local_y_mode,
    )


def explain_aime(
    aime_adapter: Optional[AIMEAdapter],
    model: Any,
    x: np.ndarray,
    target_label: Any,
) -> np.ndarray:
    if aime_adapter is None:
        raise MethodUnavailable("AIME adapter could not be initialized")
    return aime_adapter.explain(model, x, target_label)


def evaluate_attribution(
    fpde: Any,
    model: Any,
    x: np.ndarray,
    attr: np.ndarray,
    target_label: Any,
    baseline: np.ndarray,
    fractions: Sequence[float],
) -> Dict[str, float]:
    curves = fpde.perturbation_curves(
        model,
        x,
        attr,
        target_label,
        baseline,
        fractions=fractions,
    )
    return {
        "p0": float(curves["p0"]),
        "deletion_auc": float(curves["deletion_auc"]),
        "deletion_drop_auc": float(curves["deletion_drop_auc"]),
        "insertion_auc": float(curves["insertion_auc"]),
        "combined_score": float(curves["combined_score"]),
    }


def run_one_task(
    task_id: int,
    *,
    cfg: RunConfig,
    fpde: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    task, X_df, y_ser, dataset_name = load_openml_task_dataframe(task_id)

    # Drop rows with missing target.
    mask = ~pd.isna(y_ser)
    X_df = X_df.loc[mask].reset_index(drop=True)
    y_ser = y_ser.loc[mask].reset_index(drop=True)

    le = LabelEncoder()
    y_all = le.fit_transform(y_ser.astype(str))
    if len(np.unique(y_all)) < 2:
        raise ValueError("task has fewer than two classes after filtering")

    train_idx, test_idx, split_name = split_task_data(
        task,
        X_df,
        y_all,
        fold=cfg.fold,
        repeat=cfg.repeat,
        sample=cfg.sample,
        seed=cfg.seed,
    )

    X_train_raw = X_df.iloc[train_idx].reset_index(drop=True)
    y_train_full = y_all[train_idx]
    X_test_raw = X_df.iloc[test_idx].reset_index(drop=True)
    y_test_full = y_all[test_idx]

    # Optional row caps for smoke tests.
    tr_keep = stratified_subsample_indices(y_train_full, cfg.max_train_rows, cfg.seed) if cfg.max_train_rows else np.arange(len(y_train_full))
    te_keep = stratified_subsample_indices(y_test_full, cfg.max_test_rows, cfg.seed + 1) if cfg.max_test_rows else np.arange(len(y_test_full))
    X_train_raw = X_train_raw.iloc[tr_keep].reset_index(drop=True)
    y_train_full = y_train_full[tr_keep]
    X_test_raw = X_test_raw.iloc[te_keep].reset_index(drop=True)
    y_test_full = y_test_full[te_keep]

    # Train/validation split inside the OpenML train split.
    train_inner_idx, val_inner_idx = train_test_split(
        np.arange(len(y_train_full)),
        test_size=0.2,
        random_state=cfg.seed,
        stratify=safe_stratify(y_train_full),
    )
    X_fit_raw = X_train_raw.iloc[train_inner_idx].reset_index(drop=True)
    y_fit = y_train_full[train_inner_idx]
    X_val_raw = X_train_raw.iloc[val_inner_idx].reset_index(drop=True)
    y_val = y_train_full[val_inner_idx]

    preprocessor = build_preprocessor(X_fit_raw)
    X_fit = to_dense_float_array(preprocessor.fit_transform(X_fit_raw))
    X_val = to_dense_float_array(preprocessor.transform(X_val_raw))
    X_test = to_dense_float_array(preprocessor.transform(X_test_raw))

    feature_names = get_feature_names(preprocessor, X_fit.shape[1])
    model = fit_lightgbm(
        X_fit,
        y_fit,
        seed=cfg.seed,
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        n_jobs=cfg.n_jobs,
        feature_names=feature_names,
    )
    perf = model_metrics(model, X_test, y_test_full)

    baseline = np.mean(X_fit, axis=0)
    prototypes, prototype_labels = fpde.class_mean_prototypes(X_fit, y_fit)
    anchor = np.mean(X_fit, axis=0)

    val_sel_idx = stratified_subsample_indices(y_val, min(cfg.n_val_select, len(y_val)), cfg.seed + 2)
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
                    "task_id": int(task_id),
                    "dataset_name": dataset_name,
                    "selected_lambda": float(selection.best_lambda),
                }
            )
            lambda_rows.append(r)

    explain_idx = stratified_subsample_indices(y_test_full, min(cfg.n_explain, len(y_test_full)), cfg.seed + 3)

    lime_explainer = None
    if "lime" in cfg.methods:
        try:
            class_names = [str(c) for c in le.classes_]
            lime_explainer = build_lime_explainer(X_fit, y_fit, feature_names, class_names)
        except MethodUnavailable:
            lime_explainer = None

    aime_adapter = None
    if "aime" in cfg.methods:
        try:
            aime_adapter = build_aime_adapter(
                X_fit,
                model,
                feature_names,
                local_y_mode=cfg.aime_local_y,
            )
        except MethodUnavailable:
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
                    attr, target_label_for_eval, detail = explain_fpde(
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
                    attr = explain_shap(model, x, target_label_for_eval, feature_names)
                elif method == "lime":
                    if lime_explainer is None:
                        raise MethodUnavailable("LIME explainer could not be initialized")
                    attr = explain_lime(
                        lime_explainer,
                        model,
                        x,
                        target_label_for_eval,
                        num_samples=cfg.lime_num_samples,
                    )
                elif method == "aime":
                    attr = explain_aime(aime_adapter, model, x, target_label_for_eval)
                else:
                    raise ValueError(f"unknown method: {method}")

                attr = np.asarray(attr, dtype=float)
                if attr.shape != x.shape:
                    raise ValueError(f"attribution shape {attr.shape} differs from x shape {x.shape}")
                if not np.all(np.isfinite(attr)):
                    raise ValueError("attribution contains NaN or infinity")
                attr_norm_l1 = float(np.sum(np.abs(attr)))
                metrics = evaluate_attribution(fpde, model, x, attr, target_label_for_eval, baseline, cfg.fractions)
            except Exception as exc:
                status = "error"
                error = f"{type(exc).__name__}: {exc}"

            elapsed = float(time.perf_counter() - start)
            row = {
                "task_id": int(task_id),
                "dataset_name": dataset_name,
                "split_name": split_name,
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

    return per_instance_rows, lambda_rows


def _mean_abs_nan_safe(values: pd.Series) -> float:
    """Mean absolute value that returns NaN for all-missing groups without warnings."""
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(np.abs(arr)))


def aggregate_results(per_instance: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if per_instance.empty or "status" not in per_instance.columns:
        return pd.DataFrame(), pd.DataFrame()

    ok = per_instance[per_instance["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame(), pd.DataFrame()

    agg_cols = {
        "deletion_drop_auc": "mean",
        "insertion_auc": "mean",
        "combined_score": "mean",
        "runtime_seconds": "mean",
        "attr_l1_norm": "mean",
        "exactness_residual": _mean_abs_nan_safe,
        "test_accuracy": "mean",
        "test_balanced_accuracy": "mean",
        "n_features_transformed": "first",
        "n_classes": "first",
    }
    by_task = ok.groupby(["task_id", "dataset_name", "method"], as_index=False).agg(agg_cols)
    by_task = by_task.rename(columns={"exactness_residual": "mean_abs_exactness_residual"})
    status_counts = per_instance.groupby(["task_id", "dataset_name", "method", "status"]).size().unstack(fill_value=0)
    status_counts = status_counts.reset_index()
    by_task = by_task.merge(status_counts, on=["task_id", "dataset_name", "method"], how="left")

    by_method = ok.groupby(["method"], as_index=False).agg(
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
        mean_test_accuracy=("test_accuracy", "mean"),
        mean_test_balanced_accuracy=("test_balanced_accuracy", "mean"),
    )
    return by_task, by_method


def dataframe_to_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    if df.empty:
        path.write_text("% No results available.\n", encoding="utf-8")
        return
    latex = df.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.4f}")
    wrapped = (
        "\\begin{table*}[t]\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\centering\n"
        f"{latex}\n"
        "\\end{table*}\n"
    )
    path.write_text(wrapped, encoding="utf-8")


def write_outputs(
    output_dir: Path,
    cfg: RunConfig,
    per_instance_rows: List[Dict[str, Any]],
    lambda_rows: List[Dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_instance = pd.DataFrame(per_instance_rows)
    lambdas = pd.DataFrame(lambda_rows)
    by_task, by_method = aggregate_results(per_instance)

    per_instance.to_csv(output_dir / "per_instance_results.csv", index=False)
    by_task.to_csv(output_dir / "summary_by_task.csv", index=False)
    by_method.to_csv(output_dir / "summary_by_method.csv", index=False)
    lambdas.to_csv(output_dir / "lambda_distribution.csv", index=False)
    (output_dir / "run_config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")

    main_cols = [
        "method",
        "mean_deletion_drop_auc",
        "mean_insertion_auc",
        "mean_combined_score",
        "mean_runtime_seconds",
        "completed_instances",
        "completed_tasks",
    ]
    if not by_method.empty:
        main_df = by_method[[c for c in main_cols if c in by_method.columns]].copy()
    else:
        main_df = pd.DataFrame(columns=main_cols)
    dataframe_to_latex_table(
        main_df,
        output_dir / "main_results_table.tex",
        "Experimental results. Values are computed from per-instance perturbation curves.",
        "tab:main_results_computed",
    )

    if not lambdas.empty and "lambda_hyb" in lambdas.columns:
        lambda_ok = lambdas[lambdas.get("status", "ok") == "ok"].copy()
        if not lambda_ok.empty:
            lambda_table = (
                lambda_ok.groupby("lambda_hyb", as_index=False)
                .agg(mean_score=("score", "mean"), n_rows=("score", "count"))
                .sort_values("lambda_hyb")
            )
        else:
            lambda_table = pd.DataFrame()
    else:
        lambda_table = pd.DataFrame()
    dataframe_to_latex_table(
        lambda_table,
        output_dir / "lambda_distribution_table.tex",
        "Validation scores for Hyb-FPDE-grid lambda candidates.",
        "tab:lambda_distribution_computed",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run FPDE experimental evaluation on OpenML-CC18 with LightGBM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--suite-id", type=int, default=99, help="OpenML benchmark suite id. OpenML-CC18 is suite 99.")
    p.add_argument("--task-ids", type=str, default=None, help="Comma-separated OpenML task IDs. Overrides suite task list when given.")
    p.add_argument("--max-tasks", type=int, default=3, help="Maximum number of tasks for a smoke run. Use 72 or omit for full CC18.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--n-explain", type=int, default=20, help="Number of test instances explained per task.")
    p.add_argument("--n-val-select", type=int, default=20, help="Validation instances used to select Hyb-FPDE-grid lambda.")
    p.add_argument("--max-train-rows", type=int, default=5000, help="Optional cap for smoke runs. Set 0 to disable.")
    p.add_argument("--max-test-rows", type=int, default=2000, help="Optional cap for smoke runs. Set 0 to disable.")
    p.add_argument("--lambda-grid", type=str, default=LAMBDA_GRID_DEFAULT)
    p.add_argument("--fractions", type=str, default=FRACTIONS_DEFAULT)
    p.add_argument("--methods", type=str, default=",".join(METHODS_DEFAULT), help="Comma-separated method names.")
    p.add_argument("--output-dir", type=str, default="experimental_outputs")
    p.add_argument(
        "--fpde-path",
        type=str,
        default=FPDE_WHEEL_DEFAULT,
        help="Path to the fpde-xai wheel or a legacy FPDE.py module. Relative paths are resolved from the current directory or this runner's directory.",
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

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    allowed = {"diff_fpde", "cos_fpde", "hyb_fpde_grid", "shap", "lime", "aime"}
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise ValueError(f"unknown methods: {unknown}; allowed={sorted(allowed)}")

    fpde_path = Path(args.fpde_path)
    if not fpde_path.is_absolute():
        # First try current working directory, then the script directory.
        if not fpde_path.exists():
            script_dir_candidate = Path(__file__).resolve().parent / fpde_path
            if script_dir_candidate.exists():
                fpde_path = script_dir_candidate

    cfg = RunConfig(
        suite_id=int(args.suite_id),
        task_ids=parse_int_list_or_none(args.task_ids),
        max_tasks=None if args.max_tasks is None or args.max_tasks <= 0 else int(args.max_tasks),
        seed=int(args.seed),
        fold=int(args.fold),
        repeat=int(args.repeat),
        sample=int(args.sample),
        n_explain=int(args.n_explain),
        n_val_select=int(args.n_val_select),
        max_train_rows=None if args.max_train_rows is None or args.max_train_rows <= 0 else int(args.max_train_rows),
        max_test_rows=None if args.max_test_rows is None or args.max_test_rows <= 0 else int(args.max_test_rows),
        lambda_grid=parse_float_list(args.lambda_grid),
        fractions=parse_float_list(args.fractions),
        methods=methods,
        output_dir=str(args.output_dir),
        fpde_path=str(fpde_path),
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        n_jobs=int(args.n_jobs),
        lime_num_samples=int(args.lime_num_samples),
        aime_local_y=str(args.aime_local_y),
    )

    fpde = import_fpde_module(fpde_path)

    task_ids = cfg.task_ids if cfg.task_ids is not None else get_suite_task_ids(cfg.suite_id)
    if cfg.max_tasks is not None:
        task_ids = task_ids[: cfg.max_tasks]

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_instance_rows: List[Dict[str, Any]] = []
    lambda_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    print(f"Running {len(task_ids)} task(s): {task_ids}")
    for order, task_id in enumerate(task_ids, start=1):
        print(f"[{order}/{len(task_ids)}] task_id={task_id}", flush=True)
        try:
            rows, lam_rows = run_one_task(int(task_id), cfg=cfg, fpde=fpde)
            per_instance_rows.extend(rows)
            lambda_rows.extend(lam_rows)
            write_outputs(output_dir, cfg, per_instance_rows, lambda_rows)
        except Exception as exc:
            failures.append(
                {
                    "task_id": int(task_id),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            pd.DataFrame(failures).to_csv(output_dir / "task_failures.csv", index=False)
            print(f"  ERROR: {type(exc).__name__}: {exc}", flush=True)

    write_outputs(output_dir, cfg, per_instance_rows, lambda_rows)
    if failures:
        pd.DataFrame(failures).to_csv(output_dir / "task_failures.csv", index=False)
    print(f"Done. Outputs written to: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
