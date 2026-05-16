#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FPDE.py

Clean FPDE core for:

    - Diff-FPDE
    - Cos-FPDE
    - Hyb-FPDE-grid over lambda_hyb = 0.0, 0.1, ..., 1.0 by default
    - Hyb-FPDE-grid validation-based lambda_hyb selection

Standalone fixed-lambda Hyb-FPDE, Adaptive-Hyb-FPDE, and all Multi-Prototype
extensions are intentionally excluded. The hybrid formula is kept as a grid
candidate generator for Hyb-FPDE-grid.

Default recommendation for selecting lambda_hyb:

    select_lambda_by_deletion_insertion_validation(...)

which chooses lambda_hyb on a held-out validation subset by maximizing:

    combined_score = 0.5 * (deletion_drop_auc + insertion_auc)

The strict additive guarantee is kept for every explanation:

    explanation.evidence == sum(explanation.attributions)

up to floating-point roundoff.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np


Mode = Literal["diff", "cos"]
GridMode = Literal["diff", "cos", "hyb_grid"]
NormalizeMode = Literal["none", "l1"]
AnchorStrategy = Literal["mean", "zero", "none"]
GridObjective = Literal[
    "blackbox_agreement",
    "mean_positive_evidence",
    "mean_margin_weighted_evidence",
]


@dataclass(frozen=True)
class FPDEExplanation:
    """Result object for a strict FPDE explanation."""

    mode: GridMode
    evidence: float
    attributions: np.ndarray
    positive_score: float
    negative_score: float
    positive_label: Any
    negative_label: Any
    positive_prototype_index: int
    negative_prototype_index: int
    exactness_residual: float
    details: Dict[str, Any]

    @property
    def normalized_attributions(self) -> np.ndarray:
        """Return an L1-normalized attribution vector for visualization only."""
        denom = float(np.sum(np.abs(self.attributions)))
        if denom <= 0.0 or not np.isfinite(denom):
            return np.zeros_like(self.attributions, dtype=float)
        return self.attributions / denom


def _as_1d_float(name: str, x: np.ndarray | Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector, got shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf")
    return arr


def _as_2d_float(name: str, x: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf")
    return arr


def _check_same_dim(*vectors: np.ndarray) -> None:
    if not vectors:
        return
    d = vectors[0].shape[0]
    for i, v in enumerate(vectors):
        if v.shape[0] != d:
            raise ValueError(f"dimension mismatch at vector {i}: expected {d}, got {v.shape[0]}")


def _regularized_norm(v: np.ndarray, eps: float) -> float:
    return float(np.sqrt(np.dot(v, v) + eps * eps))


def _l1_denom(v: np.ndarray, eps: float) -> float:
    denom = float(np.sum(np.abs(v)))
    if denom <= eps or not np.isfinite(denom):
        return 0.0
    return denom


def _scaled_explanation_parts(
    exp: FPDEExplanation,
    *,
    normalize: NormalizeMode,
    eps: float,
) -> Tuple[np.ndarray, float, float, float, float]:
    """Return scaled attribution/evidence/scores/scale for hybridization."""
    if normalize == "none":
        return (
            exp.attributions.astype(float, copy=True),
            float(exp.evidence),
            float(exp.positive_score),
            float(exp.negative_score),
            1.0,
        )

    if normalize == "l1":
        scale = _l1_denom(exp.attributions, eps=eps)
        if scale == 0.0:
            zeros = np.zeros_like(exp.attributions, dtype=float)
            return zeros, 0.0, 0.0, 0.0, 0.0
        return (
            exp.attributions / scale,
            float(exp.evidence / scale),
            float(exp.positive_score / scale),
            float(exp.negative_score / scale),
            scale,
        )

    raise ValueError(f"unknown normalize={normalize!r}; expected 'none' or 'l1'")


def regularized_cosine(u: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> float:
    """Cosine similarity with epsilon-regularized norms."""
    u = _as_1d_float("u", u)
    v = _as_1d_float("v", v)
    _check_same_dim(u, v)
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    return float(np.dot(u, v) / (_regularized_norm(u, eps) * _regularized_norm(v, eps)))


def diff_fpde(
    x: np.ndarray | Sequence[float],
    p_pos: np.ndarray | Sequence[float],
    p_neg: np.ndarray | Sequence[float],
    *,
    positive_label: Any = "positive",
    negative_label: Any = "negative",
    positive_prototype_index: int = -1,
    negative_prototype_index: int = -1,
) -> FPDEExplanation:
    """Strict Diff-FPDE for one positive/negative prototype pair.

    Evidence:

        E_diff = ||x - p_neg||^2 - ||x - p_pos||^2

    Feature attribution:

        phi_j = (x_j - p_neg_j)^2 - (x_j - p_pos_j)^2
    """
    x_arr = _as_1d_float("x", x)
    p_pos_arr = _as_1d_float("p_pos", p_pos)
    p_neg_arr = _as_1d_float("p_neg", p_neg)
    _check_same_dim(x_arr, p_pos_arr, p_neg_arr)

    pos_sqdist_by_feature = (x_arr - p_pos_arr) ** 2
    neg_sqdist_by_feature = (x_arr - p_neg_arr) ** 2

    attributions = neg_sqdist_by_feature - pos_sqdist_by_feature

    pos_score = -float(np.sum(pos_sqdist_by_feature))
    neg_score = -float(np.sum(neg_sqdist_by_feature))
    evidence = float(np.sum(attributions))

    direct_evidence = float(np.sum(neg_sqdist_by_feature) - np.sum(pos_sqdist_by_feature))
    residual = float(evidence - direct_evidence)

    return FPDEExplanation(
        mode="diff",
        evidence=evidence,
        attributions=attributions,
        positive_score=pos_score,
        negative_score=neg_score,
        positive_label=positive_label,
        negative_label=negative_label,
        positive_prototype_index=int(positive_prototype_index),
        negative_prototype_index=int(negative_prototype_index),
        exactness_residual=residual,
        details={
            "positive_squared_distance": float(np.sum(pos_sqdist_by_feature)),
            "negative_squared_distance": float(np.sum(neg_sqdist_by_feature)),
            "definition": "E_diff = ||x-p_neg||^2 - ||x-p_pos||^2",
        },
    )


def cos_fpde(
    x: np.ndarray | Sequence[float],
    p_pos: np.ndarray | Sequence[float],
    p_neg: np.ndarray | Sequence[float],
    *,
    anchor: Optional[np.ndarray | Sequence[float]] = None,
    eps: float = 1e-12,
    positive_label: Any = "positive",
    negative_label: Any = "negative",
    positive_prototype_index: int = -1,
    negative_prototype_index: int = -1,
) -> FPDEExplanation:
    """Strict Cos-FPDE for one positive/negative prototype pair.

    Let z = x - anchor, q_pos = p_pos - anchor, q_neg = p_neg - anchor.

    Evidence:

        E_cos = cos_eps(z, q_pos) - cos_eps(z, q_neg)

    Feature attribution:

        phi_j = z_j q_pos_j / (N_z N_pos) - z_j q_neg_j / (N_z N_neg)

    This is an exact coordinate decomposition of the cosine contrast itself.
    It is not a leave-one-feature-out causal effect because the cosine norm
    denominator couples all coordinates.
    """
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    x_arr = _as_1d_float("x", x)
    p_pos_arr = _as_1d_float("p_pos", p_pos)
    p_neg_arr = _as_1d_float("p_neg", p_neg)
    _check_same_dim(x_arr, p_pos_arr, p_neg_arr)

    if anchor is None:
        anchor_arr = np.zeros_like(x_arr, dtype=float)
    else:
        anchor_arr = _as_1d_float("anchor", anchor)
        _check_same_dim(x_arr, anchor_arr)

    z = x_arr - anchor_arr
    q_pos = p_pos_arr - anchor_arr
    q_neg = p_neg_arr - anchor_arr

    n_z = _regularized_norm(z, eps)
    n_pos = _regularized_norm(q_pos, eps)
    n_neg = _regularized_norm(q_neg, eps)

    pos_by_feature = (z * q_pos) / (n_z * n_pos)
    neg_by_feature = (z * q_neg) / (n_z * n_neg)

    attributions = pos_by_feature - neg_by_feature

    pos_score = float(np.sum(pos_by_feature))
    neg_score = float(np.sum(neg_by_feature))
    evidence = float(np.sum(attributions))

    direct_evidence = regularized_cosine(z, q_pos, eps=eps) - regularized_cosine(z, q_neg, eps=eps)
    residual = float(evidence - direct_evidence)

    return FPDEExplanation(
        mode="cos",
        evidence=evidence,
        attributions=attributions,
        positive_score=pos_score,
        negative_score=neg_score,
        positive_label=positive_label,
        negative_label=negative_label,
        positive_prototype_index=int(positive_prototype_index),
        negative_prototype_index=int(negative_prototype_index),
        exactness_residual=residual,
        details={
            "anchor": anchor_arr.copy(),
            "eps": float(eps),
            "norm_x_anchor": float(n_z),
            "norm_positive_anchor": float(n_pos),
            "norm_negative_anchor": float(n_neg),
            "definition": "E_cos = cos_eps(x-a,p_pos-a) - cos_eps(x-a,p_neg-a)",
        },
    )


def _hyb_fpde_grid_candidate(
    x: np.ndarray | Sequence[float],
    p_pos: np.ndarray | Sequence[float],
    p_neg: np.ndarray | Sequence[float],
    *,
    anchor: Optional[np.ndarray | Sequence[float]] = None,
    lambda_hyb: float = 0.5,
    normalize: NormalizeMode = "l1",
    eps: float = 1e-12,
    positive_label: Any = "positive",
    negative_label: Any = "negative",
    positive_prototype_index: int = -1,
    negative_prototype_index: int = -1,
) -> FPDEExplanation:
    """Internal Hyb-FPDE-grid candidate for one positive/negative prototype pair.

    Standalone fixed-lambda Hyb-FPDE is intentionally not exposed as a public
    method. This helper exists only so Hyb-FPDE-grid can evaluate each fixed
    lambda_hyb candidate.

    The grid candidate combines Diff-FPDE and Cos-FPDE at the attribution level:

        phi_grid_j = lambda_hyb * phi_diff_j' + (1 - lambda_hyb) * phi_cos_j'

    If normalize="l1", each component attribution vector is divided by its L1
    norm before mixing. If normalize="none", raw component attributions are
    mixed directly.
    """
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if not np.isfinite(lambda_hyb):
        raise ValueError("lambda_hyb must be finite")
    if lambda_hyb < 0.0 or lambda_hyb > 1.0:
        raise ValueError("lambda_hyb must be in [0, 1]")
    if normalize not in ("none", "l1"):
        raise ValueError("normalize must be either 'none' or 'l1'")

    diff_exp = diff_fpde(
        x,
        p_pos,
        p_neg,
        positive_label=positive_label,
        negative_label=negative_label,
        positive_prototype_index=positive_prototype_index,
        negative_prototype_index=negative_prototype_index,
    )
    cos_exp = cos_fpde(
        x,
        p_pos,
        p_neg,
        anchor=anchor,
        eps=eps,
        positive_label=positive_label,
        negative_label=negative_label,
        positive_prototype_index=positive_prototype_index,
        negative_prototype_index=negative_prototype_index,
    )

    diff_attr, diff_ev, diff_pos, diff_neg, diff_scale = _scaled_explanation_parts(
        diff_exp,
        normalize=normalize,
        eps=eps,
    )
    cos_attr, cos_ev, cos_pos, cos_neg, cos_scale = _scaled_explanation_parts(
        cos_exp,
        normalize=normalize,
        eps=eps,
    )

    w_diff = float(lambda_hyb)
    w_cos = float(1.0 - lambda_hyb)

    attributions = w_diff * diff_attr + w_cos * cos_attr
    evidence = float(np.sum(attributions))

    positive_score = float(w_diff * diff_pos + w_cos * cos_pos)
    negative_score = float(w_diff * diff_neg + w_cos * cos_neg)
    direct_grid_evidence = float(w_diff * diff_ev + w_cos * cos_ev)
    residual = float(evidence - direct_grid_evidence)

    return FPDEExplanation(
        mode="hyb_grid",
        evidence=evidence,
        attributions=attributions,
        positive_score=positive_score,
        negative_score=negative_score,
        positive_label=positive_label,
        negative_label=negative_label,
        positive_prototype_index=int(positive_prototype_index),
        negative_prototype_index=int(negative_prototype_index),
        exactness_residual=residual,
        details={
            "method_family": "hyb_fpde_grid",
            "lambda_hyb": float(lambda_hyb),
            "normalize": normalize,
            "eps": float(eps),
            "definition": "phi_grid_j = lambda_hyb * phi_diff_j' + (1-lambda_hyb) * phi_cos_j'",
            "diff_evidence_raw": float(diff_exp.evidence),
            "cos_evidence_raw": float(cos_exp.evidence),
            "diff_evidence_scaled": float(diff_ev),
            "cos_evidence_scaled": float(cos_ev),
            "diff_scale": float(diff_scale),
            "cos_scale": float(cos_scale),
            "diff_exactness_residual": float(diff_exp.exactness_residual),
            "cos_exactness_residual": float(cos_exp.exactness_residual),
            "selection_note": "Prototype indices are supplied by the caller; Hyb-FPDE-grid uses diff selection for lambda candidates by default.",
        },
    )

def select_prototype_pair(
    x: np.ndarray | Sequence[float],
    prototypes: np.ndarray | Sequence[Sequence[float]],
    prototype_labels: Sequence[Any],
    *,
    positive_label: Any,
    negative_label: Optional[Any] = None,
    mode: GridMode = "diff",
    anchor: Optional[np.ndarray | Sequence[float]] = None,
    eps: float = 1e-12,
) -> Tuple[int, int]:
    """Select a positive and negative prototype for a local contrast.

    Diff mode:
        Select nearest positive and nearest negative prototype by squared distance.

    Cos mode:
        Select most cosine-similar positive and negative prototype.

    Hyb-FPDE-grid mode:
        Use the Diff selection rule by default. The lambda-grid hybridization is
        applied to the attribution formula, not to prototype selection.
    """
    x_arr = _as_1d_float("x", x)
    P = _as_2d_float("prototypes", prototypes)
    labels = np.asarray(list(prototype_labels), dtype=object)

    if P.shape[0] != labels.shape[0]:
        raise ValueError(f"number of prototypes and labels differ: {P.shape[0]} vs {labels.shape[0]}")
    if P.shape[1] != x_arr.shape[0]:
        raise ValueError(f"prototype dimension mismatch: expected {x_arr.shape[0]}, got {P.shape[1]}")

    pos_indices = np.where(labels == positive_label)[0]
    if pos_indices.size == 0:
        raise ValueError(f"no prototype found for positive_label={positive_label!r}")

    if negative_label is None:
        neg_indices = np.where(labels != positive_label)[0]
        if neg_indices.size == 0:
            raise ValueError("negative_label is None, but no non-positive prototypes exist")
    else:
        neg_indices = np.where(labels == negative_label)[0]
        if neg_indices.size == 0:
            raise ValueError(f"no prototype found for negative_label={negative_label!r}")

    if mode in ("diff", "hyb_grid"):
        pos_d = np.sum((P[pos_indices] - x_arr[None, :]) ** 2, axis=1)
        neg_d = np.sum((P[neg_indices] - x_arr[None, :]) ** 2, axis=1)
        pos_idx = int(pos_indices[int(np.argmin(pos_d))])
        neg_idx = int(neg_indices[int(np.argmin(neg_d))])
        return pos_idx, neg_idx

    if mode == "cos":
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if anchor is None:
            anchor_arr = np.zeros_like(x_arr, dtype=float)
        else:
            anchor_arr = _as_1d_float("anchor", anchor)
            _check_same_dim(x_arr, anchor_arr)

        z = x_arr - anchor_arr
        q = P - anchor_arr[None, :]

        n_z = _regularized_norm(z, eps)
        q_norms = np.sqrt(np.sum(q * q, axis=1) + eps * eps)
        scores = (q @ z) / (q_norms * n_z)

        pos_idx = int(pos_indices[int(np.argmax(scores[pos_indices]))])
        neg_idx = int(neg_indices[int(np.argmax(scores[neg_indices]))])
        return pos_idx, neg_idx

    raise ValueError(f"unknown mode={mode!r}")


def explain_with_selected_prototypes(
    x: np.ndarray | Sequence[float],
    prototypes: np.ndarray | Sequence[Sequence[float]],
    prototype_labels: Sequence[Any],
    *,
    positive_label: Any,
    negative_label: Optional[Any] = None,
    mode: Mode = "diff",
    anchor: Optional[np.ndarray | Sequence[float]] = None,
    eps: float = 1e-12,
) -> FPDEExplanation:
    """Select prototypes and compute a public Diff-FPDE or Cos-FPDE explanation.

    Standalone fixed-lambda Hyb-FPDE has been removed. Use
    explain_hyb_fpde_grid() or grid_search_hyb_fpde_grid() for the retained
    lambda-grid condition.
    """
    if mode not in ("diff", "cos"):
        raise ValueError(
            "standalone Hyb-FPDE has been removed; use explain_hyb_fpde_grid() "
            "or grid_search_hyb_fpde_grid() for Hyb-FPDE-grid"
        )
    return _explain_with_selected_prototypes_for_grid(
        x,
        prototypes,
        prototype_labels,
        positive_label=positive_label,
        negative_label=negative_label,
        mode=mode,
        anchor=anchor,
        lambda_hyb=0.5,
        normalize="l1",
        eps=eps,
    )


def _explain_with_selected_prototypes_for_grid(
    x: np.ndarray | Sequence[float],
    prototypes: np.ndarray | Sequence[Sequence[float]],
    prototype_labels: Sequence[Any],
    *,
    positive_label: Any,
    negative_label: Optional[Any] = None,
    mode: GridMode = "diff",
    anchor: Optional[np.ndarray | Sequence[float]] = None,
    lambda_hyb: float = 0.5,
    normalize: NormalizeMode = "l1",
    eps: float = 1e-12,
) -> FPDEExplanation:
    """Internal dispatcher for Diff, Cos, and Hyb-FPDE-grid lambda candidates."""
    P = _as_2d_float("prototypes", prototypes)
    labels = np.asarray(list(prototype_labels), dtype=object)

    pos_idx, neg_idx = select_prototype_pair(
        x,
        P,
        labels,
        positive_label=positive_label,
        negative_label=negative_label,
        mode=mode,
        anchor=anchor,
        eps=eps,
    )

    neg_label_value = labels[neg_idx]

    if mode == "diff":
        return diff_fpde(
            x,
            P[pos_idx],
            P[neg_idx],
            positive_label=positive_label,
            negative_label=neg_label_value,
            positive_prototype_index=pos_idx,
            negative_prototype_index=neg_idx,
        )

    if mode == "cos":
        return cos_fpde(
            x,
            P[pos_idx],
            P[neg_idx],
            anchor=anchor,
            eps=eps,
            positive_label=positive_label,
            negative_label=neg_label_value,
            positive_prototype_index=pos_idx,
            negative_prototype_index=neg_idx,
        )

    if mode == "hyb_grid":
        return _hyb_fpde_grid_candidate(
            x,
            P[pos_idx],
            P[neg_idx],
            anchor=anchor,
            lambda_hyb=lambda_hyb,
            normalize=normalize,
            eps=eps,
            positive_label=positive_label,
            negative_label=neg_label_value,
            positive_prototype_index=pos_idx,
            negative_prototype_index=neg_idx,
        )

    raise ValueError(f"unknown mode={mode!r}")

def class_mean_prototypes(
    X: np.ndarray | Sequence[Sequence[float]],
    y: Sequence[Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build one mean prototype per class."""
    X_arr = _as_2d_float("X", X)
    y_arr = np.asarray(list(y), dtype=object)
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(f"number of samples and labels differ: {X_arr.shape[0]} vs {y_arr.shape[0]}")

    classes = np.array(sorted(set(y_arr.tolist())), dtype=object)
    prototypes = []
    for c in classes:
        rows = X_arr[y_arr == c]
        if rows.size == 0:
            raise ValueError(f"class {c!r} has no samples")
        prototypes.append(np.mean(rows, axis=0))
    return np.vstack(prototypes), classes


def _get_predict_proba_vector(
    predictor: Optional[Any],
    x: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Evaluate a predict_proba-like object or callable for one sample."""
    if predictor is None:
        return None, None

    if hasattr(predictor, "predict_proba"):
        raw = predictor.predict_proba(x.reshape(1, -1))
        labels = getattr(predictor, "classes_", None)
    elif callable(predictor):
        raw = predictor(x.reshape(1, -1))
        labels = getattr(predictor, "classes_", None)
    else:
        raise ValueError("predictor must be None, a predict_proba callable, or an object with predict_proba")

    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 2:
        if arr.shape[0] != 1:
            raise ValueError(f"predict_proba for one sample must have one row, got shape={arr.shape}")
        arr = arr[0]
    elif arr.ndim != 1:
        raise ValueError(f"predict_proba output must be 1D or 2D, got shape={arr.shape}")
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("predict_proba output is empty or contains NaN/inf")

    if labels is None:
        return arr.astype(float, copy=False), None
    labels_arr = np.asarray(labels, dtype=object)
    if labels_arr.shape[0] != arr.shape[0]:
        raise ValueError(
            f"predictor.classes_ length and predict_proba width differ: {labels_arr.shape[0]} vs {arr.shape[0]}"
        )
    return arr.astype(float, copy=False), labels_arr


def _nearest_label_from_prototypes(
    x: np.ndarray,
    prototypes: np.ndarray,
    prototype_labels: np.ndarray,
) -> Any:
    d = np.sum((prototypes - x[None, :]) ** 2, axis=1)
    return prototype_labels[int(np.argmin(d))]


def _probability_for_label(
    label: Any,
    probabilities: Optional[np.ndarray],
    probability_labels: Optional[np.ndarray],
    fallback_labels: np.ndarray,
) -> Optional[float]:
    if probabilities is None:
        return None
    labels = probability_labels if probability_labels is not None else fallback_labels
    if labels.shape[0] != probabilities.shape[0]:
        return None
    matches = np.where(labels == label)[0]
    if matches.size == 0:
        return None
    return float(probabilities[int(matches[0])])


def _default_lambda_hyb_grid(step: float = 0.1) -> Tuple[float, ...]:
    """Return a deterministic inclusive lambda_hyb grid from 0.0 to 1.0."""
    step = float(step)
    if not np.isfinite(step) or step <= 0.0 or step > 1.0:
        raise ValueError("step must be finite and in (0, 1]")
    n_steps = int(round(1.0 / step))
    values = [i * step for i in range(n_steps + 1)]
    values[0] = 0.0
    values[-1] = 1.0
    return tuple(float(round(v, 12)) for v in values)


@dataclass(frozen=True)
class HybFPDEGridSearchResult:
    """Result of an exhaustive grid search over Diff/Cos/Hyb-FPDE-grid settings.

    Report this condition as "Hyb-FPDE-grid" or "Hyb-FPDE (lambda grid)".
    Do not describe it as Adaptive-Hyb-FPDE, because lambda_hyb is selected by
    grid search rather than computed sample-wise.
    """

    best_config: Dict[str, Any]
    best_score: float
    rows: Tuple[Dict[str, Any], ...]
    objective: GridObjective
    n_candidates: int
    n_eval_samples: int

    def sorted_rows(self) -> List[Dict[str, Any]]:
        """Return candidate rows sorted from best to worst."""
        return sorted(
            (dict(row) for row in self.rows),
            key=lambda r: (float(r["score"]), float(r.get("mean_evidence", 0.0))),
            reverse=True,
        )


def _hyb_fpde_grid_candidate_dicts(
    *,
    fpde_mode_grid: Sequence[GridMode] = ("diff", "cos", "hyb_grid"),
    normalize_grid: Sequence[NormalizeMode] = ("l1",),
    lambda_hyb_grid: Sequence[float] = _default_lambda_hyb_grid(0.1),
    anchor_strategy_grid: Sequence[AnchorStrategy] = ("mean",),
    include_explicit_diff_cos: bool = True,
) -> List[Dict[str, Any]]:
    """Create exhaustive candidate dictionaries for Diff/Cos/Hyb-FPDE-grid search."""
    lambdas: List[float] = []
    for lam in lambda_hyb_grid:
        value = float(lam)
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("lambda_hyb_grid values must be finite and in [0, 1]")
        lambdas.append(value)
    if not lambdas:
        raise ValueError("lambda_hyb_grid must contain at least one value")

    requested_modes = tuple(fpde_mode_grid)
    for mode in requested_modes:
        if mode not in ("diff", "cos", "hyb_grid"):
            raise ValueError("fpde_mode_grid values must be 'diff', 'cos', or 'hyb_grid'")

    candidates: List[Dict[str, Any]] = []
    for normalize in normalize_grid:
        if normalize not in ("none", "l1"):
            raise ValueError("normalize_grid values must be 'none' or 'l1'")
        for anchor_strategy in anchor_strategy_grid:
            if anchor_strategy not in ("mean", "zero", "none"):
                raise ValueError("anchor_strategy_grid values must be 'mean', 'zero', or 'none'")
            if include_explicit_diff_cos:
                if "diff" in requested_modes:
                    candidates.append(
                        {
                            "fpde_mode": "diff",
                            "method_variant": "diff_fpde",
                            "normalize": normalize,
                            "lambda_hyb": None,
                            "lambda_grid_mode": "not_applicable",
                            "anchor_strategy": anchor_strategy,
                        }
                    )
                if "cos" in requested_modes:
                    candidates.append(
                        {
                            "fpde_mode": "cos",
                            "method_variant": "cos_fpde",
                            "normalize": normalize,
                            "lambda_hyb": None,
                            "lambda_grid_mode": "not_applicable",
                            "anchor_strategy": anchor_strategy,
                        }
                    )
            if "hyb_grid" in requested_modes:
                for lambda_hyb in lambdas:
                    if lambda_hyb == 0.0:
                        variant = "hyb_fpde_grid_lambda_0_cos_endpoint"
                    elif lambda_hyb == 1.0:
                        variant = "hyb_fpde_grid_lambda_1_diff_endpoint"
                    else:
                        variant = "hyb_fpde_grid"
                    candidates.append(
                        {
                            "fpde_mode": "hyb_grid",
                            "method_variant": variant,
                            "normalize": normalize,
                            "lambda_hyb": float(lambda_hyb),
                            "lambda_grid_mode": "grid_candidate_lambda_hyb",
                            "anchor_strategy": anchor_strategy,
                        }
                    )
    return candidates


def _label_contrast_for_grid_sample(
    x: np.ndarray,
    *,
    prototypes: np.ndarray,
    prototype_labels: np.ndarray,
    classes: np.ndarray,
    predictor: Optional[Any],
) -> Tuple[Any, Optional[Any], Optional[np.ndarray], Optional[np.ndarray]]:
    """Choose positive and rival labels for one grid-search sample."""
    probabilities, probability_labels = _get_predict_proba_vector(predictor, x)
    if probabilities is None:
        return _nearest_label_from_prototypes(x, prototypes, prototype_labels), None, None, None

    labels_for_prob = probability_labels if probability_labels is not None else classes
    if labels_for_prob.shape[0] != probabilities.shape[0]:
        raise ValueError("cannot infer labels: probability label length mismatch")

    order = np.argsort(probabilities)[::-1]
    pos = labels_for_prob[int(order[0])]
    neg: Optional[Any] = None
    for idx in order[1:].tolist():
        cand = labels_for_prob[int(idx)]
        if cand != pos and np.where(prototype_labels == cand)[0].size > 0:
            neg = cand
            break
    return pos, neg, probabilities, probability_labels


def _explain_single_prototype_candidate(
    x: np.ndarray,
    *,
    prototypes: np.ndarray,
    prototype_labels: np.ndarray,
    classes: np.ndarray,
    anchor: np.ndarray,
    predictor: Optional[Any],
    cfg: Dict[str, Any],
    eps: float,
) -> FPDEExplanation:
    """Explain one sample for one single-prototype grid candidate."""
    pos, neg, probabilities, probability_labels = _label_contrast_for_grid_sample(
        x,
        prototypes=prototypes,
        prototype_labels=prototype_labels,
        classes=classes,
        predictor=predictor,
    )
    exp = _explain_with_selected_prototypes_for_grid(
        x,
        prototypes,
        prototype_labels,
        positive_label=pos,
        negative_label=neg,
        mode=cfg["fpde_mode"],
        anchor=anchor,
        lambda_hyb=0.5 if cfg.get("lambda_hyb") is None else float(cfg["lambda_hyb"]),
        normalize=cfg["normalize"],
        eps=eps,
    )
    details = dict(exp.details)
    details.update(
        {
            "positive_probability": _probability_for_label(pos, probabilities, probability_labels, classes),
            "negative_probability": _probability_for_label(exp.negative_label, probabilities, probability_labels, classes),
            "probabilities": None if probabilities is None else probabilities.copy(),
            "probability_labels": None if probability_labels is None else probability_labels.copy(),
        }
    )
    return replace(exp, details=details)


def _score_hyb_fpde_explanations(
    explanations: Sequence[FPDEExplanation],
    *,
    objective: GridObjective,
) -> Dict[str, float]:
    """Compute scalar grid-search metrics from Diff/Cos/Hyb-FPDE-grid explanations."""
    if len(explanations) == 0:
        raise ValueError("explanations must be non-empty")

    evidences = np.asarray([float(exp.evidence) for exp in explanations], dtype=float)
    residuals = np.asarray([float(exp.exactness_residual) for exp in explanations], dtype=float)
    positive = np.maximum(evidences, 0.0)
    agreement = np.asarray(evidences > 0.0, dtype=float)

    if objective == "blackbox_agreement":
        score = float(np.mean(agreement))
    elif objective == "mean_positive_evidence":
        score = float(np.mean(positive))
    elif objective == "mean_margin_weighted_evidence":
        weights = []
        for exp in explanations:
            p_pos = exp.details.get("positive_probability", None)
            p_neg = exp.details.get("negative_probability", None)
            if p_pos is None or p_neg is None:
                weights.append(1.0)
            else:
                weights.append(max(0.0, float(p_pos) - float(p_neg)))
        w = np.asarray(weights, dtype=float)
        score = float(np.mean(positive * w))
    else:
        raise ValueError(
            "objective must be 'blackbox_agreement', 'mean_positive_evidence', "
            "or 'mean_margin_weighted_evidence'"
        )

    lambdas = []
    for exp in explanations:
        value = exp.details.get("lambda_hyb", np.nan)
        try:
            lambdas.append(float(value))
        except (TypeError, ValueError):
            lambdas.append(float("nan"))
    lambda_arr = np.asarray(lambdas, dtype=float)
    finite_lambdas = lambda_arr[np.isfinite(lambda_arr)]

    return {
        "score": score,
        "agreement_rate": float(np.mean(agreement)),
        "mean_evidence": float(np.mean(evidences)),
        "median_evidence": float(np.median(evidences)),
        "mean_positive_evidence": float(np.mean(positive)),
        "mean_abs_evidence": float(np.mean(np.abs(evidences))),
        "mean_exactness_abs_residual": float(np.mean(np.abs(residuals))),
        "max_exactness_abs_residual": float(np.max(np.abs(residuals))),
        "mean_lambda_hyb": float(np.mean(finite_lambdas)) if finite_lambdas.size else float("nan"),
        "min_lambda_hyb": float(np.min(finite_lambdas)) if finite_lambdas.size else float("nan"),
        "max_lambda_hyb": float(np.max(finite_lambdas)) if finite_lambdas.size else float("nan"),
    }


def grid_search_hyb_fpde_grid(
    X_train: np.ndarray | Sequence[Sequence[float]],
    y_train: Sequence[Any],
    X_eval: np.ndarray | Sequence[Sequence[float]],
    *,
    predictor: Optional[Any] = None,
    objective: GridObjective = "blackbox_agreement",
    fpde_mode_grid: Sequence[GridMode] = ("diff", "cos", "hyb_grid"),
    normalize_grid: Sequence[NormalizeMode] = ("l1",),
    lambda_hyb_grid: Sequence[float] = _default_lambda_hyb_grid(0.1),
    anchor_strategy_grid: Sequence[AnchorStrategy] = ("mean",),
    include_explicit_diff_cos: bool = True,
    max_eval_samples: Optional[int] = None,
    eps: float = 1e-12,
    verbose: bool = False,
) -> HybFPDEGridSearchResult:
    """Exhaustively search single-prototype Diff/Cos/Hyb-FPDE-grid settings.

    This is the non-MP Hyb-FPDE-grid condition:

        Diff-FPDE: kept as an explicit baseline
        Cos-FPDE : kept as an explicit baseline
        Hyb-FPDE-grid : evaluated for every lambda_hyb_grid value

    The default lambda grid is 0.0, 0.1, ..., 1.0. One class-mean prototype is
    used for each class.
    """
    X_train_arr = _as_2d_float("X_train", X_train)
    y_train_arr = np.asarray(list(y_train), dtype=object)
    X_eval_arr = _as_2d_float("X_eval", X_eval)
    if X_train_arr.shape[0] != y_train_arr.shape[0]:
        raise ValueError(
            f"number of train samples and labels differ: {X_train_arr.shape[0]} vs {y_train_arr.shape[0]}"
        )
    if X_train_arr.shape[1] != X_eval_arr.shape[1]:
        raise ValueError(
            f"feature dimension mismatch: train has {X_train_arr.shape[1]}, eval has {X_eval_arr.shape[1]}"
        )
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if max_eval_samples is not None:
        m = int(max_eval_samples)
        if m < 1:
            raise ValueError("max_eval_samples must be positive when provided")
        X_eval_arr = X_eval_arr[: min(m, X_eval_arr.shape[0])]
    if X_eval_arr.shape[0] == 0:
        raise ValueError("X_eval must contain at least one sample")

    prototypes, classes = class_mean_prototypes(X_train_arr, y_train_arr)
    prototype_labels = classes

    candidates = _hyb_fpde_grid_candidate_dicts(
        fpde_mode_grid=fpde_mode_grid,
        normalize_grid=normalize_grid,
        lambda_hyb_grid=lambda_hyb_grid,
        anchor_strategy_grid=anchor_strategy_grid,
        include_explicit_diff_cos=include_explicit_diff_cos,
    )
    if not candidates:
        raise ValueError("grid produced no valid candidates")

    rows: List[Dict[str, Any]] = []
    best_row: Optional[Dict[str, Any]] = None
    best_config: Optional[Dict[str, Any]] = None

    for candidate_id, cfg in enumerate(candidates):
        try:
            anchor = (
                np.mean(X_train_arr, axis=0)
                if cfg["anchor_strategy"] == "mean"
                else np.zeros(X_train_arr.shape[1], dtype=float)
            )
            explanations = [
                _explain_single_prototype_candidate(
                    X_eval_arr[i],
                    prototypes=prototypes,
                    prototype_labels=prototype_labels,
                    classes=classes,
                    anchor=anchor,
                    predictor=predictor,
                    cfg=cfg,
                    eps=eps,
                )
                for i in range(X_eval_arr.shape[0])
            ]
            metrics = _score_hyb_fpde_explanations(explanations, objective=objective)
            row: Dict[str, Any] = {
                "candidate_id": int(candidate_id),
                "status": "ok",
                **cfg,
                **metrics,
                "n_eval_samples": int(X_eval_arr.shape[0]),
                "n_fitted_prototypes": int(prototypes.shape[0]),
            }
        except Exception as exc:
            row = {
                "candidate_id": int(candidate_id),
                "status": "error",
                **cfg,
                "score": float("-inf"),
                "error": f"{type(exc).__name__}: {exc}",
                "n_eval_samples": int(X_eval_arr.shape[0]),
            }

        rows.append(row)
        if verbose:
            print(
                f"[hyb-grid] candidate={candidate_id + 1}/{len(candidates)} "
                f"status={row['status']} score={row['score']} cfg={cfg}"
            )

        if row["status"] == "ok":
            if best_row is None:
                best_row = row
                best_config = dict(cfg)
            else:
                current_key = (float(row["score"]), float(row.get("mean_evidence", 0.0)))
                best_key = (float(best_row["score"]), float(best_row.get("mean_evidence", 0.0)))
                if current_key > best_key:
                    best_row = row
                    best_config = dict(cfg)

    if best_row is None or best_config is None:
        first_error = next((r.get("error", "unknown error") for r in rows if r["status"] == "error"), "unknown error")
        raise RuntimeError(f"all Hyb-FPDE grid candidates failed; first error: {first_error}")

    return HybFPDEGridSearchResult(
        best_config=best_config,
        best_score=float(best_row["score"]),
        rows=tuple(rows),
        objective=objective,
        n_candidates=len(candidates),
        n_eval_samples=int(X_eval_arr.shape[0]),
    )


def explain_hyb_fpde_grid(
    x: np.ndarray | Sequence[float],
    X_train: np.ndarray | Sequence[Sequence[float]],
    y_train: Sequence[Any],
    *,
    predictor: Optional[Any] = None,
    positive_label: Optional[Any] = None,
    negative_label: Optional[Any] = None,
    normalize: NormalizeMode = "l1",
    lambda_hyb_grid: Sequence[float] = _default_lambda_hyb_grid(0.1),
    anchor_strategy: AnchorStrategy = "mean",
    eps: float = 1e-12,
) -> Dict[str, FPDEExplanation]:
    """Explain one sample with Diff, Cos, and Hyb-FPDE-grid for all lambdas."""
    X_train_arr = _as_2d_float("X_train", X_train)
    y_train_arr = np.asarray(list(y_train), dtype=object)
    x_arr = _as_1d_float("x", x)
    if X_train_arr.shape[1] != x_arr.shape[0]:
        raise ValueError(f"feature dimension mismatch: train has {X_train_arr.shape[1]}, x has {x_arr.shape[0]}")
    prototypes, labels = class_mean_prototypes(X_train_arr, y_train_arr)
    anchor = np.mean(X_train_arr, axis=0) if anchor_strategy == "mean" else np.zeros(X_train_arr.shape[1], dtype=float)

    if positive_label is None:
        pos, neg, _, _ = _label_contrast_for_grid_sample(
            x_arr,
            prototypes=prototypes,
            prototype_labels=labels,
            classes=labels,
            predictor=predictor,
        )
    else:
        pos, neg = positive_label, negative_label

    out: Dict[str, FPDEExplanation] = {
        "diff": explain_with_selected_prototypes(
            x_arr,
            prototypes,
            labels,
            positive_label=pos,
            negative_label=neg,
            mode="diff",
            eps=eps,
        ),
        "cos": explain_with_selected_prototypes(
            x_arr,
            prototypes,
            labels,
            positive_label=pos,
            negative_label=neg,
            mode="cos",
            anchor=anchor,
            eps=eps,
        ),
    }
    for lam in lambda_hyb_grid:
        lam_f = float(lam)
        if not np.isfinite(lam_f) or lam_f < 0.0 or lam_f > 1.0:
            raise ValueError("lambda_hyb_grid values must be finite and in [0, 1]")
        out[f"hyb_grid_lambda_{lam_f:.12g}"] = _explain_with_selected_prototypes_for_grid(
            x_arr,
            prototypes,
            labels,
            positive_label=pos,
            negative_label=neg,
            mode="hyb_grid",
            anchor=anchor,
            lambda_hyb=lam_f,
            normalize=normalize,
            eps=eps,
        )
    return out



@dataclass(frozen=True)
class HybFPDEValidationSelectionResult:
    """Result of validation-based lambda_hyb selection for Hyb-FPDE-grid.

    The rows field contains one dictionary per lambda candidate.  The selected
    lambda maximizes the held-out validation combined score:

        combined_score = 0.5 * (deletion_drop_auc + insertion_auc)

    This result object is intentionally separate from HybFPDEGridSearchResult,
    because this validation-based selection evaluates perturbation curves rather
    than evidence-only FPDE grid-search objectives.
    """

    best_lambda: float
    best_config: Dict[str, Any]
    rows: Tuple[Dict[str, Any], ...]
    n_eval_samples: int

    def sorted_rows(self) -> List[Dict[str, Any]]:
        """Return lambda candidates sorted from best to worst."""
        return sorted(
            (dict(row) for row in self.rows),
            key=lambda r: (
                float(r.get("score", float("-inf"))),
                float(r.get("mean_insertion_auc", 0.0)),
                -abs(float(r.get("lambda_hyb", 0.5)) - 0.5),
            ),
            reverse=True,
        )


def parse_float_grid(text_or_values: str | Sequence[float]) -> Tuple[float, ...]:
    """Parse a comma-separated grid string or numeric sequence."""
    if isinstance(text_or_values, str):
        values = []
        for part in text_or_values.split(","):
            part = part.strip()
            if part:
                values.append(float(part))
    else:
        values = [float(v) for v in text_or_values]

    if not values:
        raise ValueError("grid must contain at least one value")
    for value in values:
        if not np.isfinite(value):
            raise ValueError(f"non-finite grid value: {value!r}")
    return tuple(values)


def _safe_auc(x: np.ndarray, y: np.ndarray) -> float:
    """Trapezoidal AUC compatible with NumPy 1.x and 2.x."""
    try:
        return float(np.trapezoid(y, x))
    except AttributeError:  # NumPy < 2.0
        return float(np.trapz(y, x))


def _class_to_probability_index(model: Any) -> Dict[Any, int]:
    if not hasattr(model, "classes_"):
        raise ValueError("model must expose classes_ when using predict_proba")
    return {c: i for i, c in enumerate(np.asarray(model.classes_, dtype=object).tolist())}


def predict_proba_for_label(model: Any, X: np.ndarray, label: Any) -> np.ndarray:
    """Return predict_proba(X) column corresponding to label."""
    if not hasattr(model, "predict_proba"):
        raise ValueError("model must implement predict_proba")
    proba = np.asarray(model.predict_proba(X), dtype=float)
    if proba.ndim != 2:
        raise ValueError(f"predict_proba must return a 2D array, got shape={proba.shape}")
    if not np.all(np.isfinite(proba)):
        raise ValueError("predict_proba contains NaN or inf")
    class_to_idx = _class_to_probability_index(model)
    if label not in class_to_idx:
        raise ValueError(f"label {label!r} is not in model.classes_={getattr(model, 'classes_', None)!r}")
    return proba[:, int(class_to_idx[label])]


def top_two_labels(model: Any, x: np.ndarray | Sequence[float]) -> Tuple[Any, Optional[Any], np.ndarray]:
    """Return predicted label, rival label, and probability vector for one sample."""
    x_arr = _as_1d_float("x", x)
    if not hasattr(model, "predict_proba"):
        raise ValueError("model must implement predict_proba")
    proba = np.asarray(model.predict_proba(x_arr.reshape(1, -1))[0], dtype=float)
    if proba.ndim != 1 or proba.size == 0:
        raise ValueError("predict_proba for one sample must return a non-empty vector")
    if not np.all(np.isfinite(proba)):
        raise ValueError("predict_proba contains NaN or inf")
    classes = np.asarray(model.classes_, dtype=object)
    if classes.shape[0] != proba.shape[0]:
        raise ValueError("model.classes_ length differs from predict_proba width")
    order = np.argsort(proba)[::-1]
    positive_label = classes[int(order[0])]
    negative_label = classes[int(order[1])] if order.size >= 2 else None
    return positive_label, negative_label, proba


def hyb_fpde_grid_attribution(
    x: np.ndarray | Sequence[float],
    X_train: np.ndarray | Sequence[Sequence[float]],
    y_train: Sequence[Any],
    model: Any,
    *,
    lambda_hyb: float,
    normalize: NormalizeMode = "l1",
    anchor_strategy: AnchorStrategy = "mean",
    eps: float = 1e-12,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute Hyb-FPDE-grid attribution for one fixed lambda_hyb.

    The positive class is the model's predicted class for x.  The negative class
    is the second-highest probability class.  This function is the basic unit
    used by validation-based lambda selection.
    """
    x_arr = _as_1d_float("x", x)
    X_train_arr = _as_2d_float("X_train", X_train)
    positive_label, negative_label, proba = top_two_labels(model, x_arr)

    explanations = explain_hyb_fpde_grid(
        x_arr,
        X_train_arr,
        y_train,
        predictor=model,
        positive_label=positive_label,
        negative_label=negative_label,
        normalize=normalize,
        lambda_hyb_grid=(float(lambda_hyb),),
        anchor_strategy=anchor_strategy,
        eps=eps,
    )
    key = f"hyb_grid_lambda_{float(lambda_hyb):.12g}"
    if key not in explanations:
        raise KeyError(f"expected explanation key {key!r}; got keys={list(explanations)}")
    exp = explanations[key]
    return np.asarray(exp.attributions, dtype=float), {
        "target_label": positive_label,
        "rival_label": negative_label,
        "target_probability": float(np.max(proba)),
        "lambda_hyb": float(lambda_hyb),
        "evidence": float(exp.evidence),
        "exactness_residual": float(exp.exactness_residual),
        "positive_score": float(exp.positive_score),
        "negative_score": float(exp.negative_score),
    }


def perturbation_curves(
    model: Any,
    x: np.ndarray | Sequence[float],
    attributions: np.ndarray | Sequence[float],
    target_label: Any,
    baseline: np.ndarray | Sequence[float],
    fractions: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
) -> Dict[str, Any]:
    """Compute deletion and insertion curves for one explanation vector.

    Important features are ranked by signed positive attribution, descending.
    Deletion replaces selected features with the baseline; insertion starts from
    the baseline and restores selected features.
    """
    x_arr = _as_1d_float("x", x)
    baseline_arr = _as_1d_float("baseline", baseline)
    attr = _as_1d_float("attributions", attributions)
    if x_arr.shape != baseline_arr.shape or x_arr.shape != attr.shape:
        raise ValueError("x, baseline, and attributions must have the same shape")

    order = np.argsort(-attr)  # signed positive support first
    d = x_arr.shape[0]

    frac_arr = np.asarray(sorted(set(float(f) for f in fractions)), dtype=float)
    if frac_arr.size == 0:
        raise ValueError("fractions must contain at least one value")
    if not np.all(np.isfinite(frac_arr)):
        raise ValueError("fractions contains NaN or inf")
    if np.any(frac_arr < 0.0) or np.any(frac_arr > 1.0):
        raise ValueError("fractions must be in [0, 1]")
    if frac_arr[0] > 0.0:
        frac_arr = np.concatenate([[0.0], frac_arr])
    if frac_arr[-1] < 1.0:
        frac_arr = np.concatenate([frac_arr, [1.0]])
    frac_arr = np.clip(frac_arr, 0.0, 1.0)

    deletion_X = []
    insertion_X = []
    for frac in frac_arr:
        k = int(round(float(frac) * d))
        idx = order[:k]

        xd = x_arr.copy()
        xd[idx] = baseline_arr[idx]
        deletion_X.append(xd)

        xi = baseline_arr.copy()
        xi[idx] = x_arr[idx]
        insertion_X.append(xi)

    deletion_X_arr = np.vstack(deletion_X)
    insertion_X_arr = np.vstack(insertion_X)
    deletion_prob = predict_proba_for_label(model, deletion_X_arr, target_label)
    insertion_prob = predict_proba_for_label(model, insertion_X_arr, target_label)

    deletion_auc = _safe_auc(frac_arr, deletion_prob)
    insertion_auc = _safe_auc(frac_arr, insertion_prob)
    p0 = float(deletion_prob[0])
    deletion_drop_auc = p0 - deletion_auc
    combined_score = 0.5 * (deletion_drop_auc + insertion_auc)

    return {
        "fractions": frac_arr.tolist(),
        "deletion_prob": deletion_prob.tolist(),
        "insertion_prob": insertion_prob.tolist(),
        "p0": p0,
        "deletion_auc": deletion_auc,
        "deletion_drop_auc": deletion_drop_auc,
        "insertion_auc": insertion_auc,
        "combined_score": combined_score,
    }


def select_lambda_by_deletion_insertion_validation(
    X_train: np.ndarray | Sequence[Sequence[float]],
    y_train: Sequence[Any],
    X_val: np.ndarray | Sequence[Sequence[float]],
    model: Any,
    *,
    lambda_hyb_grid: Sequence[float] = _default_lambda_hyb_grid(0.1),
    fractions: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
    baseline: Optional[np.ndarray | Sequence[float]] = None,
    normalize: NormalizeMode = "l1",
    anchor_strategy: AnchorStrategy = "mean",
    eps: float = 1e-12,
) -> HybFPDEValidationSelectionResult:
    """Select lambda_hyb by held-out deletion/insertion validation score.

    This is the default recommended lambda-selection API for Hyb-FPDE-grid.
    It aligns lambda selection with the final perturbation-based explanation
    evaluation criterion.

    Parameters
    ----------
    X_train, y_train:
        Data used to construct class-mean prototypes.
    X_val:
        Held-out validation samples used only for lambda selection.
    model:
        Black-box classifier with predict_proba and classes_.
    lambda_hyb_grid:
        Candidate lambda values. Default is 0.0, 0.1, ..., 1.0.
    fractions:
        Perturbation fractions for deletion/insertion AUC.
    baseline:
        Replacement vector. If None, mean(X_train, axis=0) is used.

    Returns
    -------
    HybFPDEValidationSelectionResult
        Contains best_lambda, best_config, and one row per lambda candidate.
    """
    X_train_arr = _as_2d_float("X_train", X_train)
    X_val_arr = _as_2d_float("X_val", X_val)
    if X_train_arr.shape[1] != X_val_arr.shape[1]:
        raise ValueError(
            f"feature dimension mismatch: train has {X_train_arr.shape[1]}, val has {X_val_arr.shape[1]}"
        )
    if X_val_arr.shape[0] == 0:
        raise ValueError("X_val must contain at least one sample")

    lambdas = parse_float_grid(lambda_hyb_grid)
    for lam in lambdas:
        if lam < 0.0 or lam > 1.0:
            raise ValueError("lambda_hyb_grid values must be in [0, 1]")

    baseline_arr = np.mean(X_train_arr, axis=0) if baseline is None else _as_1d_float("baseline", baseline)
    if baseline_arr.shape[0] != X_train_arr.shape[1]:
        raise ValueError("baseline dimension differs from X_train feature dimension")

    rows: List[Dict[str, Any]] = []
    for candidate_id, lam in enumerate(lambdas):
        combined_scores: List[float] = []
        deletion_drop_scores: List[float] = []
        insertion_scores: List[float] = []
        deletion_auc_scores: List[float] = []
        p0_scores: List[float] = []
        evidence_scores: List[float] = []
        residual_scores: List[float] = []
        n_error = 0
        first_error: Optional[str] = None

        for x in X_val_arr:
            try:
                attr, detail = hyb_fpde_grid_attribution(
                    x,
                    X_train_arr,
                    y_train,
                    model,
                    lambda_hyb=float(lam),
                    normalize=normalize,
                    anchor_strategy=anchor_strategy,
                    eps=eps,
                )
                target_label = detail["target_label"]
                curves = perturbation_curves(
                    model,
                    x,
                    attr,
                    target_label,
                    baseline_arr,
                    fractions=fractions,
                )
                combined_scores.append(float(curves["combined_score"]))
                deletion_drop_scores.append(float(curves["deletion_drop_auc"]))
                insertion_scores.append(float(curves["insertion_auc"]))
                deletion_auc_scores.append(float(curves["deletion_auc"]))
                p0_scores.append(float(curves["p0"]))
                evidence_scores.append(float(detail["evidence"]))
                residual_scores.append(float(detail["exactness_residual"]))
            except Exception as exc:  # keep grid search robust
                n_error += 1
                if first_error is None:
                    first_error = f"{type(exc).__name__}: {exc}"

        if combined_scores:
            residual_arr = np.asarray(residual_scores, dtype=float)
            row: Dict[str, Any] = {
                "candidate_id": int(candidate_id),
                "status": "ok",
                "fpde_mode": "hyb_grid",
                "method_variant": "hyb_fpde_grid_validation",
                "normalize": normalize,
                "lambda_hyb": float(lam),
                "lambda_grid_mode": "grid_candidate_lambda_hyb",
                "anchor_strategy": anchor_strategy,
                "selection_objective": "deletion_insertion_validation",
                "validation_metric_source": "heldout_deletion_insertion",
                "score": float(np.mean(combined_scores)),
                "mean_combined_score": float(np.mean(combined_scores)),
                "mean_deletion_drop_auc": float(np.mean(deletion_drop_scores)),
                "mean_insertion_auc": float(np.mean(insertion_scores)),
                "mean_deletion_auc": float(np.mean(deletion_auc_scores)),
                "mean_p0": float(np.mean(p0_scores)),
                "mean_evidence": float(np.mean(evidence_scores)),
                "mean_exactness_abs_residual": float(np.nanmean(np.abs(residual_arr))),
                "n_eval_samples": int(X_val_arr.shape[0]),
                "n_success": int(len(combined_scores)),
                "n_error": int(n_error),
            }
        else:
            row = {
                "candidate_id": int(candidate_id),
                "status": "error",
                "fpde_mode": "hyb_grid",
                "method_variant": "hyb_fpde_grid_validation",
                "normalize": normalize,
                "lambda_hyb": float(lam),
                "lambda_grid_mode": "grid_candidate_lambda_hyb",
                "anchor_strategy": anchor_strategy,
                "selection_objective": "deletion_insertion_validation",
                "validation_metric_source": "heldout_deletion_insertion",
                "score": float("-inf"),
                "error": first_error or "all validation samples failed",
                "n_eval_samples": int(X_val_arr.shape[0]),
                "n_success": 0,
                "n_error": int(n_error),
            }
        rows.append(row)

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if not ok_rows:
        first_error = next((r.get("error", "unknown error") for r in rows), "unknown error")
        raise RuntimeError(f"all lambda candidates failed; first error: {first_error}")

    best_row = max(
        ok_rows,
        key=lambda r: (
            float(r["score"]),
            float(r.get("mean_insertion_auc", 0.0)),
            -abs(float(r["lambda_hyb"]) - 0.5),
        ),
    )
    best_lambda = float(best_row["lambda_hyb"])
    best_config = {
        "fpde_mode": "hyb_grid",
        "method_variant": "hyb_fpde_grid_validation",
        "normalize": normalize,
        "lambda_hyb": best_lambda,
        "lambda_grid_mode": "grid_candidate_lambda_hyb",
        "anchor_strategy": anchor_strategy,
        "selection_objective": "deletion_insertion_validation",
        "validation_metric_source": "heldout_deletion_insertion",
        "validation_score": float(best_row["score"]),
        "validation_mean_deletion_drop_auc": float(best_row.get("mean_deletion_drop_auc", float("nan"))),
        "validation_mean_insertion_auc": float(best_row.get("mean_insertion_auc", float("nan"))),
        "n_grid_eval_samples": int(X_val_arr.shape[0]),
    }
    return HybFPDEValidationSelectionResult(
        best_lambda=best_lambda,
        best_config=best_config,
        rows=tuple(rows),
        n_eval_samples=int(X_val_arr.shape[0]),
    )


# Short default alias for the recommended lambda-selection API.
select_hyb_fpde_grid_lambda = select_lambda_by_deletion_insertion_validation


def explain_with_validation_selected_lambda(
    x: np.ndarray | Sequence[float],
    X_train: np.ndarray | Sequence[Sequence[float]],
    y_train: Sequence[Any],
    model: Any,
    selection: HybFPDEValidationSelectionResult,
    *,
    normalize: NormalizeMode = "l1",
    anchor_strategy: AnchorStrategy = "mean",
    eps: float = 1e-12,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Explain one sample using the validation-selected best_lambda."""
    return hyb_fpde_grid_attribution(
        x,
        X_train,
        y_train,
        model,
        lambda_hyb=selection.best_lambda,
        normalize=normalize,
        anchor_strategy=anchor_strategy,
        eps=eps,
    )

def _assert_close(name: str, a: float, b: float, tol: float = 1e-10) -> None:
    if not np.isfinite(a) or not np.isfinite(b) or abs(a - b) > tol:
        raise AssertionError(f"{name}: {a} != {b} within tol={tol}")


def _self_test() -> None:
    X = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.2, 0.1, 1.1],
            [2.0, 1.8, 0.2],
            [2.2, 2.0, 0.0],
        ],
        dtype=float,
    )
    y = np.array([0, 0, 1, 1], dtype=int)

    prototypes, labels = class_mean_prototypes(X, y)
    anchor = np.mean(X, axis=0)
    x = np.array([0.3, 0.2, 0.9], dtype=float)

    diff_exp = explain_with_selected_prototypes(x, prototypes, labels, positive_label=0, mode="diff")
    cos_exp = explain_with_selected_prototypes(x, prototypes, labels, positive_label=0, mode="cos", anchor=anchor)
    hyb_grid_exp = _explain_with_selected_prototypes_for_grid(
        x,
        prototypes,
        labels,
        positive_label=0,
        mode="hyb_grid",
        anchor=anchor,
        lambda_hyb=0.5,
        normalize="l1",
    )

    _assert_close("diff additivity", diff_exp.evidence, float(np.sum(diff_exp.attributions)))
    _assert_close("cos additivity", cos_exp.evidence, float(np.sum(cos_exp.attributions)))
    _assert_close("hyb-grid additivity", hyb_grid_exp.evidence, float(np.sum(hyb_grid_exp.attributions)))

    hyb_grid_raw_diff = _hyb_fpde_grid_candidate(
        x,
        prototypes[0],
        prototypes[1],
        anchor=anchor,
        lambda_hyb=1.0,
        normalize="none",
        positive_label=0,
        negative_label=1,
        positive_prototype_index=0,
        negative_prototype_index=1,
    )
    _assert_close("hyb raw lambda=1 equals diff", hyb_grid_raw_diff.evidence, diff_exp.evidence)

    hyb_grid_raw_cos = _hyb_fpde_grid_candidate(
        x,
        prototypes[0],
        prototypes[1],
        anchor=anchor,
        lambda_hyb=0.0,
        normalize="none",
        positive_label=0,
        negative_label=1,
        positive_prototype_index=0,
        negative_prototype_index=1,
    )
    _assert_close("hyb raw lambda=0 equals cos", hyb_grid_raw_cos.evidence, cos_exp.evidence)

    grid = grid_search_hyb_fpde_grid(X, y, X, objective="mean_positive_evidence")
    if grid.n_candidates != 13:
        raise AssertionError(f"expected 13 candidates, got {grid.n_candidates}")

    explanations = explain_hyb_fpde_grid(x, X, y, positive_label=0)
    if "diff" not in explanations or "cos" not in explanations or "hyb_grid_lambda_0.5" not in explanations:
        raise AssertionError("lambda-grid explanation output is incomplete")


    class ToyModel:
        classes_ = np.array([0, 1], dtype=object)

        def predict_proba(self, X_in: np.ndarray) -> np.ndarray:
            X_in = np.asarray(X_in, dtype=float)
            z = X_in[:, 0] - 0.5 * X_in[:, 1] + 0.2 * X_in[:, 2]
            p1 = 1.0 / (1.0 + np.exp(-z))
            return np.c_[1.0 - p1, p1]

    validation_selection = select_lambda_by_deletion_insertion_validation(
        X,
        y,
        X,
        ToyModel(),
        lambda_hyb_grid=(0.0, 0.5, 1.0),
        fractions=(0.0, 0.5, 1.0),
    )
    if validation_selection.best_lambda not in (0.0, 0.5, 1.0):
        raise AssertionError("validation lambda selection returned an unexpected lambda")
    attr, detail = explain_with_validation_selected_lambda(x, X, y, ToyModel(), validation_selection)
    if attr.shape[0] != X.shape[1] or "lambda_hyb" not in detail:
        raise AssertionError("validation-selected explanation output is incomplete")


__all__ = [
    "FPDEExplanation",
    "HybFPDEGridSearchResult",
    "Mode",
    "GridMode",
    "NormalizeMode",
    "AnchorStrategy",
    "GridObjective",
    "regularized_cosine",
    "diff_fpde",
    "cos_fpde",
    "select_prototype_pair",
    "explain_with_selected_prototypes",
    "class_mean_prototypes",
    "grid_search_hyb_fpde_grid",
    "explain_hyb_fpde_grid",
    "HybFPDEValidationSelectionResult",
    "parse_float_grid",
    "predict_proba_for_label",
    "top_two_labels",
    "hyb_fpde_grid_attribution",
    "perturbation_curves",
    "select_lambda_by_deletion_insertion_validation",
    "select_hyb_fpde_grid_lambda",
    "explain_with_validation_selected_lambda",
]


if __name__ == "__main__":
    _self_test()
    print("FPDE.py self-test passed")
