from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import roc_auc_score

METRIC_POOL_SIZE = 10


@dataclass(frozen=True, slots=True)
class MetricResult:
    target_metric: float
    evaluated_days: int
    evaluated_groups: int
    evaluated_shifts: int
    day_metrics: dict[str, float]
    group_metrics: list[dict[str, object]]


def _shift_metric_for_top_capacity(group: pd.DataFrame) -> float | None:
    if group.empty:
        return None
    capacity = int(max(1, group["capacity"].iloc[0]))
    # Metric pool is fixed to TOP-10 candidates by regulation.
    pool = group.sort_values("score", ascending=False).head(METRIC_POOL_SIZE)
    top = pool.head(capacity)
    if top["target"].nunique() < 2:
        return None
    # FPR cap depends on shift capacity relative to fixed metric pool size.
    max_fpr = min(1.0, capacity / METRIC_POOL_SIZE)
    return float(roc_auc_score(top["target"], top["score"], max_fpr=max_fpr))


def calculate_target_metric(frame: pd.DataFrame) -> MetricResult:
    """
    Target metric by regulation:
    - Evaluate per shift on TOP-K candidates where K = shift capacity
    - Candidate pool for metric is fixed to TOP-10
    - Use ROC-AUC with FPR cap = min(1.0, capacity / 10)
    - Aggregate by day and capacity groups, then average across days
    """
    required = {"shift_id", "start_at", "capacity", "target", "score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Metric frame missing columns: {sorted(missing)}")

    data = frame.copy()
    data["start_at"] = pd.to_datetime(data["start_at"], utc=True, errors="coerce")
    data = data.dropna(subset=["start_at", "capacity", "target", "score"]).copy()
    if data.empty:
        return MetricResult(
            target_metric=0.0,
            evaluated_days=0,
            evaluated_groups=0,
            evaluated_shifts=0,
            day_metrics={},
            group_metrics=[],
        )

    data["eval_date"] = data["start_at"].dt.date.astype(str)
    data["capacity"] = pd.to_numeric(data["capacity"], errors="coerce").fillna(1).astype(int).clip(lower=1)
    data["target"] = pd.to_numeric(data["target"], errors="coerce").fillna(0).astype(int)
    data["score"] = pd.to_numeric(data["score"], errors="coerce")
    data = data.dropna(subset=["score"])

    shift_metrics_rows: list[dict[str, object]] = []
    for (eval_date, capacity, shift_id), shift_df in data.groupby(["eval_date", "capacity", "shift_id"]):
        metric = _shift_metric_for_top_capacity(shift_df)
        if metric is None:
            continue
        shift_metrics_rows.append(
            {
                "eval_date": eval_date,
                "capacity": int(capacity),
                "shift_id": str(shift_id),
                "shift_metric": float(metric),
            }
        )

    if not shift_metrics_rows:
        return MetricResult(
            target_metric=0.0,
            evaluated_days=0,
            evaluated_groups=0,
            evaluated_shifts=0,
            day_metrics={},
            group_metrics=[],
        )

    shift_metrics = pd.DataFrame(shift_metrics_rows)
    group_metrics = (
        shift_metrics.groupby(["eval_date", "capacity"], as_index=False)["shift_metric"]
        .mean()
        .rename(columns={"shift_metric": "group_metric"})
    )
    day_metrics_series = group_metrics.groupby("eval_date")["group_metric"].mean()

    return MetricResult(
        target_metric=float(day_metrics_series.mean()),
        evaluated_days=int(day_metrics_series.shape[0]),
        evaluated_groups=int(group_metrics.shape[0]),
        evaluated_shifts=int(shift_metrics["shift_id"].nunique()),
        day_metrics={str(k): float(v) for k, v in day_metrics_series.to_dict().items()},
        group_metrics=[
            {
                "eval_date": str(row["eval_date"]),
                "capacity": int(row["capacity"]),
                "group_metric": float(row["group_metric"]),
            }
            for _, row in group_metrics.iterrows()
        ],
    )

