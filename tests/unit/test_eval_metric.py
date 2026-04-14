from __future__ import annotations

import pandas as pd

from hackaton.eval.metric import calculate_target_metric


def test_calculate_target_metric_smoke() -> None:
    df = pd.DataFrame(
        [
            {"shift_id": "s1", "start_at": "2026-03-01T08:00:00Z", "capacity": 2, "target": 1, "score": 0.9},
            {"shift_id": "s1", "start_at": "2026-03-01T08:00:00Z", "capacity": 2, "target": 0, "score": 0.8},
            {"shift_id": "s1", "start_at": "2026-03-01T08:00:00Z", "capacity": 2, "target": 0, "score": 0.2},
            {"shift_id": "s2", "start_at": "2026-03-01T09:00:00Z", "capacity": 1, "target": 1, "score": 0.6},
            {"shift_id": "s2", "start_at": "2026-03-01T09:00:00Z", "capacity": 1, "target": 0, "score": 0.1},
            {"shift_id": "s3", "start_at": "2026-03-02T10:00:00Z", "capacity": 2, "target": 1, "score": 0.7},
            {"shift_id": "s3", "start_at": "2026-03-02T10:00:00Z", "capacity": 2, "target": 0, "score": 0.6},
            {"shift_id": "s3", "start_at": "2026-03-02T10:00:00Z", "capacity": 2, "target": 0, "score": 0.2},
        ]
    )
    result = calculate_target_metric(df)
    assert 0.0 <= result.target_metric <= 1.0
    assert result.evaluated_days == 2
    assert result.evaluated_groups >= 1
    assert result.evaluated_shifts >= 1
    assert len(result.day_metrics) == 2


def test_calculate_target_metric_empty_returns_zero() -> None:
    df = pd.DataFrame(columns=["shift_id", "start_at", "capacity", "target", "score"])
    result = calculate_target_metric(df)
    assert result.target_metric == 0.0
    assert result.evaluated_days == 0
    assert result.day_metrics == {}
    assert result.group_metrics == []

