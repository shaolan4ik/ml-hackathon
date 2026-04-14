from __future__ import annotations

from pathlib import Path

import pandas as pd
from click.testing import CliRunner

from hackaton.train.cli import cli


def _write_train_csvs(base: Path) -> tuple[Path, Path, Path]:
    user_path = base / "user.csv"
    shift_path = base / "shift.csv"
    event_path = base / "event.csv"

    users = pd.DataFrame(
        [
            {"location_id": "loc-1", "is_strict_location": True, "id": "u1", "has_mk": True},
            {"location_id": "loc-2", "is_strict_location": False, "id": "u2", "has_mk": False},
            {"location_id": "loc-1", "is_strict_location": False, "id": "u3", "has_mk": True},
        ]
    )
    shifts = pd.DataFrame(
        [
            {
                "id": "s1",
                "start_at": "2026-01-10T08:00:00Z",
                "location_id": "loc-1",
                "task_type": "picker",
                "employer_id": "e1",
                "workplace_id": "w1",
                "need_mk": True,
                "id_differential": False,
                "hours": 8,
                "reward": 1200.0,
                "capacity": 2,
            },
            {
                "id": "s2",
                "start_at": "2026-01-20T08:00:00Z",
                "location_id": "loc-2",
                "task_type": "loader",
                "employer_id": "e2",
                "workplace_id": "w2",
                "need_mk": False,
                "id_differential": False,
                "hours": 6,
                "reward": 900.0,
                "capacity": 1,
            },
            {
                "id": "s3",
                "start_at": "2026-02-05T08:00:00Z",
                "location_id": "loc-1",
                "task_type": "picker",
                "employer_id": "e1",
                "workplace_id": "w1",
                "need_mk": True,
                "id_differential": True,
                "hours": 10,
                "reward": 1500.0,
                "capacity": 3,
            },
        ]
    )
    events = pd.DataFrame(
        [
            {"id": "ev1", "shift_id": "s1", "user_id": "u1", "interaction": "VIEW", "ts": "2026-01-08T10:00:00Z"},
            {"id": "ev2", "shift_id": "s1", "user_id": "u1", "interaction": "APPLY", "ts": "2026-01-09T10:00:00Z"},
            {"id": "ev3", "shift_id": "s2", "user_id": "u2", "interaction": "VIEW", "ts": "2026-01-19T10:00:00Z"},
            {"id": "ev4", "shift_id": "s2", "user_id": "u3", "interaction": "VIEW", "ts": "2026-01-18T10:00:00Z"},
            {"id": "ev5", "shift_id": "s3", "user_id": "u1", "interaction": "VIEW", "ts": "2026-02-03T10:00:00Z"},
            {"id": "ev6", "shift_id": "s3", "user_id": "u1", "interaction": "FINISHED", "ts": "2026-02-04T10:00:00Z"},
            {"id": "ev7", "shift_id": "s3", "user_id": "u2", "interaction": "VIEW", "ts": "2026-02-01T10:00:00Z"},
        ]
    )

    users.to_csv(user_path, index=False)
    shifts.to_csv(shift_path, index=False)
    events.to_csv(event_path, index=False)
    return user_path, shift_path, event_path


def test_train_cli_smoke(tmp_path: Path) -> None:
    user_path, shift_path, event_path = _write_train_csvs(tmp_path)
    out_dir = tmp_path / "artifacts"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "train",
            "--user-path",
            str(user_path),
            "--shift-path",
            str(shift_path),
            "--event-path",
            str(event_path),
            "--output-dir",
            str(out_dir),
            "--skip-shap",
        ],
    )

    assert result.exit_code == 0, result.output
    for path in [
        out_dir / "model.pkl",
        out_dir / "metrics.json",
        out_dir / "feature_schema.json",
        out_dir / "train_config.json",
        out_dir / "data_contract_check.json",
        out_dir / "train_report.md",
    ]:
        assert path.exists(), f"Missing artifact: {path}"

