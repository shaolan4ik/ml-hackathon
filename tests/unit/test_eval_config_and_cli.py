from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from hackaton.eval.cli import cli
from hackaton.eval.evaluator import EvalConfig, run_evaluation


def _dummy_cfg(tmp_path: Path, **overrides: object) -> EvalConfig:
    base = {
        "host": "127.0.0.1",
        "port": 8080,
        "user_path": str(tmp_path / "user.csv"),
        "shift_path": str(tmp_path / "shift.csv"),
        "event_path": str(tmp_path / "event.csv"),
        "val_apply_path": str(tmp_path / "apply.csv"),
        "val_shift_path": str(tmp_path / "val_shift.csv"),
        "val_event_path": str(tmp_path / "val_event.csv"),
        "output_dir": str(tmp_path / "out"),
    }
    base.update(overrides)
    return EvalConfig(**base)


def test_eval_config_rejects_predict_rpm_above_200(tmp_path: Path) -> None:
    cfg = _dummy_cfg(tmp_path, predict_max_rpm=201)
    try:
        run_evaluation(cfg)
    except ValueError as exc:
        assert "predict_max_rpm must be <= 200" in str(exc)
    else:
        raise AssertionError("Expected ValueError for predict_max_rpm > 200")


def test_eval_config_rejects_non_positive_concurrency(tmp_path: Path) -> None:
    cfg = _dummy_cfg(tmp_path, predict_max_concurrency=0)
    try:
        run_evaluation(cfg)
    except ValueError as exc:
        assert "predict_max_concurrency must be >= 1" in str(exc)
    else:
        raise AssertionError("Expected ValueError for predict_max_concurrency < 1")


def test_eval_cli_prints_human_readable_summary(monkeypatch, tmp_path: Path) -> None:
    user_path = tmp_path / "user.csv"
    shift_path = tmp_path / "shift.csv"
    event_path = tmp_path / "event.csv"
    apply_path = tmp_path / "apply.csv"
    val_shift_path = tmp_path / "val_shift.csv"
    val_event_path = tmp_path / "val_event.csv"
    out_dir = tmp_path / "artifacts"
    for path in [user_path, shift_path, event_path, apply_path, val_shift_path, val_event_path]:
        path.write_text("id\n", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_run_evaluation(cfg: EvalConfig) -> dict[str, object]:
        captured["cfg"] = cfg
        return {
            "overall_target_metric": 0.321,
            "days_evaluated": 3,
            "performance": {"predict_rpm": 150.5},
        }

    monkeypatch.setattr("hackaton.eval.cli.run_evaluation", fake_run_evaluation)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "--host",
            "127.0.0.1",
            "--port",
            "8080",
            "--user-path",
            str(user_path),
            "--shift-path",
            str(shift_path),
            "--event-path",
            str(event_path),
            "--val-apply-path",
            str(apply_path),
            "--val-shift-path",
            str(val_shift_path),
            "--val-event-path",
            str(val_event_path),
            "--output-dir",
            str(out_dir),
            "--predict-max-concurrency",
            "4",
            "--predict-max-rpm",
            "200",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Evaluation finished successfully." in result.output
    assert "Report:" in result.output
    assert "Overall target metric:" in result.output
    assert '"overall_target_metric"' not in result.output

    cfg = captured["cfg"]
    assert isinstance(cfg, EvalConfig)
    assert cfg.predict_max_concurrency == 4
    assert cfg.predict_max_rpm == 200
