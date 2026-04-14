from __future__ import annotations

import logging
from pathlib import Path

import click

from hackaton.eval.evaluator import EvalConfig, run_evaluation


@click.group()
def cli() -> None:
    """CLI для оценки качества модели."""


@cli.command("run")
@click.option("--host", type=str, default="127.0.0.1", show_default=True)
@click.option("--port", type=int, default=8000, show_default=True)
@click.option("--user-path", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True)
@click.option("--shift-path", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True)
@click.option("--event-path", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True)
@click.option("--val-apply-path", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True)
@click.option("--val-shift-path", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True)
@click.option("--val-event-path", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False), required=True)
@click.option("--limit", type=int, default=10, show_default=True)
@click.option("--batch-size", type=int, default=1000, show_default=True)
@click.option("--prepare-initial-timeout-sec", type=int, default=1200, show_default=True)
@click.option("--prepare-day-timeout-sec", type=int, default=300, show_default=True)
@click.option("--poll-interval-sec", type=int, default=5, show_default=True)
@click.option("--rpc-timeout-ms", type=int, default=10000, show_default=True)
@click.option("--predict-retry-on-not-ready", type=int, default=5, show_default=True)
@click.option("--predict-max-concurrency", type=int, default=1, show_default=True)
@click.option("--predict-max-rpm", type=int, default=200, show_default=True)
def run_cmd(
    host: str,
    port: int,
    user_path: Path,
    shift_path: Path,
    event_path: Path,
    val_apply_path: Path,
    val_shift_path: Path,
    val_event_path: Path,
    output_dir: Path,
    limit: int,
    batch_size: int,
    prepare_initial_timeout_sec: int,
    prepare_day_timeout_sec: int,
    poll_interval_sec: int,
    rpc_timeout_ms: int,
    predict_retry_on_not_ready: int,
    predict_max_concurrency: int,
    predict_max_rpm: int,
) -> None:
    """Запуск eval-пайплайна по дневному регламенту."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    cfg = EvalConfig(
        host=host,
        port=port,
        user_path=str(user_path),
        shift_path=str(shift_path),
        event_path=str(event_path),
        val_apply_path=str(val_apply_path),
        val_shift_path=str(val_shift_path),
        val_event_path=str(val_event_path),
        output_dir=str(output_dir),
        limit=limit,
        batch_size=batch_size,
        prepare_initial_timeout_sec=prepare_initial_timeout_sec,
        prepare_day_timeout_sec=prepare_day_timeout_sec,
        poll_interval_sec=poll_interval_sec,
        rpc_timeout_ms=rpc_timeout_ms,
        predict_retry_on_not_ready=predict_retry_on_not_ready,
        predict_max_concurrency=predict_max_concurrency,
        predict_max_rpm=predict_max_rpm,
    )
    summary = run_evaluation(cfg)
    report_path = output_dir / "eval_report.md"
    perf = summary.get("performance", {})
    click.echo("Evaluation finished successfully.")
    click.echo(f"Report: {report_path}")
    click.echo(f"Overall target metric: {summary.get('overall_target_metric', 0.0):.6f}")
    click.echo(f"Days evaluated: {summary.get('days_evaluated', 0)}")
    click.echo(f"predict_rpm: {float(perf.get('predict_rpm', 0.0)):.3f}")


if __name__ == "__main__":
    cli()

