from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from hackaton.train.training import TrainConfig, run_training


@click.group()
def cli() -> None:
    """CLI для baseline обучения."""


@cli.command("train")
@click.option("--user-path", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True)
@click.option("--shift-path", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True)
@click.option("--event-path", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False), required=True)
@click.option("--random-state", type=int, default=42, show_default=True)
@click.option("--max-iter", type=int, default=1000, show_default=True)
@click.option("--test-ratio", type=float, default=0.2, show_default=True)
@click.option("--skip-shap", is_flag=True, default=False)
@click.option("--shap-sample-size", type=int, default=1000, show_default=True)
def train_cmd(
    user_path: Path,
    shift_path: Path,
    event_path: Path,
    output_dir: Path,
    random_state: int,
    max_iter: int,
    test_ratio: float,
    skip_shap: bool,
    shap_sample_size: int,
) -> None:
    """Запуск baseline train pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    cfg = TrainConfig(
        user_path=str(user_path),
        shift_path=str(shift_path),
        event_path=str(event_path),
        output_dir=str(output_dir),
        random_state=random_state,
        max_iter=max_iter,
        test_ratio=test_ratio,
        skip_shap=skip_shap,
        shap_sample_size=shap_sample_size,
    )
    result = run_training(cfg)
    click.echo("Training finished successfully.")
    click.echo(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()

