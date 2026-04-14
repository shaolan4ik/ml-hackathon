from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hackaton.eval.metric import calculate_target_metric

LOGGER = logging.getLogger(__name__)

REQUIRED_USER_COLUMNS = ["location_id", "is_strict_location", "id", "has_mk"]
REQUIRED_SHIFT_COLUMNS = [
    "id",
    "start_at",
    "location_id",
    "task_type",
    "employer_id",
    "workplace_id",
    "need_mk",
    "id_differential",
    "hours",
    "reward",
    "capacity",
]
REQUIRED_EVENT_COLUMNS = ["id", "shift_id", "user_id", "interaction", "ts"]
VALID_INTERACTIONS = {"VIEW", "APPLY", "FINISHED", "USER_CANCEL", "SYSTEM_CANCEL"}


@dataclass(frozen=True, slots=True)
class TrainConfig:
    user_path: str
    shift_path: str
    event_path: str
    output_dir: str
    random_state: int = 42
    max_iter: int = 1000
    test_ratio: float = 0.2
    skip_shap: bool = False
    shap_sample_size: int = 1000


def _to_bool(series: pd.Series) -> pd.Series:
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.map(mapping)


def _validate_columns(df: pd.DataFrame, required: list[str], name: str) -> dict[str, object]:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")
    null_counts = {c: int(df[c].isna().sum()) for c in required}
    return {
        "rows": int(len(df)),
        "required_columns_ok": True,
        "null_counts": null_counts,
    }


def _load_and_validate_data(
    cfg: TrainConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    users = pd.read_csv(cfg.user_path)
    shifts = pd.read_csv(cfg.shift_path)
    events = pd.read_csv(cfg.event_path)

    checks = {
        "user": _validate_columns(users, REQUIRED_USER_COLUMNS, "user.csv"),
        "shift": _validate_columns(shifts, REQUIRED_SHIFT_COLUMNS, "shift.csv"),
        "event": _validate_columns(events, REQUIRED_EVENT_COLUMNS, "event.csv"),
    }

    users = users[REQUIRED_USER_COLUMNS].copy()
    shifts = shifts[REQUIRED_SHIFT_COLUMNS].copy()
    events = events[REQUIRED_EVENT_COLUMNS].copy()

    users["id"] = users["id"].astype(str)
    users["location_id"] = users["location_id"].astype(str)
    users["has_mk"] = _to_bool(users["has_mk"])
    users["is_strict_location"] = _to_bool(users["is_strict_location"])

    shifts["id"] = shifts["id"].astype(str)
    shifts["location_id"] = shifts["location_id"].astype(str)
    shifts["task_type"] = shifts["task_type"].astype(str)
    shifts["employer_id"] = shifts["employer_id"].astype(str)
    shifts["workplace_id"] = shifts["workplace_id"].astype(str)
    shifts["need_mk"] = _to_bool(shifts["need_mk"])
    shifts["id_differential"] = _to_bool(shifts["id_differential"])
    shifts["hours"] = pd.to_numeric(shifts["hours"], errors="coerce")
    shifts["reward"] = pd.to_numeric(shifts["reward"], errors="coerce")
    shifts["capacity"] = pd.to_numeric(shifts["capacity"], errors="coerce")
    shifts["start_at"] = pd.to_datetime(shifts["start_at"], utc=True, errors="coerce")

    events["id"] = events["id"].astype(str)
    events["shift_id"] = events["shift_id"].astype(str)
    events["user_id"] = events["user_id"].astype(str)
    events["interaction"] = events["interaction"].astype(str).str.upper()
    events["ts"] = pd.to_datetime(events["ts"], utc=True, errors="coerce")
    events = events[events["interaction"].isin(VALID_INTERACTIONS)]

    critical = {
        "users": ["id", "location_id", "has_mk", "is_strict_location"],
        "shifts": [
            "id",
            "start_at",
            "location_id",
            "need_mk",
            "id_differential",
            "hours",
            "reward",
            "capacity",
        ],
        "events": ["id", "shift_id", "user_id", "interaction", "ts"],
    }
    users = users.dropna(subset=critical["users"]).drop_duplicates(subset=["id"])
    shifts = shifts.dropna(subset=critical["shifts"]).drop_duplicates(subset=["id"])
    events = events.dropna(subset=critical["events"]).drop_duplicates(subset=["id"])

    checks["post_clean_rows"] = {
        "user": int(len(users)),
        "shift": int(len(shifts)),
        "event": int(len(events)),
    }
    return users, shifts, events, checks


def _build_training_frame(
    users: pd.DataFrame, shifts: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    shifts_for_join = shifts.rename(columns={"id": "shift_id"}).copy()
    merged_events = events.merge(
        shifts_for_join[["shift_id", "start_at", "employer_id", "workplace_id"]],
        on="shift_id",
        how="inner",
    )
    # Prevent leakage from events after shift start.
    merged_events = merged_events[merged_events["ts"] <= merged_events["start_at"]].copy()

    grouped = merged_events.groupby(["user_id", "shift_id"], as_index=False).agg(
        first_ts=("ts", "min"),
        view_cnt=("interaction", lambda s: int((s == "VIEW").sum())),
        apply_cnt=("interaction", lambda s: int((s == "APPLY").sum())),
        finished_cnt=("interaction", lambda s: int((s == "FINISHED").sum())),
        user_cancel_cnt=("interaction", lambda s: int((s == "USER_CANCEL").sum())),
        system_cancel_cnt=("interaction", lambda s: int((s == "SYSTEM_CANCEL").sum())),
    )
    grouped["target"] = ((grouped["apply_cnt"] + grouped["finished_cnt"]) > 0).astype(int)

    user_totals = grouped.groupby("user_id", as_index=False).agg(
        user_total_views=("view_cnt", "sum"),
        user_total_applies=("apply_cnt", "sum"),
        user_total_finished=("finished_cnt", "sum"),
    )
    grouped = grouped.merge(user_totals, on="user_id", how="left")
    grouped["user_hist_views"] = grouped["user_total_views"] - grouped["view_cnt"]
    grouped["user_hist_applies"] = grouped["user_total_applies"] - grouped["apply_cnt"]
    grouped["user_hist_finished"] = grouped["user_total_finished"] - grouped["finished_cnt"]

    base = grouped.merge(shifts_for_join, on="shift_id", how="inner")
    base = base.merge(
        users, left_on="user_id", right_on="id", how="inner", suffixes=("_shift", "_user")
    )
    base["location_match"] = (base["location_id_shift"] == base["location_id_user"]).astype(int)
    base["need_mk_match"] = (base["need_mk"] == base["has_mk"]).astype(int)

    finished = merged_events[merged_events["interaction"] == "FINISHED"][
        ["user_id", "employer_id", "workplace_id"]
    ].copy()
    emp_finished = (
        finished.groupby(["user_id", "employer_id"], as_index=False)
        .size()
        .rename(columns={"size": "user_finished_employer"})
    )
    wp_finished = (
        finished.groupby(["user_id", "workplace_id"], as_index=False)
        .size()
        .rename(columns={"size": "user_finished_workplace"})
    )
    base = base.merge(emp_finished, on=["user_id", "employer_id"], how="left")
    base = base.merge(wp_finished, on=["user_id", "workplace_id"], how="left")
    base["user_finished_employer"] = base["user_finished_employer"].fillna(0)
    base["user_finished_workplace"] = base["user_finished_workplace"].fillna(0)
    return base


def _time_split(frame: pd.DataFrame, test_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        raise ValueError("Training frame is empty after preprocessing.")
    unique_ts = np.array(sorted(frame["start_at"].dropna().unique()))
    if unique_ts.size < 2:
        raise ValueError("Not enough temporal points for 80/20 split.")
    split_idx = max(1, int(unique_ts.size * (1 - test_ratio)))
    split_idx = min(split_idx, unique_ts.size - 1)
    split_border = unique_ts[split_idx]
    train = frame[frame["start_at"] < split_border].copy()
    test = frame[frame["start_at"] >= split_border].copy()
    if train.empty or test.empty:
        raise ValueError("Time split produced empty train or test set.")
    return train, test


def _build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
    max_iter: int,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    model = LogisticRegression(max_iter=max_iter, random_state=random_state, solver="lbfgs")
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def _generate_shap_plots(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    output_dir: Path,
    sample_size: int,
) -> dict[str, str]:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    x_train_t = preprocessor.transform(x_train)
    x_test_t = preprocessor.transform(x_test)
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(x_train_t, "toarray"):
        x_train_arr = x_train_t.toarray()
    else:
        x_train_arr = np.asarray(x_train_t)
    if hasattr(x_test_t, "toarray"):
        x_test_arr = x_test_t.toarray()
    else:
        x_test_arr = np.asarray(x_test_t)

    n = min(sample_size, x_test_arr.shape[0])
    x_test_sample = x_test_arr[:n]
    x_train_sample = x_train_arr[: min(sample_size, x_train_arr.shape[0])]

    explainer = shap.LinearExplainer(model, x_train_sample)
    shap_values = explainer(x_test_sample)

    summary_path = plots_dir / "shap_summary.png"
    bar_path = plots_dir / "shap_bar.png"

    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values.values, x_test_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=140)
    plt.close()

    plt.figure(figsize=(12, 6))
    shap.summary_plot(
        shap_values.values, x_test_sample, feature_names=feature_names, plot_type="bar", show=False
    )
    plt.tight_layout()
    plt.savefig(bar_path, dpi=140)
    plt.close()

    return {"shap_summary": str(summary_path), "shap_bar": str(bar_path)}


def run_training(cfg: TrainConfig) -> dict[str, object]:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Stage 1/8: Loading and validating train CSV contracts")
    users, shifts, events, checks = _load_and_validate_data(cfg)
    LOGGER.info(
        "Loaded rows after cleanup: users=%s shifts=%s events=%s",
        len(users),
        len(shifts),
        len(events),
    )

    LOGGER.info("Stage 2/8: Building training frame and target")
    frame = _build_training_frame(users, shifts, events)
    LOGGER.info("Built training frame rows=%s", len(frame))

    LOGGER.info("Stage 3/8: Time split (~80/20) without leakage")
    train_frame, test_frame = _time_split(frame, cfg.test_ratio)
    LOGGER.info("Split rows: train=%s test=%s", len(train_frame), len(test_frame))

    feature_columns = [
        "has_mk",
        "is_strict_location",
        "need_mk",
        "id_differential",
        "hours",
        "reward",
        "capacity",
        "location_match",
        "need_mk_match",
        "view_cnt",
        "user_cancel_cnt",
        "system_cancel_cnt",
        "user_hist_views",
        "user_hist_applies",
        "user_hist_finished",
        "user_finished_employer",
        "user_finished_workplace",
        "task_type",
    ]
    missing = [c for c in feature_columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns after preprocessing: {missing}")

    numeric_features = [
        "hours",
        "reward",
        "capacity",
        "location_match",
        "need_mk_match",
        "view_cnt",
        "user_cancel_cnt",
        "system_cancel_cnt",
        "user_hist_views",
        "user_hist_applies",
        "user_hist_finished",
        "user_finished_employer",
        "user_finished_workplace",
        "has_mk",
        "is_strict_location",
        "need_mk",
        "id_differential",
    ]
    categorical_features = ["task_type"]

    x_train = train_frame[feature_columns].copy()
    x_test = test_frame[feature_columns].copy()
    for col in ["has_mk", "is_strict_location", "need_mk", "id_differential"]:
        x_train[col] = x_train[col].astype(int)
        x_test[col] = x_test[col].astype(int)
    y_train = train_frame["target"].astype(int)

    LOGGER.info("Stage 4/8: Feature list and sample preview")
    LOGGER.info("Feature columns: %s", ", ".join(feature_columns))
    LOGGER.info("Feature sample:\n%s", x_train.head(5).to_string(index=False))

    # """ EXTENSION POINT: swap baseline model/pipeline while keeping artifact contract stable. """
    LOGGER.info("Stage 5/8: Fitting LogisticRegression baseline")
    pipeline = _build_pipeline(
        numeric_features, categorical_features, cfg.random_state, cfg.max_iter
    )
    pipeline.fit(x_train, y_train)

    LOGGER.info("Stage 6/8: Running inference and calculating target metric")
    proba = pipeline.predict_proba(x_test)[:, 1]
    metric_df = test_frame[["shift_id", "start_at", "capacity", "target"]].copy()
    metric_df["score"] = proba
    metric_result = calculate_target_metric(metric_df)
    metrics = {
        "target_metric": metric_result.target_metric,
        "evaluated_days": metric_result.evaluated_days,
        "evaluated_groups": metric_result.evaluated_groups,
        "evaluated_shifts": metric_result.evaluated_shifts,
        "day_metrics": metric_result.day_metrics,
        "test_rows": int(len(test_frame)),
        "train_rows": int(len(train_frame)),
    }

    LOGGER.info("Stage 7/8: Saving model and artifacts to %s", output_dir)
    with (output_dir / "model.pkl").open("wb") as f:
        pickle.dump(pipeline, f)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "feature_schema.json").write_text(
        json.dumps(
            {
                "feature_columns": feature_columns,
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
                "examples": x_train.head(5).to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    (output_dir / "train_config.json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "data_contract_check.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_lines = [
        "# Train Report",
        "",
        "## Data",
        "",
        f"- train_rows: {len(train_frame):,}",
        f"- test_rows: {len(test_frame):,}",
        "",
        "## Target metric (by regulation)",
        "",
        f"- target_metric: {metrics['target_metric']}",
        f"- evaluated_days: {metrics['evaluated_days']}",
        f"- evaluated_groups: {metrics['evaluated_groups']}",
        f"- evaluated_shifts: {metrics['evaluated_shifts']}",
    ]

    shap_result: dict[str, str] = {}
    if cfg.skip_shap:
        report_lines.extend(["", "## SHAP", "", "- SHAP skipped by config (--skip-shap)."])
    else:
        try:
            shap_result = _generate_shap_plots(
                pipeline, x_train, x_test, output_dir, cfg.shap_sample_size
            )
            report_lines.extend(
                [
                    "",
                    "## SHAP",
                    "",
                    f"- shap_summary: {shap_result['shap_summary']}",
                    f"- shap_bar: {shap_result['shap_bar']}",
                ]
            )
        except Exception as exc:  # noqa: BLE001
            skip_path = output_dir / "plots" / "shap_skipped.txt"
            skip_path.parent.mkdir(parents=True, exist_ok=True)
            skip_path.write_text(f"SHAP generation failed: {exc}", encoding="utf-8")
            report_lines.extend(["", "## SHAP", "", f"- SHAP generation failed: {exc}"])

    (output_dir / "train_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    LOGGER.info("Stage 8/8: Training pipeline finished successfully")
    return {"metrics": metrics, "shap": shap_result}
