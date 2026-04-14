from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from uuid import NAMESPACE_DNS, UUID, uuid5

import numpy as np
import pandas as pd
from zero import ZeroClient

from hackaton.eval.metric import MetricResult, calculate_target_metric

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvalConfig:
    host: str
    port: int
    user_path: str
    shift_path: str
    event_path: str
    val_apply_path: str
    val_shift_path: str
    val_event_path: str
    output_dir: str
    limit: int = 10
    batch_size: int = 1000
    prepare_initial_timeout_sec: int = 20 * 60
    prepare_day_timeout_sec: int = 5 * 60
    poll_interval_sec: int = 5
    rpc_timeout_ms: int = 10_000
    predict_retry_on_not_ready: int = 5
    predict_max_concurrency: int = 1
    predict_max_rpm: int = 200


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=float), p))


def _with_timing(callable_fn) -> tuple[dict, float]:
    start = time.perf_counter()
    response = callable_fn()
    elapsed_ms = (time.perf_counter() - start) * 1000
    return response, elapsed_ms


def _ensure_ok(response: dict, endpoint: str) -> None:
    status_code = int(response.get("status_code", 200))
    if status_code >= 400:
        raise RuntimeError(f"{endpoint} failed with status={status_code}, response={response}")


def _to_uuid_string(raw: str) -> str:
    text = str(raw)
    try:
        return str(UUID(text))
    except Exception:  # noqa: BLE001
        return str(uuid5(NAMESPACE_DNS, text))


def _load_csvs(cfg: EvalConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    users = pd.read_csv(cfg.user_path)
    train_shifts = pd.read_csv(cfg.shift_path)
    train_events = pd.read_csv(cfg.event_path)
    val_apply = pd.read_csv(cfg.val_apply_path)
    val_shifts = pd.read_csv(cfg.val_shift_path)
    val_events = pd.read_csv(cfg.val_event_path)

    train_shifts["start_at"] = pd.to_datetime(train_shifts["start_at"], utc=True, errors="coerce")
    train_events["ts"] = pd.to_datetime(train_events["ts"], utc=True, errors="coerce")
    val_shifts["start_at"] = pd.to_datetime(val_shifts["start_at"], utc=True, errors="coerce")
    val_events["ts"] = pd.to_datetime(val_events["ts"], utc=True, errors="coerce")
    val_apply["date"] = pd.to_datetime(val_apply["date"], errors="coerce").dt.date
    return users, train_shifts, train_events, val_apply, val_shifts, val_events


def _iter_batches(df: pd.DataFrame, batch_size: int):
    for start in range(0, len(df), batch_size):
        yield df.iloc[start : start + batch_size]


def _upload_users(client: ZeroClient, users: pd.DataFrame, batch_size: int) -> int:
    accepted_total = 0
    for idx, batch in enumerate(_iter_batches(users, batch_size), start=1):
        payload = {
            "items": [
                {
                    "id": str(row["id"]),
                    "location_id": str(row["location_id"]),
                    "is_strict_location": bool(row["is_strict_location"]),
                    "has_mk": bool(row["has_mk"]),
                }
                for _, row in batch.iterrows()
            ]
        }
        response = client.call("user", payload)
        _ensure_ok(response, "user")
        accepted_total += int(response.get("accepted", 0))
        LOGGER.info("Uploaded users batch %s, accepted=%s", idx, response.get("accepted", 0))
    return accepted_total


def _shift_payload(row: pd.Series) -> dict[str, object]:
    return {
        "id": str(row["id"]),
        "start_at": pd.to_datetime(row["start_at"], utc=True, errors="coerce").isoformat(),
        "location_id": str(row["location_id"]),
        "task_type": str(row["task_type"]),
        "employer_id": str(row["employer_id"]),
        "workplace_id": str(row["workplace_id"]),
        "need_mk": bool(row["need_mk"]),
        "id_differential": bool(row["id_differential"]),
        "hours": int(row["hours"]),
        "reward": float(row["reward"]),
        "capacity": int(row["capacity"]),
    }


def _upload_shifts(client: ZeroClient, shifts: pd.DataFrame, batch_size: int) -> int:
    accepted_total = 0
    for idx, batch in enumerate(_iter_batches(shifts, batch_size), start=1):
        payload = {"items": [_shift_payload(row) for _, row in batch.iterrows()]}
        response = client.call("shift", payload)
        _ensure_ok(response, "shift")
        accepted_total += int(response.get("accepted", 0))
        LOGGER.info("Uploaded shifts batch %s, accepted=%s", idx, response.get("accepted", 0))
    return accepted_total


def _upload_events(client: ZeroClient, events: pd.DataFrame, batch_size: int) -> int:
    accepted_total = 0
    for idx, batch in enumerate(_iter_batches(events, batch_size), start=1):
        payload = {
            "items": [
                {
                    "id": _to_uuid_string(row["id"]),
                    "shift_id": str(row["shift_id"]),
                    "user_id": str(row["user_id"]),
                    "interaction": str(row["interaction"]),
                    "ts": pd.to_datetime(row["ts"], utc=True, errors="coerce").isoformat(),
                }
                for _, row in batch.iterrows()
            ]
        }
        response = client.call("event", payload)
        _ensure_ok(response, "event")
        accepted_total += int(response.get("accepted", 0))
        LOGGER.info("Uploaded events batch %s, accepted=%s", idx, response.get("accepted", 0))
    return accepted_total


def _run_prepare_and_wait(client: ZeroClient, timeout_sec: int, poll_interval_sec: int) -> float:
    response = client.call("prepare", None)
    _ensure_ok(response, "prepare")
    started = time.perf_counter()
    while True:
        ready = client.call("ready", None)
        status_code = int(ready.get("status_code", 200))
        if status_code == 200 and bool(ready.get("ready")):
            return time.perf_counter() - started
        if (time.perf_counter() - started) > timeout_sec:
            raise TimeoutError(f"prepare/ready timeout exceeded: {timeout_sec}s")
        time.sleep(poll_interval_sec)


def _wait_until_ready_only(client: ZeroClient, timeout_sec: int, poll_interval_sec: int) -> float:
    started = time.perf_counter()
    while True:
        ready = client.call("ready", None)
        status_code = int(ready.get("status_code", 200))
        if status_code == 200 and bool(ready.get("ready")):
            return time.perf_counter() - started
        if (time.perf_counter() - started) > timeout_sec:
            raise TimeoutError(f"ready timeout exceeded: {timeout_sec}s")
        time.sleep(poll_interval_sec)


def _build_day_prediction_frame(
    cfg: EvalConfig,
    day_shifts: pd.DataFrame,
    day_apply: pd.DataFrame,
    limit: int,
) -> tuple[pd.DataFrame, list[float]]:
    positive_pairs = {(str(r["user_id"]), str(r["shift_id"])) for _, r in day_apply.iterrows()}
    rows: list[dict[str, object]] = []
    latencies_ms: list[float] = []
    shifts_records = [row for _, row in day_shifts.iterrows()]
    if not shifts_records:
        return pd.DataFrame(rows), latencies_ms

    min_interval_sec = 60.0 / cfg.predict_max_rpm if cfg.predict_max_rpm > 0 else 0.0

    def predict_worker(shift_row: pd.Series) -> tuple[pd.Series, list[str], float]:
        worker_client = ZeroClient(cfg.host, cfg.port, default_timeout=cfg.rpc_timeout_ms)
        try:
            payload = {"shift": _shift_payload(shift_row), "limit": int(limit)}
            attempt = 0
            while True:
                attempt += 1
                response, latency_ms = _with_timing(lambda: worker_client.call("predict", payload))
                status_code = int(response.get("status_code", 200))
                if status_code != 503:
                    _ensure_ok(response, "predict")
                    return shift_row, [str(u) for u in response.get("user_ids", [])], latency_ms
                if attempt > cfg.predict_retry_on_not_ready:
                    _ensure_ok(response, "predict")
                LOGGER.warning(
                    "Predict returned 503 (model not ready), retry %s/%s",
                    attempt,
                    cfg.predict_retry_on_not_ready,
                )
                _wait_until_ready_only(
                    worker_client,
                    timeout_sec=max(1, cfg.poll_interval_sec * 3),
                    poll_interval_sec=cfg.poll_interval_sec,
                )
        finally:
            worker_client.close()

    max_workers = max(1, int(cfg.predict_max_concurrency))
    futures = []
    last_submit_at = 0.0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for shift_row in shifts_records:
            if min_interval_sec > 0:
                now = time.perf_counter()
                wait_for = (last_submit_at + min_interval_sec) - now
                if wait_for > 0:
                    time.sleep(wait_for)
                last_submit_at = time.perf_counter()
            futures.append(executor.submit(predict_worker, shift_row))

        for future in as_completed(futures):
            shift_row, user_ids, latency_ms = future.result()
            latencies_ms.append(latency_ms)
            for rank, user_id in enumerate(user_ids, start=1):
                score = 1.0 - (rank - 1) / max(1, len(user_ids))
                rows.append(
                    {
                        "shift_id": str(shift_row["id"]),
                        "start_at": shift_row["start_at"],
                        "capacity": int(shift_row["capacity"]),
                        "target": int((user_id, str(shift_row["id"])) in positive_pairs),
                        "score": float(score),
                    }
                )
    return pd.DataFrame(rows), latencies_ms


def run_evaluation(cfg: EvalConfig) -> dict[str, object]:
    if cfg.predict_max_rpm > 200:
        raise ValueError("predict_max_rpm must be <= 200 to satisfy evaluation limits")
    if cfg.predict_max_concurrency < 1:
        raise ValueError("predict_max_concurrency must be >= 1")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Stage 1/9: Loading evaluation datasets")
    users, train_shifts, train_events, val_apply, val_shifts, val_events = _load_csvs(cfg)

    LOGGER.info("Stage 2/9: Connecting to RPC service")
    client = ZeroClient(cfg.host, cfg.port, default_timeout=cfg.rpc_timeout_ms)
    stop_reason = ""
    day_reports: list[dict[str, object]] = []
    all_predict_latencies_ms: list[float] = []
    total_predict_wall_seconds = 0.0
    prepare_durations_sec: list[float] = []
    try:
        LOGGER.info("Stage 3/9: Uploading bootstrap train data")
        users_uploaded = _upload_users(client, users, cfg.batch_size)
        shifts_uploaded = _upload_shifts(client, train_shifts, cfg.batch_size)
        events_uploaded = _upload_events(client, train_events, cfg.batch_size)

        LOGGER.info("Stage 4/9: Running initial prepare (<= %ss)", cfg.prepare_initial_timeout_sec)
        prepare_initial_sec = _run_prepare_and_wait(client, cfg.prepare_initial_timeout_sec, cfg.poll_interval_sec)
        prepare_durations_sec.append(prepare_initial_sec)

        eval_days = sorted(d for d in val_apply["date"].dropna().unique())
        LOGGER.info("Stage 5/9: Running day-by-day evaluation for %s days", len(eval_days))

        for day_idx, day in enumerate(eval_days, start=1):
            if not isinstance(day, date):
                continue
            LOGGER.info("Day %s/%s: %s | building day slices", day_idx, len(eval_days), day.isoformat())
            day_shifts = val_shifts[val_shifts["start_at"].dt.date == day].copy()
            day_apply = val_apply[val_apply["date"] == day].copy()
            if day_shifts.empty:
                LOGGER.info("Day %s: skipped (no shifts)", day.isoformat())
                continue

            LOGGER.info("Day %s: waiting service ready before prediction", day.isoformat())
            _wait_until_ready_only(client, timeout_sec=cfg.prepare_day_timeout_sec, poll_interval_sec=cfg.poll_interval_sec)
            LOGGER.info("Day %s: predicting candidates", day.isoformat())
            predict_started = time.perf_counter()
            prediction_frame, latencies_ms = _build_day_prediction_frame(
                cfg,
                day_shifts,
                day_apply,
                cfg.limit,
            )
            predict_wall_sec = time.perf_counter() - predict_started
            total_predict_wall_seconds += predict_wall_sec
            all_predict_latencies_ms.extend(latencies_ms)
            metric_result: MetricResult = calculate_target_metric(prediction_frame)

            day_events = val_events[val_events["ts"].dt.date == day].copy()
            LOGGER.info("Day %s: waiting service ready before post-day uploads", day.isoformat())
            _wait_until_ready_only(client, timeout_sec=cfg.prepare_day_timeout_sec, poll_interval_sec=cfg.poll_interval_sec)
            LOGGER.info("Day %s: post-day upload shifts=%s events=%s", day.isoformat(), len(day_shifts), len(day_events))
            day_shifts_uploaded = _upload_shifts(client, day_shifts, cfg.batch_size)
            day_events_uploaded = _upload_events(client, day_events, cfg.batch_size) if not day_events.empty else 0

            LOGGER.info("Day %s: incremental prepare (<= %ss)", day.isoformat(), cfg.prepare_day_timeout_sec)
            prepare_day_sec = _run_prepare_and_wait(client, cfg.prepare_day_timeout_sec, cfg.poll_interval_sec)
            prepare_durations_sec.append(prepare_day_sec)

            day_reports.append(
                {
                    "date": day.isoformat(),
                    "target_metric": metric_result.target_metric,
                    "evaluated_days": metric_result.evaluated_days,
                    "evaluated_groups": metric_result.evaluated_groups,
                    "evaluated_shifts": metric_result.evaluated_shifts,
                    "group_metrics": metric_result.group_metrics,
                    "predict_calls": len(latencies_ms),
                    "predict_latency_p50_ms": _percentile(latencies_ms, 50),
                    "predict_latency_p80_ms": _percentile(latencies_ms, 80),
                    "predict_latency_p95_ms": _percentile(latencies_ms, 95),
                    "predict_wall_sec": predict_wall_sec,
                    "day_shift_upload_accepted": day_shifts_uploaded,
                    "day_event_upload_accepted": day_events_uploaded,
                    "prepare_duration_sec": prepare_day_sec,
                }
            )

        LOGGER.info("Stage 6/9: Aggregating final metric")
        overall_metric = float(np.mean([d["target_metric"] for d in day_reports])) if day_reports else 0.0
        total_predict_calls = int(sum(d["predict_calls"] for d in day_reports))
        rpm = (total_predict_calls / total_predict_wall_seconds * 60.0) if total_predict_wall_seconds > 0 else 0.0

        summary = {
            "overall_target_metric": overall_metric,
            "days_evaluated": len(day_reports),
            "bootstrap_uploaded": {
                "users": users_uploaded,
                "shifts": shifts_uploaded,
                "events": events_uploaded,
            },
            "performance": {
                "predict_latency_p50_ms": _percentile(all_predict_latencies_ms, 50),
                "predict_latency_p80_ms": _percentile(all_predict_latencies_ms, 80),
                "predict_latency_p95_ms": _percentile(all_predict_latencies_ms, 95),
                "predict_rpm": rpm,
                "prepare_durations_sec": prepare_durations_sec,
                "prepare_duration_avg_sec": float(np.mean(prepare_durations_sec)) if prepare_durations_sec else 0.0,
            },
            "day_reports": day_reports,
            "stop_reason": stop_reason or "completed",
            "config": asdict(cfg),
        }

        LOGGER.info("Stage 7/9: Writing evaluation report artifacts")
        _write_markdown_report(output_dir / "eval_report.md", summary, users, train_shifts, train_events)
        LOGGER.info("Stage 8/9: Evaluation artifacts saved")
        return summary
    except Exception as exc:  # noqa: BLE001
        stop_reason = str(exc)
        LOGGER.exception("Stage failed: %s", stop_reason)
        raise
    finally:
        LOGGER.info("Stage 9/9: Closing RPC client")
        client.close()


def _write_markdown_report(
    report_path: Path,
    summary: dict[str, object],
    users: pd.DataFrame,
    train_shifts: pd.DataFrame,
    train_events: pd.DataFrame,
) -> None:
    perf = summary["performance"]
    lines = [
        "# Eval Report",
        "",
        "## Data stats at measurement",
        "",
        f"- train_users: {len(users):,}",
        f"- train_shifts: {len(train_shifts):,}",
        f"- train_events: {len(train_events):,}",
        f"- days_evaluated: {summary['days_evaluated']}",
        "",
        "## Overall target metric",
        "",
        f"- overall_target_metric: {summary['overall_target_metric']}",
        "",
        "## Performance",
        "",
        f"- predict_latency_p50_ms: {perf['predict_latency_p50_ms']:.3f}",
        f"- predict_latency_p80_ms: {perf['predict_latency_p80_ms']:.3f}",
        f"- predict_latency_p95_ms: {perf['predict_latency_p95_ms']:.3f}",
        f"- predict_rpm: {perf['predict_rpm']:.3f}",
        f"- prepare_duration_avg_sec: {perf['prepare_duration_avg_sec']:.1f}",
        f"- prepare_durations_sec: {perf['prepare_durations_sec']}",
        "",
        f"- stop_reason: {summary['stop_reason']}",
        "",
        "## Daily metrics",
        "",
    ]
    for day in summary["day_reports"]:
        lines.extend(
            [
                f"### {day['date']}",
                "",
                f"- target_metric: {day['target_metric']}",
                f"- evaluated_groups: {day['evaluated_groups']}",
                f"- evaluated_shifts: {day['evaluated_shifts']}",
                f"- predict_calls: {day['predict_calls']}",
                f"- predict_latency_p50_ms: {day['predict_latency_p50_ms']:.3f}",
                f"- predict_latency_p80_ms: {day['predict_latency_p80_ms']:.3f}",
                f"- predict_latency_p95_ms: {day['predict_latency_p95_ms']:.3f}",
                f"- predict_wall_sec: {day['predict_wall_sec']:.3f}",
                f"- day_shift_upload_accepted: {day['day_shift_upload_accepted']}",
                f"- day_event_upload_accepted: {day['day_event_upload_accepted']}",
                f"- prepare_duration_sec: {day['prepare_duration_sec']:.1f}",
                "",
                "Group metrics:",
                "",
            ]
        )
        if day["group_metrics"]:
            for gm in day["group_metrics"]:
                lines.append(f"- capacity={gm['capacity']}: {gm['group_metric']}")
        else:
            lines.append("- no group metrics")
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")

