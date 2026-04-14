from __future__ import annotations

import argparse
import datetime as dt
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
from zero import ZeroClient


@dataclass
class LoadResult:
    p50_ms: float
    p80_ms: float
    p95_ms: float
    rpm: float
    ok_calls: int
    failed_calls: int


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=float), p * 100.0))


def _wait_until_ready(client: ZeroClient, timeout_sec: int, poll_interval_sec: float) -> None:
    started = time.perf_counter()
    while True:
        response = client.call("ready", None)
        if int(response.get("status_code", 200)) == 200 and bool(response.get("ready")):
            return
        if (time.perf_counter() - started) > timeout_sec:
            raise TimeoutError(f"ready timeout exceeded: {timeout_sec}s")
        time.sleep(poll_interval_sec)


def _bootstrap_predict(client: ZeroClient) -> dict[str, object]:
    now = dt.datetime.now(tz=dt.UTC).isoformat()
    user_payload = {
        "items": [
            {
                "id": "load-u1",
                "location_id": "load-loc",
                "is_strict_location": True,
                "has_mk": True,
            }
        ]
    }
    shift_item = {
        "id": "load-s1",
        "start_at": now,
        "location_id": "load-loc",
        "task_type": "picker",
        "employer_id": "load-emp",
        "workplace_id": "load-wp",
        "need_mk": True,
        "id_differential": False,
        "hours": 8,
        "reward": 1000.0,
        "capacity": 1,
    }
    shift_payload = {"items": [shift_item]}
    event_payload = {
        "items": [
            {
                "id": str(uuid4()),
                "shift_id": "load-s1",
                "user_id": "load-u1",
                "interaction": "VIEW",
                "ts": now,
            }
        ]
    }
    client.call("user", user_payload)
    client.call("shift", shift_payload)
    client.call("event", event_payload)
    client.call("prepare", None)
    _wait_until_ready(client, timeout_sec=60, poll_interval_sec=0.5)
    return shift_item


def run(host: str, port: int, requests: int, max_rpm: int, rpc_timeout_ms: int) -> LoadResult:
    latencies: list[float] = []
    client = ZeroClient(host, port, default_timeout=rpc_timeout_ms)
    ok_calls = 0
    failed_calls = 0
    shift = _bootstrap_predict(client)
    start = time.perf_counter()
    min_interval_sec = 60.0 / max_rpm if max_rpm > 0 else 0.0
    last_call_at = 0.0
    try:
        for _ in range(requests):
            if min_interval_sec > 0:
                now = time.perf_counter()
                wait_for = (last_call_at + min_interval_sec) - now
                if wait_for > 0:
                    time.sleep(wait_for)
                last_call_at = time.perf_counter()

            payload = {"shift": shift, "limit": 10}
            req_start = time.perf_counter()
            response = client.call("predict", payload)
            latencies.append((time.perf_counter() - req_start) * 1000)
            if int(response.get("status_code", 200)) == 200:
                ok_calls += 1
            else:
                failed_calls += 1
    finally:
        client.close()
    elapsed = time.perf_counter() - start
    rpm = requests / elapsed * 60 if elapsed else 0.0
    return LoadResult(
        p50_ms=percentile(latencies, 0.50),
        p80_ms=percentile(latencies, 0.80),
        p95_ms=percentile(latencies, 0.95),
        rpm=rpm,
        ok_calls=ok_calls,
        failed_calls=failed_calls,
    )


def _write_report(path: Path, result: LoadResult) -> None:
    lines = [
        "# Predict load-test report",
        "",
        f"- p50_ms: {result.p50_ms:.3f}",
        f"- p80_ms: {result.p80_ms:.3f}",
        f"- p95_ms: {result.p95_ms:.3f}",
        f"- rpm: {result.rpm:.3f}",
        f"- ok_calls: {result.ok_calls}",
        f"- failed_calls: {result.failed_calls}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load test for predict RPC endpoint.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--max-rpm", type=int, default=200)
    parser.add_argument("--rpc-timeout-ms", type=int, default=2000)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("artifacts/load_test/load_test_report.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        host=args.host,
        port=args.port,
        requests=args.requests,
        max_rpm=args.max_rpm,
        rpc_timeout_ms=args.rpc_timeout_ms,
    )
    _write_report(args.report_path, result)
    print("Load test result")
    print(f"p50: {result.p50_ms:.2f} ms")
    print(f"p80: {result.p80_ms:.2f} ms")
    print(f"p95: {result.p95_ms:.2f} ms")
    print(f"rpm: {result.rpm:.2f}")
    print(f"ok_calls: {result.ok_calls}")
    print(f"failed_calls: {result.failed_calls}")
    print(f"report: {args.report_path}")


if __name__ == "__main__":
    main()
