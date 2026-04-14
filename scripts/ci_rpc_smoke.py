from __future__ import annotations

import datetime as dt
import os
import time
from uuid import uuid4

from zero import ZeroClient


def _wait_health(client: ZeroClient, timeout_sec: int = 30) -> None:
    started = time.perf_counter()
    while True:
        try:
            response = client.call("health", None)
            if int(response.get("status_code", 500)) == 200:
                return
        except Exception:  # noqa: BLE001
            pass
        if (time.perf_counter() - started) > timeout_sec:
            raise TimeoutError("Service did not become healthy in time")
        time.sleep(0.5)


def _wait_ready(client: ZeroClient, timeout_sec: int = 30) -> None:
    started = time.perf_counter()
    while True:
        response = client.call("ready", None)
        if int(response.get("status_code", 500)) == 200 and bool(response.get("ready")):
            return
        if (time.perf_counter() - started) > timeout_sec:
            raise TimeoutError("Service did not become ready in time")
        time.sleep(0.5)


def main() -> None:
    host = os.getenv("RPC_HOST", "127.0.0.1")
    port = int(os.getenv("RPC_PORT", "8000"))
    client = ZeroClient(host, port, default_timeout=10_000)
    try:
        _wait_health(client)

        now = dt.datetime.now(tz=dt.UTC).isoformat()
        user_payload = {
            "items": [
                {
                    "id": "ci-user-1",
                    "location_id": "ci-loc-1",
                    "is_strict_location": True,
                    "has_mk": True,
                }
            ]
        }
        shift_payload = {
            "items": [
                {
                    "id": "ci-shift-1",
                    "start_at": now,
                    "location_id": "ci-loc-1",
                    "task_type": "picker",
                    "employer_id": "ci-emp-1",
                    "workplace_id": "ci-wp-1",
                    "need_mk": True,
                    "id_differential": False,
                    "hours": 8,
                    "reward": 1000.0,
                    "capacity": 1,
                }
            ]
        }
        event_payload = {
            "items": [
                {
                    "id": str(uuid4()),
                    "shift_id": "ci-shift-1",
                    "user_id": "ci-user-1",
                    "interaction": "VIEW",
                    "ts": now,
                }
            ]
        }

        assert int(client.call("user", user_payload).get("accepted", 0)) == 1
        assert int(client.call("shift", shift_payload).get("accepted", 0)) == 1
        assert int(client.call("event", event_payload).get("accepted", 0)) == 1

        prepare_response = client.call("prepare", None)
        assert int(prepare_response.get("status_code", 500)) in (200, 409)
        _wait_ready(client)

        predict_payload = {"shift": shift_payload["items"][0], "limit": 10}
        predict_response = client.call("predict", predict_payload)
        assert int(predict_response.get("status_code", 500)) == 200
        assert len(predict_response.get("user_ids", [])) > 0

        print("CI RPC smoke passed: service started and predict returned candidates.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
