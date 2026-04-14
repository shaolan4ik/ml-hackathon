from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path
from uuid import uuid4

from hackaton.service.app import HackatonRpcService
from hackaton.service.db import init_db_for
from hackaton.service.prepare_manager import PrepareManager
from hackaton.service.repositories import Repository


def _build_service(tmp_path: Path, prepare_sleep_seconds: float = 0.0) -> HackatonRpcService:
    db_path = str(tmp_path / "e2e.db")
    asyncio.run(init_db_for(db_path))
    repository = Repository(db_path=db_path)
    prepare = PrepareManager(sleep_seconds=prepare_sleep_seconds)
    return HackatonRpcService(repository=repository, prepare=prepare)


def test_prepare_ready_predict_contract(tmp_path: Path) -> None:
    service = _build_service(tmp_path, prepare_sleep_seconds=0.05)

    async def scenario() -> None:
        now = dt.datetime.now(tz=dt.UTC).isoformat()
        user_payload = {
            "items": [
                {
                    "id": "u1",
                    "location_id": "loc-1",
                    "is_strict_location": True,
                    "has_mk": True,
                }
            ]
        }
        shift_payload = {
            "items": [
                {
                    "id": "s1",
                    "start_at": now,
                    "location_id": "loc-1",
                    "task_type": "picker",
                    "employer_id": "emp-1",
                    "workplace_id": "wp-1",
                    "need_mk": True,
                    "id_differential": False,
                    "hours": 8,
                    "reward": 1200.0,
                    "capacity": 2,
                }
            ]
        }
        event_payload = {
            "items": [
                {
                    "id": str(uuid4()),
                    "shift_id": "s1",
                    "user_id": "u1",
                    "interaction": "VIEW",
                    "ts": now,
                }
            ]
        }
        predict_payload = {"shift": shift_payload["items"][0], "limit": 10}

        assert (await service.user(user_payload))["accepted"] == 1
        assert (await service.shift(shift_payload))["accepted"] == 1
        assert (await service.event(event_payload))["accepted"] == 1

        prepare_started = await service.prepare(None)
        assert prepare_started["status_code"] == 200
        assert prepare_started["status"] == "started"

        not_ready = await service.ready(None)
        assert not_ready["status_code"] == 425
        assert not not_ready["ready"]

        predict_while_prepare = await service.predict(predict_payload)
        assert predict_while_prepare["status_code"] == 503

        await asyncio.sleep(0.1)
        ready = await service.ready(None)
        assert ready["status_code"] == 200
        assert ready["ready"]

        predict = await service.predict(predict_payload)
        assert predict["status_code"] == 200
        assert predict["user_ids"] == ["u1"]

    asyncio.run(scenario())


def test_stat_endpoints_after_writes(tmp_path: Path) -> None:
    service = _build_service(tmp_path, prepare_sleep_seconds=0.0)
    now = dt.datetime.now(tz=dt.UTC).isoformat()
    user_payload = {
        "items": [{"id": "u1", "location_id": "loc-1", "is_strict_location": True, "has_mk": True}]
    }
    shift_payload = {
        "items": [
            {
                "id": "s1",
                "start_at": now,
                "location_id": "loc-1",
                "task_type": "picker",
                "employer_id": "emp-1",
                "workplace_id": "wp-1",
                "need_mk": True,
                "id_differential": False,
                "hours": 8,
                "reward": 1200.0,
                "capacity": 2,
            }
        ]
    }
    event_payload = {
        "items": [
            {
                "id": str(uuid4()),
                "shift_id": "s1",
                "user_id": "u1",
                "interaction": "APPLY",
                "ts": now,
            }
        ]
    }

    asyncio.run(service.user(user_payload))
    asyncio.run(service.shift(shift_payload))
    asyncio.run(service.event(event_payload))

    assert asyncio.run(service.user_stat(None))["count"] == 1
    assert asyncio.run(service.shift_stat(None))["count"] == 1
    assert asyncio.run(service.event_stat(None))["count"] == 1
