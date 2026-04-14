from __future__ import annotations

import logging

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import ValidationError

from hackaton.service.dto import (
    BatchEventsRequest,
    BatchShiftsRequest,
    BatchUsersRequest,
    PredictRequest,
    PredictResponse,
)
from hackaton.service.prepare_manager import PrepareManager
from hackaton.service.repositories import Repository

REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["endpoint"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Latency of API requests", ["endpoint"])
LOGGER = logging.getLogger(__name__)


class HackatonRpcService:
    def __init__(self, repository: Repository, prepare: PrepareManager) -> None:
        self.repository = repository
        self.prepare_manager = prepare

    async def user(self, payload: dict) -> dict:
        REQUEST_COUNT.labels("user").inc()
        with REQUEST_LATENCY.labels("user").time():
            request = BatchUsersRequest.model_validate(payload)
            LOGGER.info("RPC user called, batch_size=%s", len(request.items))
            accepted = await self.repository.upsert_users(request.items)
            return {"accepted": accepted}

    async def user_stat(self, _: dict | None = None) -> dict:
        REQUEST_COUNT.labels("user_stat").inc()
        with REQUEST_LATENCY.labels("user_stat").time():
            LOGGER.info("RPC user_stat called")
            return {"count": await self.repository.count_table("users")}

    async def event(self, payload: dict) -> dict:
        REQUEST_COUNT.labels("event").inc()
        with REQUEST_LATENCY.labels("event").time():
            request = BatchEventsRequest.model_validate(payload)
            LOGGER.info("RPC event called, batch_size=%s", len(request.items))
            accepted = await self.repository.insert_events(request.items)
            return {"accepted": accepted}

    async def event_stat(self, _: dict | None = None) -> dict:
        REQUEST_COUNT.labels("event_stat").inc()
        with REQUEST_LATENCY.labels("event_stat").time():
            LOGGER.info("RPC event_stat called")
            return {"count": await self.repository.count_table("events")}

    async def shift(self, payload: dict) -> dict:
        REQUEST_COUNT.labels("shift").inc()
        with REQUEST_LATENCY.labels("shift").time():
            request = BatchShiftsRequest.model_validate(payload)
            LOGGER.info("RPC shift called, batch_size=%s", len(request.items))
            accepted = await self.repository.upsert_shifts(request.items)
            return {"accepted": accepted}

    async def shift_stat(self, _: dict | None = None) -> dict:
        REQUEST_COUNT.labels("shift_stat").inc()
        with REQUEST_LATENCY.labels("shift_stat").time():
            LOGGER.info("RPC shift_stat called")
            return {"count": await self.repository.count_table("shifts")}

    async def prepare(self, _: dict | None = None) -> dict:
        REQUEST_COUNT.labels("prepare").inc()
        with REQUEST_LATENCY.labels("prepare").time():
            LOGGER.info("RPC prepare called")
            started = await self.prepare_manager.start()
            if not started:
                return {"status": "already_running", "status_code": 409}
            return {"status": "started", "status_code": 200}

    async def ready(self, _: dict | None = None) -> dict:
        REQUEST_COUNT.labels("ready").inc()
        with REQUEST_LATENCY.labels("ready").time():
            LOGGER.info("RPC ready called")
            if not self.prepare_manager.ready:
                return {"ready": False, "status_code": 425}
            return {"ready": True, "status_code": 200}

    async def predict(self, payload: dict) -> dict:
        REQUEST_COUNT.labels("predict").inc()
        with REQUEST_LATENCY.labels("predict").time():
            LOGGER.info("RPC predict called")
            if not self.prepare_manager.ready:
                return {"user_ids": [], "status_code": 503, "detail": "model is in prepare state"}
            try:
                request = PredictRequest.model_validate(payload)
            except ValidationError as exc:
                return {"user_ids": [], "status_code": 422, "detail": str(exc)}

            """ 
                EXTENSION POINT
                Ваше решение должно быть здесь.
            """
            candidates = await self.repository.find_top_candidates(
                location_id=request.shift.location_id,
                need_mk=request.shift.need_mk,
                limit=request.limit,
            )
            if not candidates:
                candidates = await self.repository.fallback_candidates(limit=request.limit)
            if not candidates:
                return {"user_ids": [], "status_code": 400, "detail": "no users loaded"}
            result = PredictResponse(user_ids=candidates)
            return {"user_ids": result.user_ids, "status_code": 200}

    async def health(self, _: dict | None = None) -> dict:
        REQUEST_COUNT.labels("health").inc()
        LOGGER.info("RPC health called")
        return {"status": "ok", "status_code": 200}

    async def metrics(self, _: dict | None = None) -> dict:
        REQUEST_COUNT.labels("metrics").inc()
        LOGGER.info("RPC metrics called")
        return {
            "content_type": CONTENT_TYPE_LATEST,
            "payload": generate_latest().decode("utf-8"),
            "status_code": 200,
        }
