from __future__ import annotations

import json
import logging
import pickle
from os import getenv
from pathlib import Path

import aiosqlite
import numpy as np
import pandas as pd
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
MODEL_ARTIFACT_DIR = Path(getenv("MODEL_ARTIFACT_DIR", "artifacts/train"))
PREDICTION_OFFSET_DAYS = 2
HOLIDAYS = pd.to_datetime(
    [
        "2026-01-01",
        "2026-01-02",
        "2026-01-03",
        "2026-01-04",
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
        "2026-01-09",
        "2026-01-10",
        "2026-01-11",
        "2026-02-23",
        "2026-03-09",
        "2026-05-01",
        "2026-05-11",
    ]
)


class HackatonRpcService:
    def __init__(self, repository: Repository, prepare: PrepareManager) -> None:
        self.repository = repository
        self.prepare_manager = prepare
        self.model: object | None = None
        self.feature_columns: list[str] = []
        self.category_maps: dict[str, list[object]] = {}
        self.users_df = pd.DataFrame()
        self.shifts_df = pd.DataFrame()
        self.events_df = pd.DataFrame()
        self.model_ready = False

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
            await self._refresh_model_state()
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
            if self.model_ready:
                try:
                    candidates = self._predict_with_model(request.shift, request.limit)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("Model inference failed, using baseline ranking: %s", exc)
                    candidates = []
                if candidates:
                    result = PredictResponse(user_ids=candidates)
                    return {"user_ids": result.user_ids, "status_code": 200}

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

    async def _refresh_model_state(self) -> None:
        self.model_ready = False
        self.model = None
        self.feature_columns = []
        self.category_maps = {}

        await self._load_online_tables()

        model_path = MODEL_ARTIFACT_DIR / "model.pkl"
        schema_path = MODEL_ARTIFACT_DIR / "feature_schema.json"
        if not model_path.exists() or not schema_path.exists():
            LOGGER.warning(
                "Model artifacts not found in %s, using baseline ranking",
                MODEL_ARTIFACT_DIR,
            )
            return

        with model_path.open("rb") as f:
            self.model = pickle.load(f)
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.feature_columns = list(schema["feature_columns"])
        self.category_maps = dict(schema.get("category_maps", {}))
        self.model_ready = True
        LOGGER.info(
            "Loaded model artifacts: features=%s users=%s shifts=%s events=%s",
            len(self.feature_columns),
            len(self.users_df),
            len(self.shifts_df),
            len(self.events_df),
        )

    async def _load_online_tables(self) -> None:
        async with aiosqlite.connect(self.repository.db_path) as db:
            self.users_df = await self._read_table(db, "users")
            self.shifts_df = await self._read_table(db, "shifts")
            self.events_df = await self._read_table(db, "events")

        if not self.users_df.empty:
            self.users_df["id"] = self.users_df["id"].astype(str)
            self.users_df["location_id"] = self.users_df["location_id"].astype(str)
            self.users_df["is_strict_location"] = self.users_df["is_strict_location"].astype(int)
            self.users_df["has_mk"] = self.users_df["has_mk"].astype(int)

        if not self.shifts_df.empty:
            self.shifts_df = self.shifts_df.rename(columns={"id": "shift_id"})
            for col in ["shift_id", "location_id", "task_type", "employer_id", "workplace_id"]:
                self.shifts_df[col] = self.shifts_df[col].astype(str)
            self.shifts_df["start_at"] = pd.to_datetime(
                self.shifts_df["start_at"], utc=True, errors="coerce"
            ).dt.tz_localize(None)
            for col in ["need_mk", "id_differential"]:
                self.shifts_df[col] = self.shifts_df[col].astype(int)
            for col in ["hours", "reward", "capacity"]:
                self.shifts_df[col] = pd.to_numeric(self.shifts_df[col], errors="coerce")

        if not self.events_df.empty:
            for col in ["id", "shift_id", "user_id", "interaction"]:
                self.events_df[col] = self.events_df[col].astype(str)
            self.events_df["interaction"] = self.events_df["interaction"].str.upper()
            self.events_df["ts"] = pd.to_datetime(
                self.events_df["ts"], utc=True, errors="coerce"
            ).dt.tz_localize(None)

    async def _read_table(self, db: aiosqlite.Connection, table: str) -> pd.DataFrame:
        cursor = await db.execute(f"SELECT * FROM {table}")  # nosec - fixed internal table names
        rows = await cursor.fetchall()
        columns = [item[0] for item in cursor.description]
        return pd.DataFrame(rows, columns=columns)

    def _predict_with_model(self, shift, limit: int) -> list[str]:
        if self.model is None or not self.feature_columns or self.users_df.empty:
            return []

        features = self._build_feature_frame(shift)
        if features.empty:
            return []

        missing = [col for col in self.feature_columns if col not in features.columns]
        if missing:
            raise ValueError(f"Missing online feature columns: {missing}")

        x = features[self.feature_columns].copy()
        for col, categories in self.category_maps.items():
            if col in x.columns:
                x[col] = pd.Categorical(x[col], categories=categories).codes
        x = x.replace([np.inf, -np.inf], np.nan).fillna(-1)

        scores = self.model.predict_proba(x)[:, 1]
        ranked = (
            pd.DataFrame({"user_id": features["user_id"].astype(str), "score": scores})
            .sort_values(["score", "user_id"], ascending=[False, True])
            .head(limit)
        )
        return ranked["user_id"].tolist()

    def _build_feature_frame(self, shift) -> pd.DataFrame:
        users = self.users_df.copy()
        shift_start = self._to_naive_utc(shift.start_at)
        pred_moment = shift_start - pd.Timedelta(days=PREDICTION_OFFSET_DAYS)
        history = self._history_before(pred_moment)
        merged_history = self._merged_history(history)

        frame = users.rename(
            columns={"id": "user_id", "location_id": "location_id_user"}
        ).reset_index(drop=True)
        frame["shift_id"] = str(shift.id)
        frame["start_at"] = shift_start
        frame["location_id_shift"] = str(shift.location_id)
        frame["task_type"] = str(shift.task_type)
        frame["employer_id"] = str(shift.employer_id)
        frame["workplace_id"] = str(shift.workplace_id)
        frame["need_mk"] = int(shift.need_mk)
        frame["id_differential"] = int(shift.id_differential)
        frame["hours"] = int(shift.hours)
        frame["reward"] = float(shift.reward)
        frame["capacity"] = int(shift.capacity)
        frame["reward_per_hour"] = frame["reward"] / frame["hours"].replace(0, np.nan)
        frame["shift_hour"] = shift_start.hour
        frame["shift_day_of_week"] = shift_start.dayofweek
        frame["shift_time_of_day"] = self._time_of_day(shift_start.hour)
        frame["mk_match_2"] = np.where(
            frame["need_mk"] == 1,
            np.where(frame["has_mk"] == 1, 1, 0),
            2,
        )

        self._add_user_window_features(frame, merged_history, shift_start)
        self._add_global_history_features(frame, merged_history, shift)
        return frame

    def _history_before(self, pred_moment: pd.Timestamp) -> pd.DataFrame:
        if self.events_df.empty:
            return self.events_df.copy()
        return self.events_df[self.events_df["ts"] < pred_moment].copy()

    def _merged_history(self, history: pd.DataFrame) -> pd.DataFrame:
        if history.empty or self.shifts_df.empty:
            return pd.DataFrame()
        shifts = self.shifts_df[
            ["shift_id", "start_at", "location_id", "task_type", "employer_id", "hours", "reward"]
        ].copy()
        shifts["reward_per_hour"] = shifts["reward"] / shifts["hours"].replace(0, np.nan)
        merged = history.merge(shifts, on="shift_id", how="left", suffixes=("", "_shift"))
        cancel_pairs = merged[
            merged["interaction"].isin(["USER_CANCEL", "SYSTEM_CANCEL"])
        ][["user_id", "shift_id"]].drop_duplicates()
        cancel_pairs["has_cancel"] = 1
        merged = merged.merge(cancel_pairs, on=["user_id", "shift_id"], how="left")
        merged["has_cancel"] = merged["has_cancel"].fillna(0)
        merged["is_successful"] = (
            (merged["interaction"] == "APPLY") & (merged["has_cancel"] == 0)
        ).astype(int)
        return merged

    def _add_user_window_features(
        self,
        frame: pd.DataFrame,
        history: pd.DataFrame,
        shift_start: pd.Timestamp,
    ) -> None:
        defaults = {
            "user_total_views": 0,
            "user_success_rate": -1,
            "user_avg_reward_per_hour": -1,
            "user_location_experience": -1,
            "user_high_paid_ratio": -1,
            "user_long_shift_ratio": -1,
        }
        if history.empty:
            for col, value in defaults.items():
                frame[col] = value
            return

        win_start = shift_start - pd.Timedelta(days=30)
        win_end = shift_start - pd.Timedelta(days=PREDICTION_OFFSET_DAYS)
        hist = history[(history["ts"] >= win_start) & (history["ts"] < win_end)].copy()
        if hist.empty:
            for col, value in defaults.items():
                frame[col] = value
            return

        median_pay = self._median_reward_per_hour()
        views = (
            hist[hist["interaction"] == "VIEW"]
            .groupby("user_id")["shift_id"]
            .nunique()
            .to_dict()
        )
        successful = hist[hist["is_successful"] == 1].drop_duplicates(["user_id", "shift_id"])
        success_counts = successful.groupby("user_id")["shift_id"].nunique().to_dict()

        avg_reward = {}
        loc_exp = {}
        high_paid = {}
        long_shift = {}
        if not successful.empty:
            avg_reward = successful.groupby("user_id")["reward_per_hour"].mean().to_dict()
            loc_exp = (
                successful.assign(
                    loc_match=(
                        successful["location_id"].astype(str)
                        == frame["location_id_shift"].iloc[0]
                    ).astype(int)
                )
                .groupby("user_id")["loc_match"]
                .sum()
                .to_dict()
            )
            high_paid = (
                successful.assign(
                    high_paid=(successful["reward_per_hour"] > median_pay).astype(int)
                )
                .groupby("user_id")["high_paid"]
                .mean()
                .to_dict()
            )
            long_shift = (
                successful.assign(long_shift=(successful["hours"] > 8).astype(int))
                .groupby("user_id")["long_shift"]
                .mean()
                .to_dict()
            )

        user_ids = frame["user_id"].astype(str).tolist()
        frame["user_total_views"] = [int(views.get(user_id, 0)) for user_id in user_ids]
        success = np.array([int(success_counts.get(user_id, 0)) for user_id in user_ids])
        views_arr = frame["user_total_views"].to_numpy()
        frame["user_success_rate"] = np.where(
            frame["user_total_views"] > 0,
            np.divide(success, views_arr, out=np.zeros_like(success, dtype=float), where=views_arr != 0),
            -1,
        )
        frame["user_avg_reward_per_hour"] = [avg_reward.get(user_id, -1) for user_id in user_ids]
        frame["user_location_experience"] = [loc_exp.get(user_id, -1) for user_id in user_ids]
        frame["user_high_paid_ratio"] = [high_paid.get(user_id, -1) for user_id in user_ids]
        frame["user_long_shift_ratio"] = [long_shift.get(user_id, -1) for user_id in user_ids]

    def _add_global_history_features(self, frame: pd.DataFrame, history: pd.DataFrame, shift) -> None:
        user_ids = frame["user_id"].astype(str).tolist()
        shift_id = str(shift.id)
        task_type = str(shift.task_type)
        employer_id = str(shift.employer_id)

        frame["shift_total_views"] = 0
        frame["free_spots_at_prediction"] = int(shift.capacity)
        frame["view_count_for_shift"] = 0
        frame["has_apply_before"] = 0
        frame["user_task_type_experience"] = 0
        frame["user_employer_experience"] = 0
        frame["employer_avg_reward"] = 0
        frame["employer_popularity"] = 0
        frame["employer_reliability"] = 1
        frame["user_weekend_ratio"] = -1
        frame["user_day_ratio"] = -1
        frame["matches_time_of_day"] = -1
        frame["user_shifts_on_same_day"] = 0

        if history.empty:
            return

        shift_hist = history[history["shift_id"].astype(str) == shift_id]
        frame["shift_total_views"] = int((shift_hist["interaction"] == "VIEW").sum())
        applies_before = int((shift_hist["interaction"] == "APPLY").sum())
        frame["free_spots_at_prediction"] = max(0, int(shift.capacity) - applies_before)

        per_pair = history.groupby(["user_id", "shift_id"])["interaction"].agg(list)
        for idx, user_id in enumerate(user_ids):
            interactions = per_pair.get((user_id, shift_id), [])
            frame.at[idx, "view_count_for_shift"] = int(
                sum(1 for item in interactions if item == "VIEW")
            )
            frame.at[idx, "has_apply_before"] = int("APPLY" in interactions)

        successful = history[history["is_successful"] == 1].drop_duplicates(
            ["user_id", "shift_id"]
        )
        if not successful.empty:
            by_user_tasks = successful.groupby("user_id")["task_type"].agg(set).to_dict()
            by_user_employers = successful.groupby("user_id")["employer_id"].agg(set).to_dict()
            frame["user_task_type_experience"] = [
                int(task_type in by_user_tasks.get(user_id, set())) for user_id in user_ids
            ]
            frame["user_employer_experience"] = [
                int(employer_id in by_user_employers.get(user_id, set())) for user_id in user_ids
            ]
            self._add_user_preference_features(frame, successful, user_ids)
            self._add_same_day_features(frame, successful, user_ids, shift)

        employer_hist = history[history["employer_id"].astype(str) == employer_id]
        if not employer_hist.empty:
            unique_shifts = max(1, employer_hist["shift_id"].nunique())
            frame["employer_avg_reward"] = float(employer_hist["reward"].mean())
            frame["employer_popularity"] = int(employer_hist["user_id"].nunique())
            system_cancels = employer_hist.loc[
                employer_hist["interaction"] == "SYSTEM_CANCEL", "shift_id"
            ].nunique()
            frame["employer_reliability"] = max(0, min(1, 1 - system_cancels / unique_shifts))

    def _add_user_preference_features(
        self,
        frame: pd.DataFrame,
        successful: pd.DataFrame,
        user_ids: list[str],
    ) -> None:
        data = successful.copy()
        data["is_weekday"] = self._is_weekday(data["start_at"])
        data["time_of_day"] = data["start_at"].dt.hour.map(self._time_of_day)
        prefs = {}
        for user_id, group in data.groupby("user_id"):
            total = len(group)
            prefs[user_id] = (
                float((group["is_weekday"] == 0).sum() / total),
                float((group["time_of_day"] == "day").sum() / total),
                float((group["time_of_day"] == "evening").sum() / total),
                float((group["time_of_day"] == "night").sum() / total),
            )

        frame["user_weekend_ratio"] = [
            prefs.get(user_id, (-1, -1, -1, -1))[0] for user_id in user_ids
        ]
        frame["user_day_ratio"] = [
            prefs.get(user_id, (-1, -1, -1, -1))[1] for user_id in user_ids
        ]

        shift_time = frame["shift_time_of_day"].iloc[0]
        matches = []
        for user_id in user_ids:
            pref = prefs.get(user_id)
            if pref is None:
                matches.append(-1)
                continue
            ratios = {"day": pref[1], "evening": pref[2], "night": pref[3]}
            best_time = max(ratios, key=ratios.get)
            matches.append(int(shift_time == best_time and ratios[best_time] > 0))
        frame["matches_time_of_day"] = matches

    def _add_same_day_features(
        self,
        frame: pd.DataFrame,
        successful: pd.DataFrame,
        user_ids: list[str],
        shift,
    ) -> None:
        shift_date = self._to_naive_utc(shift.start_at).date()
        same_day = successful[successful["start_at"].dt.date == shift_date]
        counts = same_day.groupby("user_id")["shift_id"].nunique().to_dict()
        frame["user_shifts_on_same_day"] = [int(counts.get(user_id, 0)) for user_id in user_ids]

    def _median_reward_per_hour(self) -> float:
        if self.shifts_df.empty:
            return 0.0
        values = self.shifts_df["reward"] / self.shifts_df["hours"].replace(0, np.nan)
        median = values.replace([np.inf, -np.inf], np.nan).dropna().median()
        return float(median) if pd.notna(median) else 0.0

    def _is_weekday(self, dates: pd.Series) -> pd.Series:
        normalized = dates.dt.normalize()
        is_weekend = dates.dt.dayofweek >= 5
        is_holiday = normalized.isin(HOLIDAYS)
        return (~is_weekend & ~is_holiday).astype(int)

    def _time_of_day(self, hour: int) -> str:
        if 6 <= hour < 12:
            return "morning"
        if 12 <= hour < 18:
            return "day"
        if 18 <= hour < 24:
            return "evening"
        return "night"

    def _to_naive_utc(self, value) -> pd.Timestamp:
        ts = pd.Timestamp(value)
        if ts.tzinfo is not None:
            return ts.tz_convert("UTC").tz_localize(None)
        return ts
