from __future__ import annotations

import json
import logging
import pickle
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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
TASK_CATEGORY_MAPPING = {
    "Выкладка товара": "store_warehouse",
    "Сборка заказов": "store_warehouse",
    "Упаковка товаров": "store_warehouse",
    "Фасовка готовой продукции": "store_warehouse",
    "Инвентаризация склада": "store_warehouse",
    "Помощь в торговом зале": "customer_service",
    "Обслуживание покупателей": "customer_service",
    "Помощь в прикассовой зоне": "customer_service",
    "Выдача заказов": "customer_service",
    "Выдача заказов (Wildberries)": "customer_service",
    "Выдача заказов (OZON)": "customer_service",
    "Контроль порядка в торговом зале": "customer_service",
    "Обслуживание посетителей кафе": "customer_service",
    "Обслуживание в гардеробе": "customer_service",
    "Приготовление пищи": "food_kitchen",
    "Помощь на кухне": "food_kitchen",
    "Мойка посуды и инвентаря": "food_kitchen",
    "Выпечка хлеба": "food_kitchen",
    "Уборка помещений": "cleaning",
    "Уборка на улице": "cleaning",
    "Погрузка и разгрузка товара": "loading",
    "Сопровождение товара и разгрузка": "loading",
    "Подсобные работы": "loading",
    "Сбор и расстановка тележек": "loading",
    "Доставка заказов на авто": "delivery_car",
    "Доставка заказов на велосипеде": "delivery_bike",
    "Доставка заказов пешком": "delivery_walk",
    "Сборка букетов": "other",
    "Помощь в парке развлечений": "other",
}


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


def _prepare_history_frame(
    merged_events: pd.DataFrame,
    shifts_for_join: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    df_all = merged_events[
        ["shift_id", "user_id", "interaction", "ts", "start_at"]
    ].copy()
    df_all["ts"] = pd.to_datetime(df_all["ts"], utc=True).dt.tz_localize(None)
    df_all["start_at"] = pd.to_datetime(df_all["start_at"], utc=True).dt.tz_localize(None)

    shifts_info = shifts_for_join[
        ["shift_id", "employer_id", "location_id", "hours", "reward", "task_type"]
    ].copy()
    shifts_info["reward_per_hour"] = shifts_info["reward"] / shifts_info["hours"]
    median_pay_global = shifts_info["reward_per_hour"].median()
    shifts_info = shifts_info.rename(
        columns={
            "location_id": "shift_location_id",
            "task_type": "shift_task_type",
        }
    )

    df_all = df_all.merge(shifts_info, on="shift_id", how="left")
    df_all["task_category"] = df_all["shift_task_type"].map(TASK_CATEGORY_MAPPING)

    pairs_with_cancel = df_all[
        df_all["interaction"].isin(["USER_CANCEL", "SYSTEM_CANCEL"])
    ][["user_id", "shift_id"]].drop_duplicates()
    pairs_with_cancel["has_cancel"] = 1
    df_all = df_all.merge(pairs_with_cancel, on=["user_id", "shift_id"], how="left")
    df_all["has_cancel"] = df_all["has_cancel"].fillna(0)
    df_all["is_successful"] = (
        (df_all["interaction"] == "APPLY") & (df_all["has_cancel"] == 0)
    ).astype(int)
    LOGGER.info("Successful actions: %s", f"{int(df_all['is_successful'].sum()):,}")

    df_all_for_hist = df_all[
        [
            "user_id",
            "shift_id",
            "interaction",
            "ts",
            "is_successful",
            "employer_id",
            "shift_task_type",
            "shift_location_id",
            "reward_per_hour",
            "hours",
        ]
    ].copy()
    df_all_for_hist = df_all_for_hist.rename(columns={"shift_id": "hist_shift_id"})
    return df_all_for_hist, median_pay_global


def _add_historical_features_fast(
    df: pd.DataFrame,
    df_all_for_hist: pd.DataFrame,
    median_pay_global: float,
    window_start_offset: int,
    window_end_offset: int,
) -> pd.DataFrame:
    df = df.copy()
    df["_row_id"] = np.arange(len(df))
    df["win_start"] = df["start_at"] - pd.Timedelta(days=window_start_offset)
    df["win_end"] = df["start_at"] - pd.Timedelta(days=window_end_offset)

    pairs = df[
        ["_row_id", "user_id", "shift_id", "win_start", "win_end", "location_id_shift"]
    ].copy()
    hist = pairs.merge(df_all_for_hist, on="user_id", how="left")
    hist = hist[(hist["ts"] >= hist["win_start"]) & (hist["ts"] < hist["win_end"])]

    result = df[["_row_id"]].copy()
    if len(hist) > 0:
        hist["event_date"] = hist["ts"].dt.date
        active_stats = (
            hist.groupby("_row_id")
            .agg(user_active_days=("event_date", "nunique"))
            .reset_index()
        )
        view_stats = (
            hist[hist["interaction"] == "VIEW"]
            .groupby("_row_id")
            .agg(user_total_views=("hist_shift_id", "nunique"))
            .reset_index()
        )
        succ_hist = hist[hist["is_successful"] == 1].drop_duplicates(
            ["_row_id", "hist_shift_id"]
        )
        succ_stats = (
            succ_hist.groupby("_row_id")
            .agg(
                user_total_successful_actions=("hist_shift_id", "nunique"),
                user_unique_employers=("employer_id", "nunique"),
                user_unique_task_types=("shift_task_type", "nunique"),
                user_avg_reward_per_hour=("reward_per_hour", "mean"),
            )
            .reset_index()
        )

        if len(succ_hist) > 0:
            succ_hist = succ_hist.copy()
            succ_hist["loc_match"] = (
                succ_hist["shift_location_id"] == succ_hist["location_id_shift"]
            ).astype(int)
            succ_hist["high_paid"] = (
                succ_hist["reward_per_hour"] > median_pay_global
            ).astype(int)
            succ_hist["long_shift"] = (succ_hist["hours"] > 8).astype(int)
            succ_hist["short_shift"] = (succ_hist["hours"] < 4).astype(int)
            ratio_stats = (
                succ_hist.groupby("_row_id")
                .agg(
                    user_location_experience=("loc_match", "sum"),
                    user_high_paid_ratio=("high_paid", "mean"),
                    user_long_shift_ratio=("long_shift", "mean"),
                    user_short_shift_ratio=("short_shift", "mean"),
                )
                .reset_index()
            )
        else:
            ratio_stats = pd.DataFrame(
                columns=[
                    "_row_id",
                    "user_location_experience",
                    "user_high_paid_ratio",
                    "user_long_shift_ratio",
                    "user_short_shift_ratio",
                ]
            )

        result = result.merge(active_stats, on="_row_id", how="left")
        result = result.merge(view_stats, on="_row_id", how="left")
        result = result.merge(succ_stats, on="_row_id", how="left")
        result = result.merge(ratio_stats, on="_row_id", how="left")
    else:
        result["user_active_days"] = np.nan
        result["user_total_views"] = np.nan
        result["user_total_successful_actions"] = np.nan

    result["user_total_views"] = result["user_total_views"].fillna(0).astype(int)
    result["user_total_successful_actions"] = (
        result["user_total_successful_actions"].fillna(0).astype(int)
    )
    result["user_active_days"] = result["user_active_days"].fillna(0).astype(int)

    has_history = result["user_active_days"] > 0
    result["user_success_rate"] = np.where(
        has_history,
        np.where(
            result["user_total_views"] > 0,
            result["user_total_successful_actions"] / result["user_total_views"],
            0,
        ),
        -1,
    )

    success_hist_features = [
        "user_unique_employers",
        "user_unique_task_types",
        "user_location_experience",
        "user_high_paid_ratio",
        "user_long_shift_ratio",
        "user_short_shift_ratio",
        "user_avg_reward_per_hour",
    ]
    for col in success_hist_features:
        if col not in result.columns:
            result[col] = -1
        else:
            result[col] = result[col].fillna(-1)

    result = result.drop(columns=["_row_id"])
    df = df.drop(columns=["win_start", "win_end"])
    df = pd.concat([df.reset_index(drop=True), result.reset_index(drop=True)], axis=1)
    return df.drop(columns=["_row_id"])


def _add_historical_features_batched(
    df: pd.DataFrame,
    df_all_for_hist: pd.DataFrame,
    median_pay_global: float,
    dataset_name: str,
    batch_size: int = 5000,
) -> pd.DataFrame:
    window_end_offset = 2
    window_start_offset = 28 + window_end_offset
    parts = []
    for start_idx in range(0, len(df), batch_size):
        end_idx = start_idx + batch_size
        batch = df.iloc[start_idx:end_idx].copy()
        batch = _add_historical_features_fast(
            batch,
            df_all_for_hist,
            median_pay_global,
            window_start_offset,
            window_end_offset,
        )
        parts.append(batch)

    result = pd.concat(parts, axis=0, ignore_index=True)
    LOGGER.info("%s: historical features added, shape=%s", dataset_name, result.shape)
    return result


def _count_success_batch(
    ts_lists: list[list[pd.Timestamp]],
    win_starts: pd.Series,
    win_ends: pd.Series,
) -> list[int]:
    results = []
    for ts_list, w_start, w_end in zip(ts_lists, win_starts, win_ends):
        if not ts_list:
            results.append(0)
            continue
        ts_arr = np.asarray(ts_list, dtype="datetime64[ns]")
        left = np.searchsorted(ts_arr, np.datetime64(pd.Timestamp(w_start)), side="left")
        right = np.searchsorted(ts_arr, np.datetime64(pd.Timestamp(w_end)), side="left")
        results.append(right - left)
    return results


def _count_applies_before_batch(
    shift_ids: np.ndarray,
    pred_moments: pd.Series,
    applies_dict: dict[str, list[pd.Timestamp]],
) -> list[int]:
    results = []
    for shift_id, pred_moment in zip(shift_ids, pred_moments):
        ts_list = applies_dict.get(shift_id, [])
        if not ts_list:
            results.append(0)
            continue
        ts_arr = np.asarray(ts_list, dtype="datetime64[ns]")
        results.append(
            int(np.searchsorted(ts_arr, np.datetime64(pd.Timestamp(pred_moment)), side="left"))
        )
    return results


def _get_match_vectorized(value: np.ndarray, ratio: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    result = np.where(ratio == -1, -1, 0)
    return np.where((value == 1) & (ratio > threshold), 1, result)


def _add_groups_5_to_14(
    frame: pd.DataFrame,
    merged_events: pd.DataFrame,
    shifts_for_join: pd.DataFrame,
) -> pd.DataFrame:
    window_end_offset = 2
    LOGGER.info("Adding feature groups 5-14")

    shift_start_map = dict(zip(shifts_for_join["shift_id"], shifts_for_join["start_at"]))
    success_by_user: dict[str, list[pd.Timestamp]] = {}
    successful_raw: dict[str, set[str]] = {}
    user_employers_successful: dict[str, set[str]] = {}
    employer_data: dict[str, dict[str, object]] = {}
    shift_applies_views: dict[str, list[int]] = {}
    applies_dict: dict[str, list[pd.Timestamp]] = {}
    user_shift_data: dict[tuple[str, str], list[int]] = {}
    user_day_shifts: dict[tuple[str, object], list[tuple[int, float, str]]] = {}
    user_successful: dict[str, list[dict[str, object]]] = {}
    holidays_set = set(HOLIDAYS.normalize())

    shifts_info = shifts_for_join[
        ["shift_id", "employer_id", "location_id", "hours", "reward", "task_type"]
    ].copy()
    shifts_info["reward_per_hour"] = shifts_info["reward"] / shifts_info["hours"]
    shifts_info = shifts_info.rename(
        columns={
            "location_id": "shift_location_id",
            "task_type": "shift_task_type",
        }
    )
    df_all = merged_events[["shift_id", "user_id", "interaction", "ts", "start_at"]].copy()
    df_all["ts"] = pd.to_datetime(df_all["ts"], utc=True).dt.tz_localize(None)
    df_all["start_at"] = pd.to_datetime(df_all["start_at"], utc=True).dt.tz_localize(None)
    df_all = df_all.merge(shifts_info, on="shift_id", how="left")
    df_all["task_category"] = df_all["shift_task_type"].map(TASK_CATEGORY_MAPPING)

    pairs_with_cancel = df_all[
        df_all["interaction"].isin(["USER_CANCEL", "SYSTEM_CANCEL"])
    ][["user_id", "shift_id"]].drop_duplicates()
    pairs_with_cancel["has_cancel"] = 1
    df_all = df_all.merge(pairs_with_cancel, on=["user_id", "shift_id"], how="left")
    df_all["has_cancel"] = df_all["has_cancel"].fillna(0)
    df_all["is_successful"] = (
        (df_all["interaction"] == "APPLY") & (df_all["has_cancel"] == 0)
    ).astype(int)

    for row in df_all.itertuples(index=False):
        shift_id = row.shift_id
        shift_start = shift_start_map.get(shift_id)
        if shift_start is None:
            continue

        pred_moment = pd.Timestamp(shift_start).tz_localize(None) - pd.Timedelta(days=2)
        ts = pd.Timestamp(row.ts)
        user_id = row.user_id
        interaction = row.interaction
        employer_id = row.employer_id

        if ts < pred_moment:
            if interaction == "APPLY":
                shift_applies_views.setdefault(shift_id, [0, 0])[0] += 1
                applies_dict.setdefault(shift_id, []).append(ts)
            elif interaction == "VIEW":
                shift_applies_views.setdefault(shift_id, [0, 0])[1] += 1

            employer_entry = employer_data.setdefault(
                employer_id,
                {"shifts": set(), "rewards": [], "users": set(), "system_cancels": set()},
            )
            employer_entry["shifts"].add(shift_id)
            employer_entry["users"].add(user_id)
            if pd.notna(row.reward):
                employer_entry["rewards"].append(row.reward)
            if interaction == "SYSTEM_CANCEL":
                employer_entry["system_cancels"].add(shift_id)

            if interaction == "VIEW":
                user_shift_data.setdefault((user_id, shift_id), [0, 0])[0] += 1

        if row.is_successful == 1 and ts < pred_moment:
            success_by_user.setdefault(user_id, []).append(ts)
            successful_raw.setdefault(user_id, set()).add(row.shift_task_type)
            user_employers_successful.setdefault(user_id, set()).add(employer_id)
            user_shift_data.setdefault((user_id, shift_id), [0, 0])[1] = 1

            successful_shift_start = pd.Timestamp(row.start_at)
            hour = successful_shift_start.hour
            is_weekday = (
                0
                if successful_shift_start.normalize() in holidays_set
                or successful_shift_start.dayofweek >= 5
                else 1
            )
            time_of_day = (
                "morning"
                if 6 <= hour < 12
                else "day"
                if 12 <= hour < 18
                else "evening"
                if 18 <= hour < 24
                else "night"
            )
            user_successful.setdefault(user_id, []).append(
                {"is_weekday": is_weekday, "time_of_day": time_of_day}
            )
            user_day_shifts.setdefault((user_id, successful_shift_start.date()), []).append(
                (hour, hour + row.hours, shift_id)
            )

    for user_id in success_by_user:
        success_by_user[user_id] = sorted(success_by_user[user_id])
    for shift_id in applies_dict:
        applies_dict[shift_id] = sorted(applies_dict[shift_id])

    employer_stats_map = {}
    for employer_id, data in employer_data.items():
        total_shifts = len(data["shifts"])
        rewards = data["rewards"]
        avg_reward = np.mean(rewards) if rewards else 0
        popularity = len(data["users"])
        system_cancels = len(data["system_cancels"])
        reliability = (
            max(0, min(1, 1 - (system_cancels / total_shifts))) if total_shifts > 0 else 1
        )
        employer_stats_map[employer_id] = (avg_reward, popularity, reliability)

    user_prefs_map = {}
    for user_id, data in user_successful.items():
        total = len(data)
        weekend_cnt = sum(1 for item in data if item["is_weekday"] == 0)
        day_cnt = sum(1 for item in data if item["time_of_day"] == "day")
        evening_cnt = sum(1 for item in data if item["time_of_day"] == "evening")
        night_cnt = sum(1 for item in data if item["time_of_day"] == "night")
        user_prefs_map[user_id] = (
            weekend_cnt / total,
            day_cnt / total,
            evening_cnt / total,
            night_cnt / total,
        )

    frame = frame.copy()
    for window_days in [7, 14]:
        win_start = frame["start_at"] - pd.Timedelta(days=window_days)
        win_end = frame["start_at"] - pd.Timedelta(days=window_end_offset)
        ts_lists = [success_by_user.get(user_id, []) for user_id in frame["user_id"]]
        frame[f"user_successful_actions_last_{window_days}d"] = _count_success_batch(
            ts_lists, win_start, win_end
        )

    frame["user_task_type_experience"] = [
        1 if task_type in successful_raw.get(user_id, set()) else 0
        for user_id, task_type in zip(frame["user_id"], frame["task_type"])
    ]
    frame["user_employer_experience"] = [
        1 if employer_id in user_employers_successful.get(user_id, set()) else 0
        for user_id, employer_id in zip(frame["user_id"], frame["employer_id"])
    ]

    employer_stats = [employer_stats_map.get(eid, (0, 0, 1)) for eid in frame["employer_id"]]
    frame["employer_avg_reward"] = [stats[0] for stats in employer_stats]
    frame["employer_popularity"] = [stats[1] for stats in employer_stats]
    frame["employer_reliability"] = [stats[2] for stats in employer_stats]

    frame["shift_total_views"] = [
        shift_applies_views.get(sid, [0, 0])[1] for sid in frame["shift_id"]
    ]
    pred_moments = frame["start_at"] - pd.Timedelta(days=2)
    applies_before = _count_applies_before_batch(
        frame["shift_id"].values, pred_moments, applies_dict
    )
    frame["free_spots_at_prediction"] = (
        frame["capacity"] - np.array(applies_before)
    ).clip(lower=0)
    frame["is_full_at_prediction"] = (frame["free_spots_at_prediction"] == 0).astype(int)

    keys = list(zip(frame["user_id"].values, frame["shift_id"].values))
    frame["view_count_for_shift"] = [user_shift_data.get(key, [0, 0])[0] for key in keys]
    frame["has_apply_before"] = [user_shift_data.get(key, [0, 0])[1] for key in keys]

    prefs = [user_prefs_map.get(user_id, (-1, -1, -1, -1)) for user_id in frame["user_id"]]
    frame["user_weekend_ratio"] = [pref[0] for pref in prefs]
    frame["user_day_ratio"] = [pref[1] for pref in prefs]
    frame["user_evening_ratio"] = [pref[2] for pref in prefs]
    frame["user_night_ratio"] = [pref[3] for pref in prefs]

    frame["matches_high_paid"] = _get_match_vectorized(
        frame["is_high_paid_window"].values, frame["user_high_paid_ratio"].values
    )
    frame["matches_long_shift"] = _get_match_vectorized(
        frame["is_long_shift"].values, frame["user_long_shift_ratio"].values
    )
    frame["matches_short_shift"] = _get_match_vectorized(
        frame["is_short_shift"].values, frame["user_short_shift_ratio"].values
    )

    no_data = (
        (frame["user_day_ratio"] == -1)
        & (frame["user_evening_ratio"] == -1)
        & (frame["user_night_ratio"] == -1)
    )
    ratios = frame[["user_day_ratio", "user_evening_ratio", "user_night_ratio"]].values
    times = np.array(["day", "evening", "night"])
    preferred = times[np.argmax(ratios, axis=1)]
    match = (frame["shift_time_of_day"].values == preferred).astype(int)
    max_ratio = np.max(ratios, axis=1)
    frame["matches_time_of_day"] = np.where(no_data | (max_ratio == 0), -1, match)
    frame["matches_location"] = (frame["user_location_experience"] > 0).astype(int)

    same_day_counts = []
    for row in frame[
        ["user_id", "shift_id", "start_at", "hours", "has_apply_before"]
    ].itertuples(index=False):
        key = (row.user_id, row.start_at.date())
        valid = [
            (start_hour, end_hour)
            for start_hour, end_hour, shift_id in user_day_shifts.get(key, [])
            if shift_id != row.shift_id
        ]
        same_day_counts.append(len(valid))
    same_day_counts_arr = np.array(same_day_counts)
    same_day_counts_arr[frame["has_apply_before"].values == 1] = -1
    frame["user_shifts_on_same_day"] = same_day_counts_arr

    frame["is_full_and_has_apply"] = (
        (frame["is_full_at_prediction"] == 1) & (frame["has_apply_before"] == 1)
    ).astype(int)

    added_columns = [
        "user_successful_actions_last_7d",
        "user_successful_actions_last_14d",
        "user_task_type_experience",
        "user_employer_experience",
        "employer_avg_reward",
        "employer_popularity",
        "employer_reliability",
        "shift_total_views",
        "free_spots_at_prediction",
        "is_full_at_prediction",
        "view_count_for_shift",
        "has_apply_before",
        "user_weekend_ratio",
        "user_day_ratio",
        "user_evening_ratio",
        "user_night_ratio",
        "matches_high_paid",
        "matches_long_shift",
        "matches_short_shift",
        "matches_time_of_day",
        "matches_location",
        "user_shifts_on_same_day",
        "is_full_and_has_apply",
    ]
    LOGGER.info("Added feature groups 5-14 columns: %s", added_columns)
    return frame


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
    unique_pairs = (
        merged_events.groupby(["user_id", "shift_id"]).size().reset_index(name="count")
    )
    duplicated_pairs = unique_pairs[unique_pairs["count"] > 1]
    LOGGER.info("Total rows in merged_events: %s", len(merged_events))
    LOGGER.info("Unique pairs (user_id, shift_id): %s", len(unique_pairs))
    LOGGER.info(
        "Records per pair statistics:\n%s",
        unique_pairs["count"].describe().to_string(),
    )
    LOGGER.info("Pairs with multiple records: %s", len(duplicated_pairs))
    LOGGER.info(
        "Share of pairs with multiple records: %.2f%%",
        len(duplicated_pairs) / len(unique_pairs) * 100 if len(unique_pairs) else 0,
    )
    duplicated_examples = merged_events[
        merged_events.groupby(["user_id", "shift_id"])["user_id"].transform("size") > 1
    ]
    LOGGER.info(
        "Duplicated pair examples (first 10):\n%s",
        duplicated_examples[["user_id", "shift_id", "interaction", "ts"]]
        .head(10)
        .to_string(index=False),
    )

    sequences = []
    for (_user, _shift), group in duplicated_examples.groupby(["user_id", "shift_id"]):
        seq = tuple(group.sort_values("ts")["interaction"].values)
        sequences.append(seq)

    sequence_counts = Counter(sequences)
    LOGGER.info("Top-5 event sequences in duplicated pairs:")
    for seq, count in sequence_counts.most_common(5):
        LOGGER.info("    %s: %s times", seq, count)

    LOGGER.info("Analysis of time between VIEW and START_AT")
    view_events = merged_events[merged_events["interaction"] == "VIEW"].copy()
    view_events["start_at"] = pd.to_datetime(
        view_events["start_at"], utc=True
    ).dt.tz_localize(None)
    view_events["ts"] = pd.to_datetime(view_events["ts"], utc=True).dt.tz_localize(None)
    view_events["days_between"] = (view_events["start_at"] - view_events["ts"]).dt.days
    view_events_valid = view_events[view_events["days_between"] >= 0]

    LOGGER.info("Total VIEW events: %s", f"{len(view_events):,}")
    LOGGER.info(
        "VIEW events with correct date (VIEW before shift): %s",
        f"{len(view_events_valid):,}",
    )
    LOGGER.info("Mean: %.2f days", view_events_valid["days_between"].mean())
    LOGGER.info("Median: %.2f days", view_events_valid["days_between"].median())
    view_mode = view_events_valid["days_between"].mode()
    LOGGER.info(
        "Mode: %s days",
        view_mode.iloc[0] if len(view_mode) > 0 else "N/A",
    )
    LOGGER.info("Std: %.2f", view_events_valid["days_between"].std())
    LOGGER.info("Min: %s days", view_events_valid["days_between"].min())
    LOGGER.info("Max: %s days", view_events_valid["days_between"].max())
    LOGGER.info("Quantiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        quantile_val = view_events_valid["days_between"].quantile(p / 100)
        LOGGER.info("%s%%: %.2f days", p, quantile_val)

    LOGGER.info("Analysis of long-term views by task type")
    long_term_days = 15
    if "task_type" not in view_events_valid.columns:
        view_events_valid = view_events_valid.merge(
            shifts_for_join[["shift_id", "task_type"]],
            on="shift_id",
            how="left",
        )
        LOGGER.info("Added task_type information")
    LOGGER.info("Columns: %s", view_events_valid.columns.tolist())
    LOGGER.info("Has task_type: %s", "task_type" in view_events_valid.columns)

    view_events_valid["view_type"] = view_events_valid["days_between"].apply(
        lambda x: "long_term" if x >= long_term_days else "short_term"
    )
    task_stats = view_events_valid.groupby("task_type").agg(
        total_views=("interaction", "count"),
        long_term_views=("view_type", lambda x: (x == "long_term").sum()),
        short_term_views=("view_type", lambda x: (x == "short_term").sum()),
        avg_days=("days_between", "mean"),
        median_days=("days_between", "median"),
    ).reset_index()
    task_stats["long_term_pct"] = (
        task_stats["long_term_views"] / task_stats["total_views"] * 100
    )
    task_stats_sorted = task_stats.sort_values("long_term_pct", ascending=False)

    LOGGER.info(
        "Top-10 task types by long-term view share (>=%s days):",
        long_term_days,
    )
    for _, row in task_stats_sorted.head(10).iterrows():
        LOGGER.info(
            "%s: total_views=%s long_term_views=%s (%.2f%%) avg_days=%.1f median_days=%.0f",
            row["task_type"],
            f"{int(row['total_views']):,}",
            f"{int(row['long_term_views']):,}",
            row["long_term_pct"],
            row["avg_days"],
            row["median_days"],
        )

    LOGGER.info("Bottom-10 task types by long-term view share:")
    for _, row in task_stats_sorted.tail(10).iterrows():
        LOGGER.info(
            "%s: total_views=%s long_term_views=%s (%.2f%%) avg_days=%.1f median_days=%.0f",
            row["task_type"],
            f"{int(row['total_views']):,}",
            f"{int(row['long_term_views']):,}",
            row["long_term_pct"],
            row["avg_days"],
            row["median_days"],
        )

    finished_without_apply = merged_events[
        (merged_events["interaction"] == "FINISHED")
        & (
            ~merged_events.duplicated(
                subset=["user_id", "shift_id", "interaction"], keep=False
            )
        )
    ]
    pairs_with_finished = merged_events[merged_events["interaction"] == "FINISHED"][
        ["user_id", "shift_id"]
    ].drop_duplicates()
    pairs_with_apply = merged_events[merged_events["interaction"] == "APPLY"][
        ["user_id", "shift_id"]
    ].drop_duplicates()
    finished_no_apply = pairs_with_finished.merge(
        pairs_with_apply,
        on=["user_id", "shift_id"],
        how="left",
        indicator=True,
    )
    finished_no_apply = finished_no_apply[finished_no_apply["_merge"] == "left_only"]
    LOGGER.info("Pairs with FINISHED: %s", len(pairs_with_finished))
    LOGGER.info("Pairs with APPLY: %s", len(pairs_with_apply))
    LOGGER.info("Pairs with FINISHED but without APPLY: %s", len(finished_no_apply))

    pairs_with_both = (
        merged_events[merged_events["interaction"].isin(["USER_CANCEL", "APPLY"])]
        .groupby(["user_id", "shift_id"])
        .filter(lambda x: len(x["interaction"].unique()) == 2)
    )
    cancel_apply_pairs = []

    for (user, shift), group in pairs_with_both.groupby(["user_id", "shift_id"]):
        cancel_dates = group[group["interaction"] == "USER_CANCEL"]["ts"].values
        apply_dates = group[group["interaction"] == "APPLY"]["ts"].values

        for cancel_date in cancel_dates:
            for apply_date in apply_dates:
                if apply_date > cancel_date:
                    days_diff = (apply_date - cancel_date) / np.timedelta64(1, "D")
                    cancel_apply_pairs.append(
                        {
                            "user_id": user,
                            "shift_id": shift,
                            "cancel_date": cancel_date,
                            "apply_date": apply_date,
                            "days_after": int(days_diff),
                        }
                    )

    df_cancel_then_apply = pd.DataFrame(cancel_apply_pairs)
    LOGGER.info(
        "Pairs with APPLY after USER_CANCEL (different days): %s",
        len(df_cancel_then_apply),
    )
    if len(df_cancel_then_apply) > 0:
        LOGGER.info(
            "Cancel then apply examples:\n%s",
            df_cancel_then_apply.head(10).to_string(index=False),
        )
        LOGGER.info(
            "Days between CANCEL and APPLY distribution:\n%s",
            df_cancel_then_apply["days_after"].value_counts().sort_index().to_string(),
        )
    else:
        LOGGER.info("No pairs with APPLY after USER_CANCEL on different days")

    same_day_pairs = []
    for (user, shift), group in pairs_with_both.groupby(["user_id", "shift_id"]):
        cancel_dates = set(group[group["interaction"] == "USER_CANCEL"]["ts"].values)
        apply_dates = set(group[group["interaction"] == "APPLY"]["ts"].values)

        if cancel_dates & apply_dates:
            same_day_pairs.append((user, shift))

    LOGGER.info("Found pairs (with APPLY and USER_CANCEL): %s", len(same_day_pairs))

    same_day_shifts = list({shift for (_user, shift) in same_day_pairs})
    LOGGER.info("Total same-day shifts: %s", len(same_day_shifts))

    cancel_then_apply_count = 0
    apply_then_cancel_count = 0
    cancel_then_apply_with_finished = 0
    cancel_then_apply_without_finished = 0

    for shift_id in same_day_shifts:
        group = merged_events[merged_events["shift_id"] == shift_id].sort_index()

        for user_id in group["user_id"].unique():
            user_group = group[group["user_id"] == user_id]
            interactions = user_group["interaction"].tolist()

            if "APPLY" in interactions and "USER_CANCEL" in interactions:
                last_apply = max(i for i, e in enumerate(interactions) if e == "APPLY")
                last_cancel = max(
                    i for i, e in enumerate(interactions) if e == "USER_CANCEL"
                )
                has_finished = "FINISHED" in interactions

                if last_apply > last_cancel:
                    cancel_then_apply_count += 1
                    if has_finished:
                        cancel_then_apply_with_finished += 1
                    else:
                        cancel_then_apply_without_finished += 1
                else:
                    apply_then_cancel_count += 1

    LOGGER.info("Quick analysis results:")
    LOGGER.info("    CANCEL_then_APPLY (total): %s", cancel_then_apply_count)
    LOGGER.info("        with FINISHED: %s", cancel_then_apply_with_finished)
    LOGGER.info("        without FINISHED: %s", cancel_then_apply_without_finished)
    LOGGER.info("    APPLY_then_CANCEL: %s", apply_then_cancel_count)

    target_agg = merged_events.groupby(["user_id", "shift_id"]).agg(
        has_apply=("interaction", lambda x: "APPLY" in x.values),
        has_finished=("interaction", lambda x: "FINISHED" in x.values),
        has_cancel=("interaction", lambda x: "USER_CANCEL" in x.values),
        has_system_cancel=("interaction", lambda x: "SYSTEM_CANCEL" in x.values),
    ).reset_index()
    LOGGER.info("Aggregated pairs: %s", f"{len(target_agg):,}")

    def set_target(row: pd.Series) -> int | None:
        if row["has_system_cancel"]:
            return None
        if row["has_cancel"]:
            return 0
        if row["has_apply"] or row["has_finished"]:
            return 1
        return 0

    target_agg["target"] = target_agg.apply(set_target, axis=1)
    target_correct = target_agg.dropna(subset=["target"]).copy()
    target_correct["target"] = target_correct["target"].astype(int)
    positive_targets = int(target_correct["target"].sum())
    LOGGER.info("Target results:")
    LOGGER.info("    Total pairs: %s", f"{len(target_correct):,}")
    LOGGER.info(
        "    target=1: %s (%.4f%%)",
        f"{positive_targets:,}",
        target_correct["target"].mean() * 100 if len(target_correct) else 0,
    )
    LOGGER.info(
        "    target=0: %s",
        f"{len(target_correct) - positive_targets:,}",
    )

    pairs_user_cancel = merged_events[merged_events["interaction"] == "USER_CANCEL"][
        ["user_id", "shift_id"]
    ].drop_duplicates()
    LOGGER.info("Pairs with USER_CANCEL: %s", f"{len(pairs_user_cancel):,}")

    pairs_system_cancel = merged_events[merged_events["interaction"] == "SYSTEM_CANCEL"][
        ["user_id", "shift_id"]
    ].drop_duplicates()
    LOGGER.info("Pairs with SYSTEM_CANCEL: %s", f"{len(pairs_system_cancel):,}")

    all_cancel_pairs = pd.concat([pairs_user_cancel, pairs_system_cancel]).drop_duplicates()
    LOGGER.info("Pairs with any cancel (USER or SYSTEM): %s", f"{len(all_cancel_pairs):,}")

    both_cancel = pairs_user_cancel.merge(
        pairs_system_cancel, on=["user_id", "shift_id"], how="inner"
    )
    LOGGER.info("Pairs with both cancels: %s", f"{len(both_cancel):,}")

    target_true = target_correct[target_correct["target"] == 1][["user_id", "shift_id"]]
    target_with_user_cancel = target_true.merge(
        pairs_user_cancel, on=["user_id", "shift_id"], how="inner"
    )
    target_with_system_cancel = target_true.merge(
        pairs_system_cancel, on=["user_id", "shift_id"], how="inner"
    )
    target_with_any_cancel = target_true.merge(
        all_cancel_pairs, on=["user_id", "shift_id"], how="inner"
    )

    LOGGER.info(
        "target=1, but has USER_CANCEL: %s (should be 0)",
        f"{len(target_with_user_cancel):,}",
    )
    LOGGER.info(
        "target=1, but has SYSTEM_CANCEL: %s (should be 0)",
        f"{len(target_with_system_cancel):,}",
    )
    LOGGER.info(
        "target=1, but has any cancel: %s (should be 0)",
        f"{len(target_with_any_cancel):,}",
    )

    LOGGER.info("Unique shift statistics:")
    shifts_with_user_cancel = merged_events[merged_events["interaction"] == "USER_CANCEL"][
        "shift_id"
    ].nunique()
    shifts_with_system_cancel = merged_events[
        merged_events["interaction"] == "SYSTEM_CANCEL"
    ]["shift_id"].nunique()
    shifts_with_any_cancel = len(
        set(
            merged_events[merged_events["interaction"] == "USER_CANCEL"][
                "shift_id"
            ].unique()
        )
        | set(
            merged_events[merged_events["interaction"] == "SYSTEM_CANCEL"][
                "shift_id"
            ].unique()
        )
    )

    LOGGER.info(
        "Shifts with USER_CANCEL (at least one user): %s",
        f"{shifts_with_user_cancel:,}",
    )
    LOGGER.info("Shifts with SYSTEM_CANCEL: %s", f"{shifts_with_system_cancel:,}")
    LOGGER.info("Shifts with any cancel: %s", f"{shifts_with_any_cancel:,}")

    LOGGER.info("Target variable logic check:")
    check_df = merged_events.groupby(["user_id", "shift_id"]).agg(
        has_apply=("interaction", lambda x: "APPLY" in x.values),
        has_finished=("interaction", lambda x: "FINISHED" in x.values),
        has_user_cancel=("interaction", lambda x: "USER_CANCEL" in x.values),
        has_system_cancel=("interaction", lambda x: "SYSTEM_CANCEL" in x.values),
    ).reset_index()

    apply_and_user_cancel = check_df[
        (check_df["has_apply"] == True) & (check_df["has_user_cancel"] == True)
    ]
    LOGGER.info("Pairs with APPLY and USER_CANCEL: %s", f"{len(apply_and_user_cancel):,}")

    apply_only_no_finished = check_df[
        ((check_df["has_apply"] == True) | (check_df["has_finished"] == True))
        & (check_df["has_user_cancel"] == False)
        & (check_df["has_system_cancel"] == False)
    ]
    LOGGER.info(
        "Pairs with APPLY, but without USER_CANCEL: %s",
        f"{len(apply_only_no_finished):,}",
    )

    test1 = target_correct[target_correct["shift_id"] == "49938"]
    if len(test1) > 0:
        LOGGER.info(
            "Shift 49938:\n%s",
            test1[
                [
                    "user_id",
                    "shift_id",
                    "has_apply",
                    "has_finished",
                    "has_cancel",
                    "target",
                ]
            ].to_string(index=False),
        )
        LOGGER.info("    target = %s (should be 0)", test1["target"].iloc[0])

    correct_user = "d0b587858e4eba68a2d5b71dd7cc91b3"
    test2 = target_correct[
        (target_correct["shift_id"] == "64679")
        & (target_correct["user_id"] == correct_user)
    ]
    if len(test2) > 0:
        LOGGER.info(
            "Shift 64679, user with FINISHED:\n%s",
            test2[
                [
                    "user_id",
                    "shift_id",
                    "has_apply",
                    "has_finished",
                    "has_cancel",
                    "target",
                ]
            ].to_string(index=False),
        )
        LOGGER.info("    target = %s (should be 1)", test2["target"].iloc[0])
    else:
        LOGGER.info(
            "Shift 64679, user %s... not found in target_correct",
            correct_user[:30],
        )

    system_user = "7cdf53fb7eec3353e93cbf41a2f0e3c6"
    test3 = target_correct[
        (target_correct["shift_id"] == "64679")
        & (target_correct["user_id"] == system_user)
    ]
    LOGGER.info(
        "User with SYSTEM_CANCEL: present in target? %s (should be excluded)",
        len(test3) > 0,
    )

    LOGGER.info("Final correctness check:")
    invalid_cancel = target_correct[
        target_correct["has_cancel"] & (target_correct["target"] == 1)
    ]
    LOGGER.info(
        "Pairs with USER_CANCEL and target=1: %s (should be 0)",
        len(invalid_cancel),
    )

    finished_no_cancel = target_correct[
        target_correct["has_finished"] & ~target_correct["has_cancel"]
    ]
    finished_no_cancel_positive = int(finished_no_cancel["target"].sum())
    LOGGER.info("Pairs with FINISHED without CANCEL: %s", len(finished_no_cancel))
    LOGGER.info("    target=1: %s", finished_no_cancel_positive)
    LOGGER.info(
        "    target=0: %s (should be 0)",
        len(finished_no_cancel) - finished_no_cancel_positive,
    )

    apply_no_cancel_no_finished = target_correct[
        target_correct["has_apply"]
        & ~target_correct["has_cancel"]
        & ~target_correct["has_finished"]
    ]
    apply_no_cancel_no_finished_positive = int(apply_no_cancel_no_finished["target"].sum())
    LOGGER.info(
        "Pairs with APPLY without CANCEL and without FINISHED: %s",
        len(apply_no_cancel_no_finished),
    )
    LOGGER.info("    target=1: %s", apply_no_cancel_no_finished_positive)
    LOGGER.info(
        "    target=0: %s (should be 0)",
        len(apply_no_cancel_no_finished) - apply_no_cancel_no_finished_positive,
    )

    target_correct[["user_id", "shift_id", "target"]].to_csv(
        "target_final.csv", index=False
    )

    merged_with_target = merged_events.merge(
        target_correct[["user_id", "shift_id", "target"]],
        on=["user_id", "shift_id"],
        how="left",
    )
    LOGGER.info(
        "merged_with_target table: %s records",
        f"{len(merged_with_target):,}",
    )

    grouped = merged_events.groupby(["user_id", "shift_id"], as_index=False).agg(
        first_ts=("ts", "min"),
        view_cnt=("interaction", lambda s: int((s == "VIEW").sum())),
        apply_cnt=("interaction", lambda s: int((s == "APPLY").sum())),
        finished_cnt=("interaction", lambda s: int((s == "FINISHED").sum())),
        user_cancel_cnt=("interaction", lambda s: int((s == "USER_CANCEL").sum())),
        system_cancel_cnt=("interaction", lambda s: int((s == "SYSTEM_CANCEL").sum())),
    )
    grouped = grouped.merge(
        target_correct[["user_id", "shift_id", "target"]],
        on=["user_id", "shift_id"],
        how="inner",
    )

    user_totals = grouped.groupby("user_id", as_index=False).agg(
        user_total_views=("view_cnt", "sum"),
        user_total_applies=("apply_cnt", "sum"),
        user_total_finished=("finished_cnt", "sum"),
    )
    grouped = grouped.merge(user_totals, on="user_id", how="left")
    grouped["user_hist_views"] = grouped["user_total_views"] - grouped["view_cnt"]
    grouped["user_hist_applies"] = grouped["user_total_applies"] - grouped["apply_cnt"]
    grouped["user_hist_finished"] = grouped["user_total_finished"] - grouped["finished_cnt"]
    grouped = grouped.drop(
        columns=["user_total_views", "user_total_applies", "user_total_finished"]
    )

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
    base["task_category"] = base["task_type"].map(TASK_CATEGORY_MAPPING)
    LOGGER.info("Added task_category feature")

    df_all_for_hist, median_pay_global = _prepare_history_frame(
        merged_events, shifts_for_join
    )
    cols_before = set(base.columns)
    base["start_at"] = pd.to_datetime(base["start_at"], utc=True).dt.tz_localize(None)
    base = _add_historical_features_batched(
        base,
        df_all_for_hist,
        median_pay_global,
        dataset_name="all",
        batch_size=5000,
    )
    new_cols = set(base.columns) - cols_before
    LOGGER.info("Added %s historical features: %s", len(new_cols), sorted(new_cols))
    base = _add_shift_features(base)
    base = _add_groups_5_to_14(base, merged_events, shifts_for_join)
    return base


def _add_window_features_vectorized(
    df: pd.DataFrame,
    shifts_history: pd.DataFrame,
    window_start_offset: int,
    window_end_offset: int,
) -> pd.DataFrame:
    df = df.copy()

    daily_median = (
        shifts_history.groupby(shifts_history["start_at"].dt.normalize())["reward_per_hour"]
        .median()
        .reset_index()
    )
    daily_median.columns = ["date", "daily_median_pay"]
    daily_median = daily_median.sort_values("date")

    date_range = pd.date_range(
        start=daily_median["date"].min() - pd.Timedelta(days=window_start_offset),
        end=daily_median["date"].max(),
        freq="D",
    )
    daily_median_full = (
        daily_median.set_index("date").reindex(date_range, method="ffill").reset_index()
    )
    daily_median_full.columns = ["date", "daily_median_pay"]
    daily_median_full["rolling_median"] = daily_median_full["daily_median_pay"].rolling(
        window=window_start_offset - window_end_offset,
        min_periods=1,
    ).median()
    daily_median_full["median_pay_window"] = daily_median_full["rolling_median"].shift(
        window_end_offset
    )

    df["shift_date"] = df["start_at"].dt.normalize()
    df = df.merge(
        daily_median_full[["date", "median_pay_window"]],
        left_on="shift_date",
        right_on="date",
        how="left",
    )
    df["median_pay_window"] = df["median_pay_window"].fillna(
        shifts_history["reward_per_hour"].median()
    )
    df["is_high_paid_window"] = (
        df["reward_per_hour"] > df["median_pay_window"]
    ).astype(int)
    return df.drop(columns=["shift_date", "date"])


def _add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_long_shift"] = (df["hours"] > 8).astype(int)
    df["is_short_shift"] = (df["hours"] < 4).astype(int)
    df["mk_match"] = (df["has_mk"] == df["need_mk"]).astype(int)
    df["mk_match_2"] = np.where(
        df["need_mk"].astype(int) == 1,
        np.where(df["has_mk"].astype(int) == 1, 1, 0),
        2,
    ).astype(int)
    df["location_match"] = (df["location_id_user"] == df["location_id_shift"]).astype(int)
    return df


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


def _is_weekday_vectorized(date_series: pd.Series) -> pd.Series:
    is_weekend = date_series.dt.dayofweek >= 5
    is_holiday = date_series.isin(HOLIDAYS)
    return (~is_weekend & ~is_holiday).astype(int)


def _get_time_of_day_vectorized(hour_series: pd.Series) -> np.ndarray:
    conditions = [
        (hour_series >= 6) & (hour_series < 12),
        (hour_series >= 12) & (hour_series < 18),
        (hour_series >= 18) & (hour_series < 24),
        (hour_series < 6) | (hour_series >= 24),
    ]
    choices = ["morning", "day", "evening", "night"]
    return np.select(conditions, choices, default="night")


def _add_temporal_features_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["shift_day_of_week"] = df["start_at"].dt.dayofweek
    df["shift_hour"] = df["start_at"].dt.hour
    df["shift_time_of_day"] = _get_time_of_day_vectorized(df["shift_hour"])
    df["shift_date"] = df["start_at"].dt.normalize()
    df["is_weekday"] = _is_weekday_vectorized(df["shift_date"])
    return df.drop(columns=["shift_date"])


def _add_shift_features(frame: pd.DataFrame) -> pd.DataFrame:
    window_end_offset = 2
    window_start_offset = 28 + window_end_offset

    frame = frame.copy()
    frame["start_at"] = pd.to_datetime(frame["start_at"], utc=True).dt.tz_localize(None)
    frame["reward_per_hour"] = frame["reward"] / frame["hours"]

    shifts_history = frame[
        ["shift_id", "start_at", "reward", "hours", "reward_per_hour"]
    ].drop_duplicates(subset=["shift_id"])
    LOGGER.info("Adding window features for all pairs")
    frame = _add_window_features_vectorized(
        frame, shifts_history, window_start_offset, window_end_offset
    )
    LOGGER.info(
        "Window features added: median_pay_window, is_high_paid_window"
    )

    frame = _add_basic_features(frame)
    LOGGER.info("Basic shift features added")

    frame = _add_temporal_features_vectorized(frame)
    LOGGER.info("Temporal shift features added")
    return frame


def _calendar_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame[
        (frame["start_at"] >= pd.Timestamp("2026-02-01"))
        & (frame["start_at"] <= pd.Timestamp("2026-02-28 23:59:59.999999999"))
    ].copy()
    test = frame[
        (frame["start_at"] >= pd.Timestamp("2026-03-01"))
        & (frame["start_at"] <= pd.Timestamp("2026-03-22 23:59:59.999999999"))
    ].copy()
    if train.empty or test.empty:
        raise ValueError("Calendar split produced empty train or test set.")
    return train, test


def _generate_shap_plots(
    model: object,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    output_dir: Path,
    sample_size: int,
) -> dict[str, str]:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    x_test_sample = x_test.head(min(sample_size, len(x_test))).copy()
    background = x_train.head(min(sample_size, len(x_train))).copy()

    explainer = shap.TreeExplainer(model, data=background)
    shap_values = explainer.shap_values(x_test_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif getattr(shap_values, "ndim", 0) == 3:
        shap_values = shap_values[:, :, 1]

    summary_path = plots_dir / "shap_summary.png"
    bar_path = plots_dir / "shap_bar.png"

    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, x_test_sample, show=False)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=140, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, x_test_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=140, bbox_inches="tight")
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

    LOGGER.info("Stage 3/8: Basic, temporal, window and historical feature groups ready")
    LOGGER.info("Feature frame rows=%s columns=%s", len(frame), frame.shape[1])

    LOGGER.info("Stage 4/8: Calendar split train=Feb 2026 test=Mar 1-22 2026")
    train_frame, test_frame = _calendar_split(frame)
    LOGGER.info("Split rows: train=%s test=%s", len(train_frame), len(test_frame))
    tech_cols = ["id", "shift_date", "workplace_id", "id_differential", "rate_bin"]
    train_features_export = train_frame.drop(
        columns=[col for col in tech_cols if col in train_frame.columns]
    )
    test_features_export = test_frame.drop(
        columns=[col for col in tech_cols if col in test_frame.columns]
    )
    train_features_export.to_csv(output_dir / "train_features_final.csv", index=False)
    test_features_export.to_csv(output_dir / "test_features_final.csv", index=False)
    LOGGER.info("Saved train_features_final.csv shape=%s", train_features_export.shape)
    LOGGER.info("Saved test_features_final.csv shape=%s", test_features_export.shape)
    LOGGER.info("Total train features: %s", len(train_features_export.columns) - 3)

    feature_columns = [
        "reward_per_hour",
        "user_total_views",
        "user_success_rate",
        "shift_total_views",
        "user_avg_reward_per_hour",
        "location_id_shift",
        "shift_hour",
        "free_spots_at_prediction",
        "user_location_experience",
        "user_weekend_ratio",
        "view_count_for_shift",
        "hours",
        "employer_reliability",
        "employer_avg_reward",
        "employer_popularity",
        "user_day_ratio",
        "user_high_paid_ratio",
        "user_task_type_experience",
        "shift_day_of_week",
        "capacity",
        "matches_time_of_day",
        "has_apply_before",
        "user_employer_experience",
        "user_shifts_on_same_day",
        "user_long_shift_ratio",
        "mk_match_2",
    ]
    missing = [c for c in feature_columns if c not in train_features_export.columns]
    if missing:
        raise ValueError(f"Missing final LightGBM feature columns: {missing}")

    x_train = train_features_export[feature_columns].copy()
    x_test = test_features_export[feature_columns].copy()
    category_maps: dict[str, list[object]] = {}
    for col in x_train.select_dtypes(include=["object"]).columns:
        categories = pd.Categorical(x_train[col]).categories
        category_maps[col] = list(categories)
        x_train[col] = pd.Categorical(x_train[col], categories=categories).codes
        x_test[col] = pd.Categorical(x_test[col], categories=categories).codes

    y_train = train_features_export["target"].astype(int)
    y_test = test_features_export["target"].astype(int)

    stable_params = {
        "learning_rate": 0.0836687516498472,
        "num_leaves": 281,
        "max_depth": 4,
        "feature_fraction": 0.5878428289981247,
        "bagging_fraction": 0.7814848755455905,
        "bagging_freq": 5,
        "min_child_samples": 174,
        "lambda_l1": 0.005335617091358604,
        "lambda_l2": 0.9232524971186427,
        "min_gain_to_split": 0.3697662395911488,
        "objective": "binary",
        "metric": "auc",
        "is_unbalance": True,
        "random_state": cfg.random_state,
        "verbosity": -1,
    }

    LOGGER.info("Stage 5/8: Fitting final LightGBM on 26 features")
    try:
        import lightgbm as lgb
        from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score
    except Exception as exc:  # noqa: BLE001
        raise ImportError("LightGBM final training dependencies are unavailable") from exc

    model = lgb.LGBMClassifier(**stable_params)
    model.fit(x_train, y_train)

    LOGGER.info("Stage 6/8: Running LightGBM inference and calculating target metric")
    proba = model.predict_proba(x_test)[:, 1]
    train_proba = model.predict_proba(x_train)[:, 1]
    train_roc_auc = float(roc_auc_score(y_train, train_proba))
    test_roc_auc = float(roc_auc_score(y_test, proba))
    LOGGER.info("Train ROC-AUC: %.4f", train_roc_auc)
    LOGGER.info("Test ROC-AUC: %.4f", test_roc_auc)

    precision, recall, thresholds = precision_recall_curve(y_test, proba)
    f1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    best_idx = int(np.argmax(f1_scores))
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 1.0
    y_pred = (proba >= best_threshold).astype(int)
    best_f1 = float(f1_scores[best_idx])
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    LOGGER.info("Best threshold by F1: %.4f", best_threshold)
    LOGGER.info("Best F1: %.4f", best_f1)
    LOGGER.info("Confusion matrix:\n%s", conf_matrix)

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
        "train_roc_auc": train_roc_auc,
        "test_roc_auc": test_roc_auc,
        "best_f1": best_f1,
        "best_threshold": float(best_threshold),
        "confusion_matrix": conf_matrix,
    }

    LOGGER.info("Stage 7/8: Saving model and artifacts to %s", output_dir)
    with (output_dir / "model.pkl").open("wb") as f:
        pickle.dump(model, f)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "feature_schema.json").write_text(
        json.dumps(
            {
                "feature_columns": feature_columns,
                "category_maps": category_maps,
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
        "",
        "## Model",
        "",
        "- model: LightGBM",
        f"- features: {len(feature_columns)}",
        f"- best_threshold: {best_threshold}",
        f"- train_roc_auc: {train_roc_auc}",
        f"- test_roc_auc: {test_roc_auc}",
        f"- best_f1: {best_f1}",
    ]

    shap_result: dict[str, str] = {}
    if cfg.skip_shap:
        report_lines.extend(["", "## SHAP", "", "- SHAP skipped by config (--skip-shap)."])
    else:
        try:
            shap_result = _generate_shap_plots(
                model, x_train, x_test, output_dir, cfg.shap_sample_size
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
