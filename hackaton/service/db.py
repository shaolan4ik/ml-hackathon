from __future__ import annotations

import asyncio
from pathlib import Path

import aiosqlite

from hackaton.service.config import settings

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    location_id TEXT NOT NULL,
    is_strict_location INTEGER NOT NULL,
    has_mk INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS shifts (
    id TEXT PRIMARY KEY,
    start_at TEXT NOT NULL,
    location_id TEXT NOT NULL,
    task_type TEXT NOT NULL,
    employer_id TEXT NOT NULL,
    workplace_id TEXT NOT NULL,
    need_mk INTEGER NOT NULL,
    id_differential INTEGER NOT NULL,
    hours INTEGER NOT NULL,
    reward REAL NOT NULL,
    capacity INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    shift_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    interaction TEXT NOT NULL,
    ts TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_shift_id ON events(shift_id);
CREATE INDEX IF NOT EXISTS idx_events_user_id ON events(user_id);
CREATE INDEX IF NOT EXISTS idx_shifts_location_id ON shifts(location_id);
"""


async def init_db() -> None:
    db_path = Path(settings.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(settings.db_path) as db:
        await db.executescript(SCHEMA_SQL)
        await db.commit()


async def init_db_for(db_path: str) -> None:
    target = Path(db_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(SCHEMA_SQL)
        await db.commit()


def migrate() -> None:
    asyncio.run(init_db())


if __name__ == "__main__":
    migrate()
