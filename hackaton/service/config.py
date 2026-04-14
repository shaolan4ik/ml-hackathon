from dataclasses import dataclass
from os import getenv


@dataclass(frozen=True, slots=True)
class Settings:
    app_host: str = getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(getenv("APP_PORT", "8000"))
    db_path: str = getenv("DB_PATH", "./data/hackaton.db")
    prepare_sleep_seconds: int = int(getenv("PREPARE_SLEEP_SECONDS", "10"))


settings = Settings()
