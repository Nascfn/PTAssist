"""App settings loaded from environment variables. Import `settings` to read them."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Backend settings, read from environment variables (see /.env.example)."""

    database_url: str


settings = Settings(database_url=os.environ["DATABASE_URL"])
