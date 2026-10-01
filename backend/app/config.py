"""App settings loaded from environment variables. Import `settings` to read them."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Backend settings, read from environment variables (see /.env.example)."""

    database_url: str


settings = Settings()
