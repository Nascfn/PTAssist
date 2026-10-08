"""Shared test setup: a real Postgres test database and a fake Clerk.

Tests never call Clerk. FakeClerk signs session tokens with a key made here and answers
the two network calls the Clerk SDK makes, so the SDK still checks every token for real.
"""

# The environment must be set before the app is imported, so imports come after it.
# ruff: noqa: E402

import os

WEB_APP_ORIGIN = "http://localhost:5173"

# Always the test database, never DATABASE_URL from your shell: tests wipe it.
os.environ["DATABASE_URL"] = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://ptassist:ptassist@localhost:5433/ptassist_test",
)
os.environ["CLERK_SECRET_KEY"] = "sk_test_fake"
os.environ["CLERK_AUTHORIZED_PARTIES"] = WEB_APP_ORIGIN

import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jwt
import pytest
from alembic import command
from alembic.config import Config
from clerk_backend_api import Clerk
from clerk_backend_api.security import verifytoken
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from jwt.algorithms import RSAAlgorithm
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.auth.clerk import get_clerk
from app.db import engine
from app.main import app
from app.models import Base

SIGNING_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
KEY_ID = "test-key"


class FakeClerk:
    """Plays Clerk: signs session tokens and answers user lookups."""

    def __init__(self) -> None:
        self.emails: dict[str, str | None] = {}  # Clerk user id -> primary email
        self.lookups = 0  # user lookups the API made
        self.lookup_error: Exception | None = None  # raise this from user lookups

    def add_user(self, clerk_id: str, email: str | None) -> None:
        self.emails[clerk_id] = email

    def token(
        self, clerk_id: str, *, key: Any = SIGNING_KEY, kid: str = KEY_ID, **claims: Any
    ) -> str:
        """A session token shaped like Clerk's. Keyword claims override the defaults."""
        now = int(time.time())
        payload = {
            "sub": clerk_id,
            "azp": WEB_APP_ORIGIN,
            "sid": "sess_test",
            "iat": now,
            "nbf": now - 10,
            "exp": now + 60,
            "v": 2,
            **claims,
        }
        return jwt.encode(payload, key, algorithm="RS256", headers={"kid": kid})

    def headers(self, clerk_id: str, **claims: Any) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token(clerk_id, **claims)}"}

    def fetch_jwks(self, options: Any) -> dict[str, Any]:
        """Stands in for the SDK's call that fetches Clerk's public keys."""
        public_key = RSAAlgorithm.to_jwk(SIGNING_KEY.public_key(), as_dict=True)
        return {"keys": [{**public_key, "kid": KEY_ID, "alg": "RS256", "use": "sig"}]}

    def get_user(self, *, user_id: str) -> SimpleNamespace:
        """Stands in for clerk.users.get. Returns a primary and a second address."""
        self.lookups += 1
        if self.lookup_error is not None:
            raise self.lookup_error
        email = self.emails[user_id]
        if email is None:
            return SimpleNamespace(primary_email_address_id=None, email_addresses=[])
        return SimpleNamespace(
            primary_email_address_id="idn_primary",
            email_addresses=[
                SimpleNamespace(id="idn_other", email_address="other@example.com"),
                SimpleNamespace(id="idn_primary", email_address=email),
            ],
        )


@pytest.fixture(scope="session")
def database() -> Iterator[None]:
    """Build the schema with the real migrations, then drop it (tests downgrade too)."""
    # No alembic.ini: its logging settings would change the log levels tests check.
    config = Config()
    config.set_main_option(
        "script_location", str(Path(__file__).parents[1] / "migrations")
    )
    command.upgrade(config, "head")
    yield
    command.downgrade(config, "base")


@pytest.fixture
def db(database: None) -> Iterator[Session]:
    """A session on the test database. Every test starts with empty tables."""
    tables = ", ".join(table.name for table in Base.metadata.sorted_tables)
    with engine.begin() as connection:
        connection.execute(text(f"TRUNCATE {tables}"))
    with Session(engine) as session:
        yield session


@pytest.fixture
def clerk(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeClerk]:
    """A FakeClerk, used by the app in place of the real Clerk."""
    fake = FakeClerk()
    sdk = Clerk(bearer_auth=os.environ["CLERK_SECRET_KEY"])
    sdk.users = SimpleNamespace(get=fake.get_user)
    monkeypatch.setattr(verifytoken, "_fetch_jwks", fake.fetch_jwks)
    app.dependency_overrides[get_clerk] = lambda: sdk
    yield fake
    del app.dependency_overrides[get_clerk]


@pytest.fixture
def client(db: Session, clerk: FakeClerk) -> TestClient:
    return TestClient(app)
