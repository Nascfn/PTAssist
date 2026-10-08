"""Tests for GET /me and the Clerk token check behind it."""

import logging
import time

import httpx
import pytest
from clerk_backend_api.security import (
    TokenVerificationError,
    TokenVerificationErrorReason,
    verifytoken,
)
from cryptography.hazmat.primitives.asymmetric import rsa
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.auth.current_user import create_patient
from app.models import Role, User

OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def count_users(db: Session) -> int:
    return db.scalar(select(func.count()).select_from(User))


def test_first_call_creates_a_patient(client, clerk, db) -> None:
    clerk.add_user("user_new", "pat@example.com")

    response = client.get("/me", headers=clerk.headers("user_new"))

    assert response.status_code == 200
    user = db.scalars(select(User).where(User.clerk_id == "user_new")).one()
    assert user.email == "pat@example.com"
    assert user.role == Role.PATIENT
    assert response.json() == {
        "id": str(user.id),
        "email": "pat@example.com",
        "role": "patient",
    }


def test_later_calls_return_the_same_user_without_asking_clerk(
    client, clerk, db
) -> None:
    clerk.add_user("user_new", "pat@example.com")

    first = client.get("/me", headers=clerk.headers("user_new"))
    second = client.get("/me", headers=clerk.headers("user_new"))

    assert second.status_code == 200
    assert second.json() == first.json()
    assert clerk.lookups == 1
    assert count_users(db) == 1


def test_role_comes_from_our_database(client, clerk, db) -> None:
    therapist = User(clerk_id="user_t", email="ther@example.com", role=Role.THERAPIST)
    db.add(therapist)
    db.commit()

    response = client.get("/me", headers=clerk.headers("user_t"))

    assert response.status_code == 200
    assert response.json() == {
        "id": str(therapist.id),
        "email": "ther@example.com",
        "role": "therapist",
    }


def test_role_in_the_token_is_ignored(client, clerk, db) -> None:
    """Wrong role: a role claim in the token can't make a patient a therapist."""
    db.add(User(clerk_id="user_p", email="pat@example.com", role=Role.PATIENT))
    db.commit()

    response = client.get("/me", headers=clerk.headers("user_p", role="therapist"))

    assert response.status_code == 200
    assert response.json()["role"] == "patient"


def test_returns_only_the_callers_own_user(client, clerk, db) -> None:
    """Wrong owner: a token only ever sees its own user, never another one."""
    alice = User(clerk_id="user_a", email="alice@example.com", role=Role.PATIENT)
    bob = User(clerk_id="user_b", email="bob@example.com", role=Role.PATIENT)
    db.add_all([alice, bob])
    db.commit()

    response = client.get("/me", headers=clerk.headers("user_b"))

    assert response.status_code == 200
    assert response.json()["id"] == str(bob.id)
    assert response.json()["email"] == "bob@example.com"


BAD_HEADERS = {
    "no token": lambda clerk: {},
    "not a bearer token": lambda clerk: {"Authorization": "Basic dXNlcjpwYXNz"},
    "not a JWT": lambda clerk: {"Authorization": "Bearer not-a-jwt"},
    "signed with another key": lambda clerk: clerk.headers("user_1", key=OTHER_KEY),
    "unknown key id": lambda clerk: clerk.headers("user_1", kid="unknown-key"),
    "expired": lambda clerk: clerk.headers("user_1", exp=int(time.time()) - 60),
    "from another origin": lambda clerk: clerk.headers(
        "user_1", azp="https://evil.example"
    ),
    "Clerk API key": lambda clerk: {"Authorization": "Bearer ak_test_123"},
}


@pytest.mark.parametrize("make_headers", BAD_HEADERS.values(), ids=list(BAD_HEADERS))
def test_bad_token_gets_401(client, clerk, db, make_headers) -> None:
    clerk.add_user("user_1", "pat@example.com")

    response = client.get("/me", headers=make_headers(clerk))

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert count_users(db) == 0


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectError("Clerk is down"),
        TokenVerificationError(TokenVerificationErrorReason.JWK_FAILED_TO_LOAD),
    ],
    ids=["unreachable", "keys not loaded"],
)
def test_clerk_down_while_checking_the_token_gets_503(
    client, clerk, monkeypatch, failure
) -> None:
    def fail(options: object) -> None:
        raise failure

    monkeypatch.setattr(verifytoken, "_fetch_jwks", fail)

    # A key id the SDK has not cached yet, so it has to ask Clerk for it.
    response = client.get("/me", headers=clerk.headers("user_1", kid="new-key"))

    assert response.status_code == 503


def test_clerk_down_while_looking_up_the_email_gets_503(client, clerk, db) -> None:
    clerk.add_user("user_new", "pat@example.com")
    clerk.lookup_error = httpx.ConnectError("Clerk is down")

    response = client.get("/me", headers=clerk.headers("user_new"))

    assert response.status_code == 503
    assert count_users(db) == 0


def test_account_without_an_email_gets_403(client, clerk, db) -> None:
    clerk.add_user("user_phone_only", None)

    response = client.get("/me", headers=clerk.headers("user_phone_only"))

    assert response.status_code == 403
    assert count_users(db) == 0


def test_create_patient_twice_keeps_one_row(db) -> None:
    """Two first calls at once both insert. The second must reuse the row, not fail."""
    first = create_patient(db, "user_new", "pat@example.com")
    second = create_patient(db, "user_new", "pat@example.com")

    assert second.id == first.id
    assert count_users(db) == 1


def test_token_and_email_are_not_logged(client, clerk, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    # As if someone turned on SQL logging to debug (SQLAlchemy keeps it off by default).
    caplog.set_level(logging.INFO, logger="sqlalchemy.engine")
    clerk.add_user("user_new", "pat@example.com")
    token = clerk.token("user_new")

    response = client.get("/me", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    assert "INSERT INTO users" in caplog.text  # SQL is logged at this level...
    assert token not in caplog.text
    assert "pat@example.com" not in caplog.text  # ...but not its values
