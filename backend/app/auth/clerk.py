"""The API's only Clerk calls: check a session token and look up a user's email."""

from functools import cache

import httpx
from clerk_backend_api import Clerk
from clerk_backend_api.models import ClerkBaseError, NoResponseError
from clerk_backend_api.security import (
    AuthenticateRequestOptions,
    TokenVerificationErrorReason,
)
from fastapi import Request

from app.config import CLERK_AUTHORIZED_PARTIES, CLERK_SECRET_KEY

# Reasons that mean Clerk failed us (no usable signing keys), not that the token is bad.
_CLERK_FAILURES = {
    TokenVerificationErrorReason.JWK_FAILED_TO_LOAD,
    TokenVerificationErrorReason.JWK_REMOTE_INVALID,
    TokenVerificationErrorReason.JWK_FAILED_TO_RESOLVE,
}


class InvalidTokenError(Exception):
    """The session token is invalid, expired, or from an origin we don't allow."""


class ClerkUnavailableError(Exception):
    """Clerk could not be reached, so the request could not be checked."""


@cache
def get_clerk() -> Clerk:
    """One Clerk client for all requests. A dependency, so tests can swap it."""
    if not CLERK_SECRET_KEY:
        raise RuntimeError("CLERK_SECRET_KEY is not set")
    # The SDK retries failed calls for up to an hour by default. Fail fast instead.
    return Clerk(bearer_auth=CLERK_SECRET_KEY, retry_config=None, timeout_ms=10_000)


def verify_session_token(clerk: Clerk, request: Request) -> str:
    """Check the session token in the Authorization header and return the Clerk user id.

    The SDK checks the signature against our Clerk instance's public keys (fetched with
    the secret key and cached for 5 minutes), the expiry, and the origin (azp claim).
    """
    options = AuthenticateRequestOptions(
        # None skips the origin check. An empty list would reject every token.
        authorized_parties=CLERK_AUTHORIZED_PARTIES or None,
        # Sign-in session tokens only, not Clerk API keys or machine tokens.
        accepts_token=["session_token"],
    )
    try:
        state = clerk.authenticate_request(request, options)
    except httpx.HTTPError as exc:  # Clerk unreachable while fetching its public keys
        raise ClerkUnavailableError from exc
    if state.reason in _CLERK_FAILURES:
        raise ClerkUnavailableError
    if not state.is_signed_in:
        raise InvalidTokenError(state.message or "Invalid token")
    return state.payload["sub"]


def get_primary_email(clerk: Clerk, clerk_id: str) -> str | None:
    """Look up the user's primary email address in Clerk. None if they have none."""
    try:
        user = clerk.users.get(user_id=clerk_id)
    except (ClerkBaseError, NoResponseError, httpx.HTTPError) as exc:
        raise ClerkUnavailableError from exc
    for address in user.email_addresses:
        if address.id is not None and address.id == user.primary_email_address_id:
            return address.email_address
    return None
