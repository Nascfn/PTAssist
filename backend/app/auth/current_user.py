"""The signed-in user: check the Clerk token, then load or create the caller's row."""

from typing import Annotated

from clerk_backend_api import Clerk
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.auth.clerk import (
    ClerkUnavailableError,
    InvalidTokenError,
    get_clerk,
    get_primary_email,
    verify_session_token,
)
from app.db import get_db
from app.models import Role, User

# Reads "Authorization: Bearer <token>" and adds the Authorize button to /docs.
bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    clerk: Annotated[Clerk, Depends(get_clerk)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    """Return the caller's User. A new user is created as a patient on first call."""
    if credentials is None:
        raise _unauthorized("Missing bearer token")
    try:
        clerk_id = verify_session_token(clerk, request)
        user = db.scalar(select(User).where(User.clerk_id == clerk_id))
        if user is None:
            email = get_primary_email(clerk, clerk_id)
            if email is None:
                raise HTTPException(
                    status.HTTP_403_FORBIDDEN, "Your account needs an email address"
                )
            user = create_patient(db, clerk_id, email)
    except InvalidTokenError as exc:
        raise _unauthorized(str(exc)) from None
    except ClerkUnavailableError:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Could not reach Clerk. Try again."
        ) from None
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def create_patient(db: Session, clerk_id: str, email: str) -> User:
    """Insert a patient, or reuse the row if a parallel first call just made it."""
    db.execute(
        insert(User)
        .values(clerk_id=clerk_id, email=email, role=Role.PATIENT)
        .on_conflict_do_nothing(index_elements=[User.clerk_id])
    )
    db.commit()
    return db.scalars(select(User).where(User.clerk_id == clerk_id)).one()


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status.HTTP_401_UNAUTHORIZED, detail, headers={"WWW-Authenticate": "Bearer"}
    )
