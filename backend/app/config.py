"""App settings, read from environment variables (see /.env.example)."""

import os

DATABASE_URL = os.environ["DATABASE_URL"]

# Clerk settings are optional here so Alembic can run without them. Without
# CLERK_SECRET_KEY, every route that needs sign-in fails with 500.
CLERK_SECRET_KEY = os.environ.get("CLERK_SECRET_KEY", "")
# Web app origins allowed to send tokens (the token's "azp" claim), comma-separated.
# Empty means the origin is not checked.
CLERK_AUTHORIZED_PARTIES = [
    origin.strip()
    for origin in os.environ.get("CLERK_AUTHORIZED_PARTIES", "").split(",")
    if origin.strip()
]
