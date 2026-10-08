"""Entry point: creates the FastAPI app and plugs in every router."""

from fastapi import FastAPI

from app.routes import health, me

app = FastAPI(title="PTAssist API")
app.include_router(health.router)
app.include_router(me.router)
