"""SceneSense FastAPI entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import router
from app.config import get_settings
from app.utils.hardware import detect_hardware
from app.utils.logging_setup import get_logger, setup_logging

setup_logging()
logger = get_logger("scenesense")


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    settings.ensure_directories()
    hw = detect_hardware()
    logger.info(
        "SceneSense v%s starting | device=%s cuda=%s ram_gb=%s",
        __version__,
        hw.get("recommended_device"),
        hw.get("cuda_available"),
        hw.get("ram_gb"),
    )
    yield
    logger.info("SceneSense shutting down")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="SceneSense",
        description="Efficient Video Question Answering research MVP",
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix="/api")
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
