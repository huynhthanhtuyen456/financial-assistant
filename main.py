import logging
import sys
# Set up the scheduler
from contextlib import asynccontextmanager
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler  # runs tasks in the background
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer

from config import get_settings
from core.models import HealthCheck
from db import session_manager
# Import routes
# from routers import chart_config
from routers import dividend_events
from routers import scfa
from routers import scta
from routers import stock_price
from scripts.refresh_materialized_view import refresh_materialized_view

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG if get_settings().debug_logs else logging.INFO)
logger = logging.getLogger(__name__)

token_auth_scheme = HTTPBearer()


# The task to run
def refresh_materialized_view_daily_task():
    logger.info(f"Start refreshing materialized view at {datetime.now()}")
    refresh_materialized_view()
    logger.info(f"Finish refreshing materialized view at {datetime.now()}")


# Set up the scheduler
scheduler = BackgroundScheduler()
trigger = CronTrigger(hour=0)  # <-- CHANGED LINE (run every 00:00 AM UTC, 07:00 AM HO_CHI_MINH/ASIA)
scheduler.add_job(refresh_materialized_view_daily_task, trigger)
scheduler.start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Function that handles startup and shutdown events.
    To understand more, read https://fastapi.tiangolo.com/advanced/events/
    """
    yield
    if session_manager.engine is not None:
        # Close the DB connection
        await session_manager.close()
    scheduler.shutdown()


app = FastAPI(
    lifespan=lifespan,
    title=get_settings().project_name,
    docs_url="/docs",
    root_path="/api/v1",
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = {}
    for error in exc.errors():
        if len(error["loc"]) > 2:
            errors[error["loc"][1]] = error["msg"]
        else:
            errors[error["loc"][0]] = error["msg"]

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=jsonable_encoder({"status": False, "errors": errors, "message": "Invalid Request"}),
    )

app.include_router(stock_price.router)
app.include_router(dividend_events.router)
app.include_router(scfa.router)
# app.include_router(scta.router)


if get_settings().debug_logs:
    origins = [
        "http://backend.localhost",
        "http://localhost",
        "http://localhost:8080",
        "http://localhost:5500",
        "http://localhost:3000",
        "http://127.0.0.1:5500",
        "http://127.0.0.1:3000",
    ]
else:
    origins = [
        "http://localhost:5000",
        "http://127.0.0.1:5000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health-check", response_model=HealthCheck, tags=["status"])
async def health_check():
   return {
       "name": get_settings().project_name,
       "version": get_settings().version,
       "description": get_settings().description
   }
