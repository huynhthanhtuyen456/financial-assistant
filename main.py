import logging
import sys
# Set up the scheduler
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from fastapi.staticfiles import StaticFiles

from config import get_settings
from core.models import HealthCheck
from db import session_manager
# Import routes
from routers import (
    dividend_events,
    scfa,
    stock_price,
    balancesheet,
    income_statement,
    cashflow,
    overall_dashboard,
    eda,
    benchmark_testing
)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG if get_settings().debug_logs else logging.INFO)
logger = logging.getLogger(__name__)

token_auth_scheme = HTTPBearer()


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


app = FastAPI(
    lifespan=lifespan,
    title=get_settings().project_name,
    docs_url="/docs",
    root_path="/",
)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

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

# Include routes
API_V1 = "/api/v1"
app.include_router(stock_price.router, prefix=API_V1)
app.include_router(dividend_events.router, prefix=API_V1)
app.include_router(scfa.router, prefix=API_V1)
app.include_router(balancesheet.router)
app.include_router(income_statement.router)
app.include_router(cashflow.router)
app.include_router(overall_dashboard.router)
app.include_router(eda.router)
app.include_router(benchmark_testing.router)

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

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
