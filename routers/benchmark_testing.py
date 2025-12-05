import math
import plotly.graph_objects as go

from fastapi.templating import Jinja2Templates

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from db import session_manager
import pandas as pd
import numpy as np
import plotly.express as px
from core.helpers import get_balance_sheet, get_income_statement, get_cash_flow
from plotly.subplots import make_subplots


router = APIRouter(
    prefix="/pipeline",
    tags=["pipeline"],
)
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def eda_balance_sheet(request: Request, session: AsyncSession = Depends(session_manager.session)):
    return templates.TemplateResponse("pipeline.html", {"request": request})