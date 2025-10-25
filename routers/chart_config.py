import json
from datetime import datetime
from typing import Annotated, Optional

from fastapi import APIRouter, HTTPException, Depends, Query, Form, Security
from pydantic import Json
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlmodel import select

from db import session_manager
from models.chart_config import ChartConfig
from models.study_template import StudyTemplate
from schemas.chart_config import DeleteChartConfigResponseModel, DeleteStudyTemplateResponseModel


router = APIRouter()


# @router.get("/1.1/charts", tags=["chart-config"], response_model=ChartConfigOut)
@router.get("/1.1/charts", tags=["chart-config"])
async def get_chart_config(
        client: str = Query(...),
        user: str = Query(...),
        chart: Optional[int] = None,
        session: AsyncSession = Depends(session_manager.session),
        # auth_result: dict | str = Security(auth.verify)
):
    if client != user:
        raise HTTPException(status_code=400, detail={"msg": "client and user param does not match", "field": "user"})

    # sub = auth_result["sub"].split("|")[-1]
    #
    # if sub != user:
    #     raise HTTPException(status_code=403, detail={"msg": "Invalid credential!"})

    if chart:
        stmt = (select(ChartConfig)
                .where(ChartConfig.userId == f"{user}")
                .where(ChartConfig.id == int(chart)))
        queryset = await session.execute(stmt)
        try:
            chart_config = queryset.one()
        except Exception as e:
            raise HTTPException(status_code=400, detail={"msg": "Invalid Chart ID", "field": "chart"})

        if not chart_config:
            return {
                "data": None,
                "status": "no_data"
            }

        data = chart_config[0].__dict__
        data["timestamp"] = int(datetime.timestamp(chart_config[0].created_at))

        return {
            "data": data,
            "status": "ok"
        }

    else:
        stmt = select(ChartConfig).where(ChartConfig.userId == f"{user}")
        queryset = await session.execute(stmt)
        records = queryset.mappings().all()
        if not records:
            return {
                "data": [],
                "status": "no_data"
            }

        data = []
        for rd in records:
            chart_config = rd["ChartConfig"].__dict__
            chart_config["timestamp"] = int(datetime.timestamp(rd["ChartConfig"].created_at))
            data.append(chart_config)

        return {
            "data": data,
            "status": "ok"
        }


@router.get("/1.1/study_templates", tags=["study-templates"])
async def get_study_templates(
        client: str = Query(...),
        user: str = Query(...),
        template: Optional[str] = None,
        session: AsyncSession = Depends(session_manager.session),
        # auth_result: dict | str = Security(auth.verify)
):
    if client != user:
        raise HTTPException(status_code=400, detail={"msg": "client and user param does not match", "field": "user"})

    # sub = auth_result["sub"].split("|")[-1]
    #
    # if sub != user:
    #     raise HTTPException(status_code=403, detail={"msg": "Invalid credential!"})

    if template:
        stmt = (select(StudyTemplate)
                .where(StudyTemplate.userId == f"{user}")
                .where(StudyTemplate.name == template))
        queryset = await session.execute(stmt)
        try:
            study_template = queryset.one()
        except Exception as e:
            raise HTTPException(status_code=400, detail={"msg": "Invalid Template Name", "field": "template"})

        if not study_template:
            return {
                "data": None,
                "status": "no_data"
            }

        data = study_template[0].__dict__
        data["timestamp"] = int(datetime.timestamp(study_template[0].created_at))

        return {
            "data": data,
            "status": "ok"
        }

    else:
        stmt = select(StudyTemplate.name).where(StudyTemplate.userId == f"{user}")
        queryset = await session.execute(stmt)
        records = queryset.mappings().all()
        if not records:
            return {
                "data": [],
                "status": "ok"
            }

        return {
            "data": records,
            "status": "ok"
        }


@router.post("/1.1/study_templates", tags=["study-templates"])
async def post_study_templates(
        client: str = Query(...),
        user: str = Query(...),
        name: Annotated[str, Form()] = "",
        content: Annotated[Json, Form()] = None,
        session: AsyncSession = Depends(session_manager.session),
        # auth_result: dict | str = Security(auth.verify)
):
    if client != user:
        raise HTTPException(status_code=400, detail={"msg": "client and user param does not match", "field": "user"})
    if not content:
        raise HTTPException(status_code=400, detail={"msg": "Data cannot be null"})

    config_item = {
        "name": name,
        "userId": f"{user}"
    }

    # sub = auth_result["sub"].split("|")[-1]
    #
    # if sub != config_item["userId"]:
    #     raise HTTPException(status_code=403, detail={"msg": "Invalid credential!"})

    try:
        config_item["content"] = json.dumps(content)
    except json.JSONDecoder:
        raise HTTPException(status_code=400, detail={"msg": "Invalid JSON Content", "field": "content"})

    stmt = (select(StudyTemplate)
            .where(StudyTemplate.userId == f"{user}")
            .where(StudyTemplate.name == name))
    queryset = await session.execute(stmt)

    is_created = False

    try:
        study_template = queryset.one()
    except Exception as e:
        is_created = True
        study_template = None

    if is_created:
        study_template = StudyTemplate(**config_item)
        session.add(study_template)
        await session.commit()
        await session.refresh(study_template)
    else:
        study_template = study_template[0]
        study_template.content = json.dumps(content)
        session.add(study_template)
        await session.commit()
        await session.refresh(study_template)

    return {
        "id": study_template.id,
        "status": "ok"
    }


@router.post("/1.1/charts", tags=["chart-config"])
async def post_chart_config(
        client: str = Query(...),
        user: str = Query(...),
        chart: Optional[int] = None,
        # content: ChartConfigIn = dict,
        name: Annotated[str, Form()] = "",
        content: Annotated[Json, Form()] = None,
        symbol: Annotated[str, Form()] = "",
        resolution: Annotated[str, Form()] = "",
        session: AsyncSession = Depends(session_manager.session),
        # auth_result: dict | str = Security(auth.verify)
):
    if client != user:
        raise HTTPException(status_code=400, detail={"msg": "client and user param does not match", "field": "user"})
    if not content:
        raise HTTPException(status_code=400, detail={"msg": "Data cannot be null"})
    # TODO: Need to use request application/json instead of form data
    # config_item = {**content.dict(), **{"userId": f"{user}{client[:8]}"}}
    config_item = {
        "name": name,
        "symbol": symbol,
        "resolution": resolution,
        "userId": f"{user}"
    }
    try:
        # config_item["content"] = json.dumps(config_item["content"])
        config_item["content"] = json.dumps(content)
    except json.JSONDecoder:
        raise HTTPException(status_code=400, detail={"msg": "Invalid JSON Content", "field": "content"})
    # if auth_result["sub"] != config_item["user_id"]:
    #     raise HTTPException(status_code=400, detail={"msg": "Invalid credential.", "field": "user_id"})
    if chart:
        stmt = (select(ChartConfig)
                .where(ChartConfig.userId == f"{user}")
                .where(ChartConfig.id == int(chart)))
        queryset = await session.execute(stmt)
        try:
            chart_config = queryset.one()[0]
        except Exception as e:
            raise HTTPException(status_code=400, detail={"msg": "Invalid Chart ID", "field": "chart"})
        else:
            chart_config.name = config_item["name"]
            chart_config.symbol = config_item["symbol"]
            chart_config.resolution = config_item["resolution"]
            chart_config.content = config_item["content"]
            session.add(chart_config)
            await session.commit()
            await session.refresh(chart_config)

            return {
                "id": chart_config.id,
                "status": "ok"
            }
    else:
        item = ChartConfig(**config_item)
        session.add(item)
        await session.commit()
        await session.refresh(item)

        return {
            "id": item.id,
            "status": "ok"
        }
#
#
# @router.put("/1.1/charts", tags=["chart-config"], response_model=ChartConfigOut)
# async def update_item(
#         config_item: ChartConfigIn,
#         session: AsyncSession = Depends(session_manager.session),
#         config_id: int = None,
#         # auth_result: dict | str = Security(auth.verify)
# ):
#     """
#     Update an item.
#     """
#     item = session.get(ChartConfig, config_id)
#     # if auth_result["sub"] != item.user_id:
#     #     raise HTTPException(status_code=400, detail="Invalid credential.")
#     update_dict = config_item.model_dump(exclude_unset=True)
#     await item.sqlmodel_update(update_dict)
#     session.add(item)
#     await session.commit()
#     await session.refresh(item)
#     return item


@router.delete("/1.1/charts", tags=["chart-config"], response_model=DeleteChartConfigResponseModel)
async def delete_chart_config(
        client: str = Query(...),
        user: str = Query(...),
        chart: int = Query(...),
        session: AsyncSession = Depends(session_manager.session),
        # auth_result: dict | str = Security(auth.verify)
):
    """
    Delete chart config.
    """
    if client != user:
        raise HTTPException(status_code=400, detail={"msg": "client and user param does not match", "field": "user"})

    # sub = auth_result["sub"].split("|")[-1]
    #
    # if sub != user:
    #     raise HTTPException(status_code=403, detail={"msg": "Invalid credential!"})

    stmt = (select(ChartConfig)
            .where(ChartConfig.userId == f"{user}")
            .where(ChartConfig.id == int(chart)))
    queryset = await session.execute(stmt)
    try:
        chart_config = queryset.one()
    except Exception as e:
        raise HTTPException(status_code=400, detail={"msg": "Invalid Chart ID", "field": "chart"})

    await session.delete(chart_config[0])
    await session.commit()

    return {
        "status": "ok"
    }


@router.delete("/1.1/study_templates", tags=["study-templates"], response_model=DeleteStudyTemplateResponseModel)
async def delete_study_template(
        client: str = Query(...),
        user: str = Query(...),
        template: str = Query(...),
        session: AsyncSession = Depends(session_manager.session),
        # auth_result: dict | str = Security(auth.verify)
):
    """
    Delete chart config.
    """
    if client != user:
        raise HTTPException(status_code=400, detail={"msg": "client and user param does not match", "field": "user"})

    # sub = auth_result["sub"].split("|")[-1]
    #
    # if sub != user:
    #     raise HTTPException(status_code=403, detail={"msg": "Invalid credential!"})

    stmt = (select(StudyTemplate)
            .where(StudyTemplate.userId == f"{user}")
            .where(StudyTemplate.name == template))
    queryset = await session.execute(stmt)
    try:
        study_template = queryset.one()
    except Exception as e:
        raise HTTPException(status_code=400, detail={"msg": "Invalid Template Name", "field": "template"})

    await session.delete(study_template[0])
    await session.commit()

    return {
        "status": "ok"
    }
