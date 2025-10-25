import logging
import sys
from typing import Any

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base

from config import get_settings

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG if get_settings().debug_logs else logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

# Heavily inspired by https://praciano.com.br/fastapi-and-async-sqlalchemy-20-with-pytest-done-right.html


class DatabaseSessionManager:
    def __init__(self, host: str, engine_kwargs: dict[str, Any] = None):
        self.engine = create_async_engine(host, **engine_kwargs)
        self._session_maker = async_sessionmaker(autocommit=False, bind=self.engine)

    async def close(self):
        if self.engine is None:
            raise Exception("DatabaseSessionManager is not initialized")
        await self.engine.dispose()

        self.engine = None
        self._session_maker = None

    async def connect(self) -> AsyncConnection:
        if self.engine is None:
            raise Exception("DatabaseSessionManager is not initialized")

        async with self.engine.begin() as connection:
            try:
                yield connection
            except Exception:
                await connection.rollback()
                raise

    async def session(self) -> AsyncSession:
        if self._session_maker is None:
            raise Exception("DatabaseSessionManager is not initialized")

        async with self._session_maker() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()


session_manager = DatabaseSessionManager(
    get_settings().asyncpg_database_url,
    {"echo": get_settings().echo_sql}
)


# Create a new client and connect to the server
client = MongoClient(get_settings().mongodb_url, server_api=ServerApi('1'))
# Send a ping to confirm a successful connection

db = client.admin
collection = db["dividend_events"]

try:
    client.admin.command('ping')
    logger.info("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    logger.error(e)
