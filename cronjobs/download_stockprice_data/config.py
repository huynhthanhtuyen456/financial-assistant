from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    db_host: str = Field(..., env='DB_HOST')
    db_user: str = Field(..., env='DB_USER')
    db_password: str = Field(..., env='DB_PASSWORD')
    project_name: str = Field(..., env='PROJECT_NAME')
    debug_logs: bool = Field(False, env='DEBUG_LOGS')
    echo_sql: bool = Field(True, env='ECHO_SQL')
    auth0_domain: str = Field(..., env='AUTH0_DOMAIN')
    auth0_api_audience: str = Field(..., env='AUTH0_API_AUDIENCE')
    auth0_issuer: str = Field(..., env='AUTH0_ISSUER')
    auth0_algorithms: str = Field(..., env='AUTH0_ALGORITHMS')
    dnse_username: str = Field(..., env='DNSE_USERNAME')
    dnse_password: str = Field(..., env='DNSE_PASSWORD')
    dnse_host: str = Field(..., env='DNSE_HOST')
    version: str = Field(..., env='VERSION')
    description: str = Field(..., env='DESCRIPTION')

    @property
    def asyncpg_database_url(self):
        return f'postgresql+asyncpg://{self.db_user}:{self.db_password}@{self.db_host}/{self.db_user}'

    @property
    def database_psycopg_url(self):
        return f'postgresql+psycopg2://{self.db_user}:{self.db_password}@{self.db_host}/{self.db_user}'


@lru_cache
def get_settings():
    return Settings()
