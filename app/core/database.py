import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


# Default to a local postgres DB if environment variable is not set
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/keystone"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class LazyAsyncPostgresSaver(BaseCheckpointSaver):
    """
    Lazily initializes AsyncPostgresSaver when an asyncio event loop is active.
    Prevents 'RuntimeError: no running event loop' during synchronous module imports.
    """

    def __init__(self, pool: AsyncConnectionPool):
        super().__init__()
        self._pool = pool
        self._saver = None

    def _get_saver(self):
        if self._saver is None:
            self._saver = AsyncPostgresSaver(self._pool)
        return self._saver

    def __getattr__(self, name):
        return getattr(self._get_saver(), name)

    async def aget_tuple(self, config):
        return await self._get_saver().aget_tuple(config)

    async def alist(self, config, *, filter=None, before=None, limit=None):
        async for item in self._get_saver().alist(
            config, filter=filter, before=before, limit=limit
        ):
            yield item

    async def aput(self, config, checkpoint, metadata, new_versions):
        return await self._get_saver().aput(
            config, checkpoint, metadata, new_versions
        )

    async def aput_writes(self, config, writes, task_id):
        return await self._get_saver().aput_writes(config, writes, task_id)


# Connection pool for the LangGraph checkpointer
# Adjust max_size if deploying to production
pool = AsyncConnectionPool(
    conninfo=DATABASE_URL, max_size=5, kwargs={"autocommit": True}, open=False
)
checkpointer = LazyAsyncPostgresSaver(pool)

