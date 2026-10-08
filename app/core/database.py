import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from psycopg_pool import AsyncConnectionPool
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


# Connection pool for the LangGraph checkpointer
# Adjust max_size if deploying to production
pool = AsyncConnectionPool(
    conninfo=DATABASE_URL, max_size=5, kwargs={"autocommit": True}, open=False
)
checkpointer = AsyncPostgresSaver(pool)
