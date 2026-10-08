from fastapi import FastAPI
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables (Note: Uvicorn also loads .env automatically)
load_dotenv()

from app.api import chat, business
from app.core.database import Base, engine, checkpointer, pool
from app.models.order import Order  # noqa: F401


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Create all business models in PostgreSQL
    Base.metadata.create_all(bind=engine)

    # 2. Setup LangGraph checkpointer tables in PostgreSQL
    await pool.open()
    await checkpointer.setup()

    yield
    # Cleanup pool on shutdown
    await pool.close()


app = FastAPI(title="Keystone Chatbot API", lifespan=lifespan)

app.include_router(chat.router, tags=["chat"])
app.include_router(business.router, tags=["business"])


@app.get("/")
def read_root():
    return {"message": "Welcome to Keystone Chatbot API"}
