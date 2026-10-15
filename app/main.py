# =====================================================================
# DEMO NOTICE:
# Automatically seeds demo customer profiles into PostgreSQL at app startup
# for demonstration and testing purposes (e.g. Vivek: 9876543210, Alice: 1234567890).
# =====================================================================

import os
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables (Note: Uvicorn also loads .env automatically)
load_dotenv()

from app.api import chat, business, admin
from app.core.database import Base, engine, checkpointer, pool, SessionLocal
from app.models.order import Order  # noqa: F401
from app.models.customer import Customer

logger = logging.getLogger("uvicorn.info")


def seed_demo_customers():
    """
    Inserts demo customer profiles into the PostgreSQL database at app startup.
    """
    print("\n[STARTUP] Adding customer profiles for demo purposes...")
    logger.info("Adding customer profiles for demo purposes...")
    db = SessionLocal()
    try:
        demo_customers = [
            {
                "phone_number": "9876543210",
                "name": "Vivek",
                "delivery_address": "42 Tech Boulevard, Suite 5",
                "preferences": "Non-dairy milk, spicy",
                "frequent_items": ["Large Cappuccino", "Croissant"],
            },
            {
                "phone_number": "1234567890",
                "name": "Alice",
                "delivery_address": "742 Evergreen Terrace",
                "preferences": "No onions",
                "frequent_items": ["Veggie Burger", "Fries"],
            },
        ]
        for data in demo_customers:
            existing = (
                db.query(Customer)
                .filter(Customer.phone_number == data["phone_number"])
                .first()
            )
            if not existing:
                cust = Customer(**data)
                db.add(cust)
        db.commit()
        print("[STARTUP] Demo customer profiles inserted successfully.\n")
        logger.info("Demo customer profiles inserted successfully.")
    except Exception as e:
        db.rollback()
        logger.warning(f"Failed to seed demo customer profiles: {e}")
    finally:
        db.close()


def seed_default_business():
    """Inserts a default demo business into PostgreSQL if not present."""
    from app.models.business import Business
    db = SessionLocal()
    try:
        default_biz = db.query(Business).filter(Business.id == "default").first()
        if not default_biz:
            db.add(
                Business(
                    id="default",
                    name="Keystone Cafe & Pizza",
                    services=["Wood-fired Pizza", "Coffee", "Garlic Bread"],
                    timings="10:00 AM - 10:00 PM",
                    description="Artisan wood-fired pizza, garlic bread, and specialty coffee."
                )
            )
            db.commit()
    except Exception as e:
        db.rollback()
        logger.warning(f"Failed to seed default business: {e}")
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Create all business models in PostgreSQL
    Base.metadata.create_all(bind=engine)

    # 2. Add customer profiles and default business for demo purposes at startup
    seed_demo_customers()
    seed_default_business()

    # 3. Setup LangGraph checkpointer tables in PostgreSQL
    await pool.open()
    await checkpointer.setup()

    yield
    # Cleanup pool on shutdown
    await pool.close()


app = FastAPI(title="Keystone Chatbot API", lifespan=lifespan)

# Mount static assets if the folder exists
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

app.include_router(chat.router, tags=["chat"])
app.include_router(business.router, tags=["business"])
app.include_router(admin.router, tags=["admin"])


@app.get("/")
@app.get("/ui")
def read_ui():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Welcome to Keystone Chatbot API"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}
