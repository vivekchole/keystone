from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from app.core.database import get_db
from app.models.business import Business
from app.models.order import Order
from app.models.customer import Customer

router = APIRouter(prefix="/keystone/admin")


@router.get("/overview")
async def get_admin_overview(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Unified Admin Dashboard Overview:
    Aggregates platform statistics, all businesses, all orders, and customer directory.
    """
    businesses = db.query(Business).order_by(Business.name).all()
    orders = db.query(Order).order_by(Order.created_at.desc()).all()
    customers = db.query(Customer).order_by(Customer.name).all()

    return {
        "stats": {
            "total_businesses": len(businesses),
            "total_orders": len(orders),
            "total_customers": len(customers),
            "pending_orders": sum(1 for o in orders if o.status == "pending"),
            "confirmed_orders": sum(1 for o in orders if o.status == "confirmed"),
        },
        "businesses": [
            {
                "id": b.id,
                "name": b.name,
                "services": b.services,
                "timings": b.timings,
                "description": b.description,
            }
            for b in businesses
        ],
        "orders": [
            {
                "id": o.id,
                "business_id": o.business_id,
                "customer_name": o.customer_name,
                "items": o.items,
                "special_instructions": o.special_instructions,
                "status": o.status,
                "idempotency_key": o.idempotency_key,
                "created_at": o.created_at.isoformat() if o.created_at else None,
            }
            for o in orders
        ],
        "customers": [
            {
                "id": c.id,
                "phone_number": c.phone_number,
                "name": c.name,
                "delivery_address": c.delivery_address,
                "preferences": c.preferences,
                "frequent_items": c.frequent_items,
            }
            for c in customers
        ],
    }


@router.get("/orders")
async def get_all_orders(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """List all orders across all businesses."""
    orders = db.query(Order).order_by(Order.created_at.desc()).all()
    return [
        {
            "id": o.id,
            "business_id": o.business_id,
            "customer_name": o.customer_name,
            "items": o.items,
            "special_instructions": o.special_instructions,
            "status": o.status,
            "idempotency_key": o.idempotency_key,
            "created_at": o.created_at.isoformat() if o.created_at else None,
        }
        for o in orders
    ]


@router.get("/customers")
async def get_all_customers(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """List all customer profiles."""
    customers = db.query(Customer).order_by(Customer.name).all()
    return [
        {
            "id": c.id,
            "phone_number": c.phone_number,
            "name": c.name,
            "delivery_address": c.delivery_address,
            "preferences": c.preferences,
            "frequent_items": c.frequent_items,
        }
        for c in customers
    ]
