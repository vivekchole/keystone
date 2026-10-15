from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Dict, Any
from app.core.database import get_db
from app.models.business import Business
from app.models.order import Order

router = APIRouter(prefix="/keystone/business")


class BusinessCreate(BaseModel):
    name: str
    services: List[str]
    timings: str
    description: Optional[str] = None

    def dict(self, *args, **kwargs):
        # Backward compatibility for legacy tests
        return self.model_dump(*args, **kwargs)


class BusinessUpdate(BaseModel):
    name: Optional[str] = None
    services: Optional[List[str]] = None
    timings: Optional[str] = None
    description: Optional[str] = None


class BusinessResponse(BusinessCreate):
    id: str

    model_config = ConfigDict(from_attributes=True)


@router.get("/", response_model=List[BusinessResponse])
async def list_businesses(db: Session = Depends(get_db)):
    """List all registered businesses."""
    return db.query(Business).order_by(Business.name).all()


@router.post("/", response_model=BusinessResponse)
async def create_business(
    business: BusinessCreate, db: Session = Depends(get_db)
):
    """Create a new business."""
    payload = business.model_dump() if hasattr(business, "model_dump") else business.dict()
    db_business = Business(**payload)
    db.add(db_business)
    db.commit()
    db.refresh(db_business)
    return db_business


@router.get("/{business_id}", response_model=BusinessResponse)
async def get_business(business_id: str, db: Session = Depends(get_db)):
    """Retrieve details of a business by ID."""
    db_business = db.query(Business).filter(Business.id == business_id).first()
    if not db_business:
        raise HTTPException(status_code=404, detail="Business not found")
    return db_business


@router.put("/{business_id}", response_model=BusinessResponse)
async def update_business(
    business_id: str,
    update_data: BusinessUpdate,
    db: Session = Depends(get_db),
):
    """Update settings and profile for a business account."""
    db_business = db.query(Business).filter(Business.id == business_id).first()
    if not db_business:
        raise HTTPException(status_code=404, detail="Business not found")

    payload = update_data.model_dump(exclude_unset=True)
    for field, val in payload.items():
        setattr(db_business, field, val)

    db.commit()
    db.refresh(db_business)
    return db_business


@router.get("/{business_id}/orders")
async def get_business_orders(
    business_id: str, db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """Retrieve all orders placed for a specific business."""
    db_business = db.query(Business).filter(Business.id == business_id).first()
    if not db_business:
        raise HTTPException(status_code=404, detail="Business not found")

    orders = (
        db.query(Order)
        .filter(Order.business_id == business_id)
        .order_by(Order.created_at.desc())
        .all()
    )

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
