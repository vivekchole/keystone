from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from app.core.database import get_db
from app.models.business import Business

router = APIRouter(prefix="/keystone/business")


class BusinessCreate(BaseModel):
    name: str
    services: List[str]
    timings: str
    description: Optional[str] = None


class BusinessResponse(BusinessCreate):
    id: str

    class Config:
        orm_mode = True


@router.post("/", response_model=BusinessResponse)
async def create_business(
    business: BusinessCreate, db: Session = Depends(get_db)
):
    db_business = Business(**business.dict())
    db.add(db_business)
    db.commit()
    db.refresh(db_business)
    return db_business


@router.get("/{business_id}", response_model=BusinessResponse)
async def get_business(business_id: str, db: Session = Depends(get_db)):
    db_business = db.query(Business).filter(Business.id == business_id).first()
    if not db_business:
        raise HTTPException(status_code=404, detail="Business not found")
    return db_business
