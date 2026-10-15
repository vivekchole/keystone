from datetime import datetime
from sqlalchemy import Column, String, Text, ForeignKey, DateTime, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY
from app.core.database import Base
import uuid


def generate_uuid():
    return str(uuid.uuid4())


class Order(Base):
    __tablename__ = "orders"

    id = Column(String, primary_key=True, index=True, default=generate_uuid)
    business_id = Column(String, ForeignKey("businesses.id"), nullable=False)
    customer_name = Column(String, nullable=False)
    items = Column(ARRAY(String), nullable=False)
    special_instructions = Column(Text, nullable=True)
    status = Column(String, default="pending")
    idempotency_key = Column(String, index=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=True)

    __table_args__ = (
        UniqueConstraint("business_id", "idempotency_key", name="uq_business_idempotency_key"),
    )
