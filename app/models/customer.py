from sqlalchemy import Column, String, Text
from sqlalchemy.dialects.postgresql import ARRAY
from app.core.database import Base
import uuid


def generate_uuid():
    return str(uuid.uuid4())


class Customer(Base):
    __tablename__ = "customers"

    id = Column(String, primary_key=True, index=True, default=generate_uuid)
    phone_number = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    delivery_address = Column(String, nullable=True)
    preferences = Column(Text, nullable=True)
    frequent_items = Column(ARRAY(String), nullable=True)
