from sqlalchemy import Column, String, Text
from sqlalchemy.dialects.postgresql import ARRAY
from app.core.database import Base
import uuid


def generate_uuid():
    return str(uuid.uuid4())


class Business(Base):
    __tablename__ = "businesses"

    id = Column(String, primary_key=True, index=True, default=generate_uuid)
    name = Column(String, index=True, nullable=False)
    services = Column(ARRAY(String), nullable=False)
    timings = Column(String, nullable=False)
    description = Column(Text, nullable=True)
