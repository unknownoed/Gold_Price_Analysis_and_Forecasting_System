from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy import func
from .base import Base

class PredictionRecord(Base):
    __tablename__ = "prediction_records"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)

    prediction = Column(String)
    confidence = Column(Float)
    predicted_price = Column(Float)

    actual_price = Column(Float)
    is_correct = Column(Integer)

    created_at = Column(DateTime, default=func.now())
    evaluated_at = Column(DateTime)