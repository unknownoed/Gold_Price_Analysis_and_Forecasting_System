from sqlalchemy import Column, Integer, JSON, Float, DateTime
from sqlalchemy import func
from .base import Base

class LearningWeights(Base):
    __tablename__ = "learning_weights"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)

    weights = Column(JSON)
    learning_rate = Column(Float, default=0.05)

    updated_at = Column(DateTime, default=func.now())  