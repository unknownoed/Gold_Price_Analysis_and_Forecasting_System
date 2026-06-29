from sqlalchemy import Column, Integer, String, DateTime, JSON
from sqlalchemy import func
from .base import Base


class UserBehavior(Base):
    __tablename__ = "user_behaviors"

    id = Column(Integer, primary_key=True)

    user_id = Column(Integer)

    action = Column(String)         # buy / sell / hold
    bias = Column(String)           # bullish / bearish / neutral
    horizon = Column(String)        # short / medium / long

    raw_text = Column(String)       # 原始用户输入
    extracted_strategy = Column(JSON)

    created_at = Column(DateTime, default=func.now())