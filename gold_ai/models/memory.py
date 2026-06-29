from sqlalchemy import Column, Integer, String, DateTime, JSON
from .base import Base
from sqlalchemy import func
# from datetime import datetime

class ConversationMemory(Base):
    __tablename__ = "conversation_memory"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)

    role = Column(String)  # user / assistant
    content = Column(String)

    # embedding（用于向量检索）
    embedding = Column(JSON)  # 可选（后期接入）

    created_at = Column(DateTime, default=func.now())