from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import func
from .base import Base

class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True)

    username = Column(String, unique=True)
    password_hash = Column(String)

    # 风险偏好
    risk_preference = Column(String, default="medium")  # low / medium / high

    # 市场偏好（偏多/偏空）
    market_bias = Column(String, default="neutral")  # bullish / bearish / neutral

    # 用户策略标签（LLM提取）
    strategy_tags = Column(JSON)  # ["逢低买入", "趋势跟随"]

    keywords = Column(JSON)  # 用户关注的关键词（LLM提取） ["美元指数", "美债收益率"]

    # 用户关注指标权重（个性化）
    indicator_weights = Column(JSON)
    # e.g. {"usd_index": 0.3, "bond_yield": 0.3, "oil": 0.1}

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
