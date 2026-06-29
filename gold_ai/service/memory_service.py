from sqlalchemy.orm import Session
from gold_ai.models.user import UserProfile
from gold_ai.models.memory import ConversationMemory
from datetime import datetime, UTC
from openai import OpenAI
import json

from gold_ai.config import MOONSHOT_API_KEY, MOONSHOT_BASE_URL

client = OpenAI(
    api_key=MOONSHOT_API_KEY,
    base_url=MOONSHOT_BASE_URL
)


def save_message(db: Session, user_id: int, role: str, content: str):
    msg = ConversationMemory(
        user_id=user_id,
        role=role,
        content=content,
        created_at=datetime.now(UTC)
    )
    db.add(msg)
    db.commit()


def get_recent_memory(db: Session, user_id: int, limit=5):
    records = db.query(ConversationMemory)\
        .filter_by(user_id=user_id)\
        .order_by(ConversationMemory.created_at.desc())\
        .limit(limit).all()

    return [
        {"role": r.role, "content": r.content}
        for r in reversed(records)
    ]


def clear_memory(db: Session, user_id: int):
    """删除指定用户的所有对话记录"""
    db.query(ConversationMemory)\
        .filter_by(user_id=user_id)\
        .delete()
    db.commit()


def extract_user_strategy(query: str):
    prompt = f"""
    请从用户输入中提取交易信息，返回JSON：
    {{
        "bias": "bullish/bearish/neutral",
        "strategy": "简短描述",
        "risk": "low/medium/high"
    }}

    用户输入：
    {query}
    """

    response = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {"bias": "neutral", "strategy": "", "risk": "medium"}


def update_user_profile(db: Session, user_id: int, strategy_info: dict):
    profile = db.query(UserProfile).filter_by(id=user_id).first()

    if not profile:
        profile = UserProfile(id=user_id)
        db.add(profile)

    profile.market_bias = strategy_info.get("bias", "neutral")
    profile.risk_preference = strategy_info.get("risk", "medium")

    strategy = strategy_info.get("strategy")
    if strategy:
        if not profile.strategy_tags:
            profile.strategy_tags = []
        if strategy not in profile.strategy_tags:
            profile.strategy_tags.append(strategy)

    db.commit()


def get_user_profile_context(db: Session, user_id: int):
    profile = db.query(UserProfile).filter_by(id=user_id).first()

    if not profile:
        return "暂无用户画像"

    return f"""
    用户风险偏好: {profile.risk_preference}
    市场倾向: {profile.market_bias}
    策略: {profile.strategy_tags}
    """
