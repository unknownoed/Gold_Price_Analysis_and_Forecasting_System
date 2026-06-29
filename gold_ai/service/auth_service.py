from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.orm import Session
from gold_ai.models.user import UserProfile


def register_user(db: Session, username: str, password: str):
    """Register a new user. Returns (user, error_message)."""
    if not username or not password:
        return None, "用户名和密码不能为空"
    if len(username) < 3:
        return None, "用户名至少3个字符"
    if len(password) < 6:
        return None, "密码至少6个字符"

    existing = db.query(UserProfile).filter_by(username=username).first()
    if existing:
        return None, "用户名已存在"

    user = UserProfile(
        username=username,
        password_hash=generate_password_hash(password),
        risk_preference="medium",
        market_bias="neutral"
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user, None


def authenticate_user(db: Session, username: str, password: str):
    """Verify credentials. Returns user or None."""
    user = db.query(UserProfile).filter_by(username=username).first()
    if not user or not user.password_hash:
        return None
    if check_password_hash(user.password_hash, password):
        return user
    return None


def get_user_by_id(db: Session, user_id: int):
    return db.query(UserProfile).filter_by(id=user_id).first()
