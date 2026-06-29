# learning_service.py

from sqlalchemy.orm import Session
from gold_ai.models.learning import LearningWeights
from gold_ai.models.prediction import PredictionRecord
from datetime import datetime, UTC


# =========================
# 1️⃣ 记录预测
# =========================
def record_prediction(db: Session, user_id: int, prediction: str, price: float, confidence: float):
    record = PredictionRecord(
        user_id=user_id,
        prediction=prediction,
        predicted_price=price,
        confidence=confidence,
        created_at=datetime.now(UTC)
    )
    db.add(record)
    db.commit()

    return record.id


# =========================
# 2️⃣ 更新实际结果
# =========================
def update_prediction_result(db: Session, record_id: int, actual_price: float):
    record = db.query(PredictionRecord).filter_by(id=record_id).first()

    if not record:
        return

    record.actual_price = actual_price
    record.evaluated_at = datetime.now(UTC)

    # 判断对错（简单规则）
    if record.prediction == "bullish" and actual_price > record.predicted_price:
        record.is_correct = 1
    elif record.prediction == "bearish" and actual_price < record.predicted_price:
        record.is_correct = 1
    else:
        record.is_correct = 0

    db.commit()


# =========================
# 3️⃣ 获取/初始化权重
# =========================
def get_or_create_weights(db: Session, user_id: int):
    weights = db.query(LearningWeights).filter_by(user_id=user_id).first()

    if not weights:
        weights = LearningWeights(
            user_id=user_id,
            weights={
                "usd_index": 0.3,
                "bond_yield": 0.3,
                "oil": 0.1,
                "news": 0.3
            },
            learning_rate=0.05
        )
        db.add(weights)
        db.commit()

    return weights


# =========================
# 4️⃣ 权重更新（核心）
# =========================

def update_weights(db: Session, user_id: int):
    weights_obj = get_or_create_weights(db, user_id)

    records = db.query(PredictionRecord)\
        .filter_by(user_id=user_id)\
        .filter(PredictionRecord.is_correct != None)\
        .order_by(PredictionRecord.created_at.desc())\
        .limit(20).all()

    if not records:
        return weights_obj.weights

    accuracy = sum(r.is_correct for r in records) / len(records)

    lr = weights_obj.learning_rate
    weights = weights_obj.weights

    # 简单策略：根据准确率调节“新闻 vs 宏观”
    if accuracy < 0.5:
        weights["news"] += lr
        weights["usd_index"] -= lr / 2
        weights["bond_yield"] -= lr / 2
    else:
        weights["usd_index"] += lr / 2
        weights["bond_yield"] += lr / 2
        weights["news"] -= lr

    # 归一化
    total = sum(weights.values())
    for k in weights:
        weights[k] = max(0.01, weights[k] / total)

    weights_obj.weights = weights
    db.commit()

    return weights


# =========================
# 5️⃣ 获取当前权重（给分析用）
# =========================
def get_weights_context(db: Session, user_id: int):
    weights = get_or_create_weights(db, user_id).weights

    return f"""
    当前模型权重：
    美元指数: {weights['usd_index']:.2f}
    美债收益率: {weights['bond_yield']:.2f}
    原油: {weights['oil']:.2f}
    新闻: {weights['news']:.2f}
    """