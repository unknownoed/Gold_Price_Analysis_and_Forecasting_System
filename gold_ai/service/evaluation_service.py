from sqlalchemy.orm import Session
from datetime import datetime, timedelta, timezone, UTC

from gold_ai.models.prediction import PredictionRecord
from gold_ai.service.learning_service import update_weights
from gold_ai.service import market_service as ms 

# 获取当前金价（用于评估预测结果）
def get_current_gold_price():
    try:
        price = float(ms.gold.history(period="1d")['Close'].iloc[-1])
        return price
    except Exception as e:
        print("获取金价失败:", e)
        return None
    

# 判断预测是否准确
def evaluate_single_prediction(record, current_price):
    if record.prediction == "bullish":
        return 1 if current_price > record.predicted_price else 0

    elif record.prediction == "bearish":
        return 1 if current_price < record.predicted_price else 0

    else:
        # 中性预测（简单处理）
        return 1 if abs(current_price - record.predicted_price) < 0.5 else 0
    

def evaluate_predictions(db: Session, hours_delay=6):
    """
    只评估超过一定时间的预测（避免刚预测就评估）
    """

    current_price = get_current_gold_price()
    if current_price is None:
        return {"status": "fail", "reason": "无法获取金价"}

    cutoff_time = datetime.now(UTC) - timedelta(hours=hours_delay)

    # 找到未评估的预测
    records = db.query(PredictionRecord)\
        .filter(PredictionRecord.actual_price == None)\
        .filter(PredictionRecord.created_at < cutoff_time)\
        .all()

    if not records:
        return {"status": "ok", "message": "无可评估预测"}

    results = []

    for r in records:
        is_correct = evaluate_single_prediction(r, current_price)

        r.actual_price = current_price
        r.is_correct = is_correct
        r.evaluated_at = datetime.now(UTC)

        results.append({
            "id": r.id,
            "prediction": r.prediction,
            "predicted_price": r.predicted_price,
            "actual_price": current_price,
            "correct": is_correct
        })

    db.commit()

    # =========================
    # 4️⃣ 触发学习（按用户分组）
    # =========================
    user_ids = list(set(r.user_id for r in records))

    updated_weights = {}
    for uid in user_ids:
        weights = update_weights(db, uid)
        updated_weights[uid] = weights

    return {
        "status": "ok",
        "evaluated_count": len(records),
        "details": results,
        "updated_weights": updated_weights
    }


# =========================
# 4️⃣ 统计准确率（可用于dashboard）
# =========================
def get_accuracy_stats(db: Session, user_id: int, limit=50):
    records = db.query(PredictionRecord)\
        .filter(PredictionRecord.user_id == user_id)\
        .filter(PredictionRecord.is_correct != None)\
        .order_by(PredictionRecord.created_at.desc())\
        .limit(limit).all()

    if not records:
        return {"accuracy": 0, "total": 0}

    accuracy = sum(r.is_correct for r in records) / len(records)

    return {
        "accuracy": round(accuracy, 2),
        "total": len(records)
    }