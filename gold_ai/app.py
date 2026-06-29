from flask import Flask, render_template, jsonify, request, session
from functools import wraps
from gold_ai.service import market_service as ms
from gold_ai.service import news_service as ns
from gold_ai.service import analysis_service as as_logic
from gold_ai.service import chat_service as chat_logic
from gold_ai.service import usernews_service as uns
from gold_ai.service.memory_service import get_recent_memory, get_user_profile_context, clear_memory
from gold_ai.service.learning_service import get_weights_context, get_or_create_weights
from gold_ai.service.evaluation_service import get_accuracy_stats, evaluate_predictions
from gold_ai.service.usernews_service import get_personalized_news, agent_search_news
from gold_ai.service.tts_service import generate_tts_audio
from gold_ai.service.cache import TTLCache
from gold_ai.config import (
    MOONSHOT_API_KEY, MOONSHOT_BASE_URL,
    DEEPSEEK_API_KEY, MARKET_CACHE_TTL, SNAPSHOT_CACHE_TTL,
    FLASK_DEBUG, FLASK_SECRET_KEY
)
from gold_ai.db.session import SessionLocal
from gold_ai.models.user import UserProfile
from gold_ai.models.prediction import PredictionRecord
from openai import OpenAI
import datetime
import json
import os
import time

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), 'settings.json')


def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f)
    return {}


def persist_settings(data):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(data, f, indent=2)


# Caches
_market_summary_cache = {"summary": None, "timestamp": 0}
_market_summary_ttl = 300
_market_data_cache = TTLCache(ttl_seconds=MARKET_CACHE_TTL)
_snapshot_cache = TTLCache(ttl_seconds=SNAPSHOT_CACHE_TTL)

app = Flask(__name__)

# Auto-init DB tables
from gold_ai.models.base import Base
from gold_ai.db.session import engine
Base.metadata.create_all(bind=engine)

app.secret_key = FLASK_SECRET_KEY

client = OpenAI(
    api_key=MOONSHOT_API_KEY,
    base_url=MOONSHOT_BASE_URL
)

DEFAULT_USER_ID = 1


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"error": "请先登录"}), 401
        return f(*args, **kwargs)
    return decorated


def get_current_user_id():
    return session.get('user_id', DEFAULT_USER_ID)


from gold_ai.service.auth_service import register_user, authenticate_user


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ==================== 页面路由 ====================

@app.route('/')
def portal():
    return render_template('index.html')


@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/market')
def market():
    return render_template('index.html')


@app.route('/news')
def news():
    return render_template('index.html')


@app.route('/analysis')
def analysis():
    return render_template('index.html')


@app.route('/settings')
def settings():
    return render_template('index.html')


# ==================== 认证 API ====================

@app.route('/api/auth/register', methods=['POST'])
def api_register():
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        db = SessionLocal()
        try:
            user, error = register_user(db, username, password)
            if error:
                return jsonify({"error": error}), 400
            session['user_id'] = user.id
            return jsonify({"id": user.id, "username": user.username})
        finally:
            db.close()
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/auth/login', methods=['POST'])
def api_login():
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        if not username or not password:
            return jsonify({"error": "请输入用户名和密码"}), 400
        db = SessionLocal()
        try:
            user = authenticate_user(db, username, password)
            if not user:
                return jsonify({"error": "用户名或密码错误"}), 401
            session['user_id'] = user.id
            return jsonify({"id": user.id, "username": user.username})
        finally:
            db.close()
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/auth/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({"status": "ok"})


@app.route('/api/auth/me')
def api_me():
    if 'user_id' not in session:
        return jsonify({"logged_in": False})
    db = SessionLocal()
    try:
        user = db.query(UserProfile).filter_by(id=session['user_id']).first()
        if not user:
            session.clear()
            return jsonify({"logged_in": False})
        return jsonify({
            "logged_in": True,
            "id": user.id,
            "username": user.username
        })
    finally:
        db.close()


# ==================== 行情 API ====================

@app.route('/api/market_data')
@login_required
def api_market():
    try:
        period = request.args.get('period', '1mo')
        interval = request.args.get('interval', '1d')
        cache_key = f"market_data:{period}:{interval}"

        cached = _market_data_cache.get(cache_key)
        if cached:
            return jsonify({**cached, "cached": True})

        gold_df = ms.gold.history(period=period, interval=interval)
        usd_df = ms.meiyuanzhishu.history(period=period, interval=interval)

        if gold_df.empty:
            return jsonify({"error": "金价数据获取失败"}), 500

        result = {
            "dates": gold_df.index.strftime('%Y-%m-%d').tolist(),
            "gold_k": gold_df[['Open', 'Close', 'Low', 'High']].values.tolist(),
            "usd_index": usd_df['Close'].tolist() if not usd_df.empty else []
        }
        _market_data_cache.set(cache_key, result)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/market_data/full')
@login_required
def api_market_full():
    try:
        period = request.args.get('period', '1mo')
        cache_key = f"market_full:{period}"

        cached = _market_data_cache.get(cache_key)
        if cached:
            return jsonify({**cached, "cached": True})

        gold_df = ms.gold.history(period=period)
        usd_df = ms.meiyuanzhishu.history(period=period)
        bond_df = ms.meizhaililv.history(period=period)
        sp500_df = ms.biaopu500.history(period=period)
        oil_df = ms.yuanyoujiage.history(period=period)
        vix_df = ms.vix.history(period=period)
        gld_df = ms.gld_etf.history(period=period)

        result = {
            "dates": gold_df.index.strftime('%Y-%m-%d').tolist(),
            "gold_close": gold_df['Close'].tolist(),
            "gold_k": gold_df[['Open', 'Close', 'Low', 'High']].values.tolist(),
            "usd_index": usd_df['Close'].tolist() if not usd_df.empty else [],
            "bond_yield": bond_df['Close'].tolist() if not bond_df.empty else [],
            "sp500": sp500_df['Close'].tolist() if not sp500_df.empty else [],
            "oil": oil_df['Close'].tolist() if not oil_df.empty else [],
            "vix": vix_df['Close'].tolist() if not vix_df.empty else [],
            "gld_etf": gld_df['Close'].tolist() if not gld_df.empty else []
        }
        _market_data_cache.set(cache_key, result)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/market_snapshot')
@login_required
def api_market_snapshot():
    try:
        cached = _snapshot_cache.get("snapshot")
        if cached:
            return jsonify({**cached, "cached": True})

        def get_snapshot(ticker, name):
            df = ticker.history(period="5d")
            if df.empty or len(df) < 2:
                return {"name": name, "price": None, "change_pct": None}
            latest = float(df['Close'].iloc[-1])
            prev = float(df['Close'].iloc[-2])
            change_pct = round((latest - prev) / prev * 100, 2)
            return {"name": name, "price": round(latest, 2), "change_pct": change_pct}

        result = {
            "gold": get_snapshot(ms.gold, "黄金期货"),
            "usd_index": get_snapshot(ms.meiyuanzhishu, "美元指数"),
            "bond_yield": get_snapshot(ms.meizhaililv, "美债10Y"),
            "oil": get_snapshot(ms.yuanyoujiage, "原油"),
            "sp500": get_snapshot(ms.biaopu500, "标普500"),
            "vix": get_snapshot(ms.vix, "VIX恐慌指数"),
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat()
        }
        _snapshot_cache.set("snapshot", result)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/market_summary')
@login_required
def api_market_summary():
    global _market_summary_cache
    force = request.args.get('force', '0') == '1'

    if not force and _market_summary_cache["summary"] is not None:
        if time.time() - _market_summary_cache["timestamp"] < _market_summary_ttl:
            return jsonify({"summary": _market_summary_cache["summary"], "cached": True})

    try:
        # Use cached snapshot data
        snapshot = api_market_snapshot().get_json()
        news_data = ns.get_jinshi_news(limit=10)
        classified = ns.classify_news(news_data)
        gold_news = [n['content'] for n in classified if n.get('gold_related')]

        prompt = f"""请根据以下市场数据和新闻，用3-5句中文简要总结当前黄金市场状况：
        市场数据：{snapshot}
        相关新闻：{gold_news[:5] if gold_news else '无'}
        要求：判断趋势（上涨/下跌/震荡），给出关键价位，点出核心影响因素。"""

        response = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        summary = response.choices[0].message.content.strip()
        _market_summary_cache = {"summary": summary, "timestamp": time.time()}
        return jsonify({"summary": summary, "cached": False})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== 新闻 API ====================

@app.route('/api/news')
@login_required
def api_news():
    news_data = ns.get_jinshi_news(limit=30)
    return jsonify(ns.classify_news(news_data))


@app.route('/api/news/categories')
@login_required
def api_news_categories():
    return jsonify(ns.get_categories())


@app.route('/api/news/search', methods=['POST'])
@login_required
def api_news_search():
    try:
        data = request.get_json()
        query = data.get('query', '')
        if not query:
            return jsonify({"error": "请输入搜索内容"}), 400

        db = SessionLocal()
        try:
            # Check both env var and settings.json for DeepSeek key
            ds_key = DEEPSEEK_API_KEY or load_settings().get("deepseek_api_key", "")
            if ds_key:
                result = agent_search_news(db, get_current_user_id(), query, ds_key)
                result["mode"] = "agent"
            else:
                result = get_personalized_news(db, get_current_user_id(), query, client)
                result["mode"] = "pipeline"
            return jsonify(result)
        finally:
            db.close()
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== 预测 API ====================

@app.route('/api/prediction/latest')
@login_required
def api_prediction_latest():
    db = SessionLocal()
    try:
        record = db.query(PredictionRecord)\
            .filter_by(user_id=get_current_user_id())\
            .order_by(PredictionRecord.created_at.desc())\
            .first()

        accuracy = get_accuracy_stats(db, get_current_user_id())

        if not record:
            return jsonify({"prediction": None, "accuracy": accuracy})

        return jsonify({
            "prediction": record.prediction,
            "confidence": record.confidence,
            "predicted_price": record.predicted_price,
            "actual_price": record.actual_price,
            "is_correct": record.is_correct,
            "created_at": record.created_at.isoformat() if record.created_at else None,
            "accuracy": accuracy
        })
    finally:
        db.close()


@app.route('/api/prediction/history')
@login_required
def api_prediction_history():
    db = SessionLocal()
    try:
        records = db.query(PredictionRecord)\
            .filter_by(user_id=get_current_user_id())\
            .order_by(PredictionRecord.created_at.desc())\
            .limit(30).all()

        history = []
        for r in records:
            history.append({
                "id": r.id,
                "prediction": r.prediction,
                "confidence": r.confidence,
                "predicted_price": r.predicted_price,
                "actual_price": r.actual_price,
                "is_correct": r.is_correct,
                "created_at": r.created_at.isoformat() if r.created_at else None
            })

        accuracy = get_accuracy_stats(db, get_current_user_id())

        return jsonify({"history": history, "accuracy": accuracy})
    finally:
        db.close()


@app.route('/api/prediction/evaluate', methods=['POST'])
@login_required
def api_prediction_evaluate():
    db = SessionLocal()
    try:
        hours = request.json.get('hours_delay', 1) if request.json else 1
        result = evaluate_predictions(db, hours_delay=hours)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "fail", "reason": str(e)}), 500
    finally:
        db.close()


# ==================== 报告 API ====================

@app.route('/api/report', methods=['POST'])
@login_required
def api_report():
    """生成结构化 AI 分析报告（13章节完整模板）"""
    try:
        user_input = request.json.get('message', '')
        if not user_input:
            return jsonify({"error": "请输入分析问题"}), 400

        db = SessionLocal()
        try:
            result = as_logic.generate_ai_report(db, get_current_user_id(), user_input)
            return jsonify(result)
        finally:
            db.close()
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== 对话 API（聊天） ====================

@app.route('/api/chat', methods=['POST'])
@login_required
def api_chat():
    """对话式 AI：金融专家，多轮记忆"""
    try:
        user_input = request.json.get('message', '')
        if not user_input:
            return jsonify({"error": "请输入消息"}), 400

        db = SessionLocal()
        try:
            result = chat_logic.ai_chat_conversation(db, get_current_user_id(), user_input)
            return jsonify(result)
        finally:
            db.close()
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat/history', methods=['GET', 'DELETE'])
@login_required
def api_chat_history():
    """获取对话历史（最近30条）或清空"""
    db = SessionLocal()
    try:
        if request.method == 'DELETE':
            clear_memory(db, get_current_user_id())
            return jsonify({"status": "ok"})
        memory = get_recent_memory(db, get_current_user_id(), limit=30)
        return jsonify({"messages": memory})
    finally:
        db.close()


# ==================== TTS API ====================

@app.route('/api/tts', methods=['POST'])
@login_required
def api_tts():
    import threading
    import os

    text = request.json.get('text', '') if request.json else ''
    if not text:
        return jsonify({"error": "请输入文本"}), 400

    result = {"path": None, "error": None}

    def _generate():
        try:
            result["path"] = generate_tts_audio(text)
        except Exception as e:
            result["error"] = str(e)

    t = threading.Thread(target=_generate)
    t.start()
    t.join(timeout=30)

    if result["error"]:
        return jsonify({"error": result["error"]}), 500
    if not result["path"] or not os.path.exists(result["path"]):
        return jsonify({"error": "语音生成失败"}), 500

    try:
        data = open(result["path"], 'rb').read()
        mimetype = 'audio/mpeg' if result["path"].endswith('.mp3') else 'audio/wav'
        return app.response_class(data, mimetype=mimetype)
    finally:
        try:
            os.unlink(result["path"])
        except OSError:
            pass


# ==================== 设置 API ====================

@app.route('/api/settings/profile', methods=['GET', 'POST'])
@login_required
def api_settings_profile():
    db = SessionLocal()
    try:
        if request.method == 'GET':
            profile = db.query(UserProfile).filter_by(id=get_current_user_id()).first()
            if not profile:
                return jsonify({
                    "risk_preference": "medium",
                    "market_bias": "neutral",
                    "strategy_tags": [],
                    "keywords": []
                })
            return jsonify({
                "risk_preference": profile.risk_preference,
                "market_bias": profile.market_bias,
                "strategy_tags": profile.strategy_tags or [],
                "keywords": profile.keywords or []
            })
        else:
            data = request.get_json()
            profile = db.query(UserProfile).filter_by(id=get_current_user_id()).first()
            if not profile:
                profile = UserProfile(id=get_current_user_id())
                db.add(profile)
            if 'risk_preference' in data:
                profile.risk_preference = data['risk_preference']
            if 'market_bias' in data:
                profile.market_bias = data['market_bias']
            if 'strategy_tags' in data:
                profile.strategy_tags = data['strategy_tags']
            if 'keywords' in data:
                profile.keywords = data['keywords']
            db.commit()
            return jsonify({"status": "ok"})
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@app.route('/api/settings/model', methods=['GET', 'POST'])
@login_required
def api_settings_model():
    db = SessionLocal()
    try:
        if request.method == 'GET':
            weights = get_or_create_weights(db, get_current_user_id())
            return jsonify({
                "weights": weights.weights,
                "learning_rate": weights.learning_rate
            })
        else:
            data = request.get_json()
            weights = get_or_create_weights(db, get_current_user_id())
            if 'weights' in data:
                weights.weights = data['weights']
            if 'learning_rate' in data:
                weights.learning_rate = data['learning_rate']
            db.commit()
            return jsonify({"status": "ok"})
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@app.route('/api/settings/apikeys', methods=['GET', 'POST'])
@login_required
def api_settings_apikeys():
    try:
        if request.method == 'GET':
            s = load_settings()
            return jsonify({
                "kimi_api_key": s.get("kimi_api_key", ""),
                "tavily_api_key": s.get("tavily_api_key", ""),
                "serpapi_key": s.get("serpapi_key", ""),
                "deepseek_api_key": s.get("deepseek_api_key", "")
            })
        else:
            data = request.get_json()
            s = load_settings()
            for key in ["kimi_api_key", "tavily_api_key", "serpapi_key", "deepseek_api_key"]:
                if key in data:
                    s[key] = data[key]
            persist_settings(s)
            return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=FLASK_DEBUG)
