import datetime
import os
import json
import re
from openai import OpenAI

from gold_ai.config import HTTP_PROXY, HTTPS_PROXY, MOONSHOT_API_KEY, MOONSHOT_BASE_URL

if HTTP_PROXY:
    os.environ['HTTP_PROXY'] = HTTP_PROXY
if HTTPS_PROXY:
    os.environ['HTTPS_PROXY'] = HTTPS_PROXY

from gold_ai.service import market_service as ms
from gold_ai.service.memory_service import (
    save_message,
    extract_user_strategy,
    update_user_profile,
    get_recent_memory,
    get_user_profile_context,
)

client = OpenAI(
    api_key=MOONSHOT_API_KEY,
    base_url=MOONSHOT_BASE_URL
)

CHAT_SYSTEM_PROMPT = """你是一位资深的金融学专家，专攻宏观经济、货币政策、大宗商品和黄金交易领域。

你的专业背景：
- 拥有金融学博士学位，15年宏观对冲基金研究经验
- 精通黄金定价机制、实际利率模型、多资产联动分析
- 熟悉全球央行政策、地缘政治风险、资金流分析

你的对话风格：
1. 自然、专业但不故作高深，把复杂概念讲清楚
2. 用具体数据和逻辑支撑观点，不泛泛而谈
3. 简短问题简短回答，深入问题可以展开，但不要写成报告
4. 可以反问用户以确认其真正意图

你的核心能力——评估用户的观点和策略：
1. 当用户表达一个观点或策略时，你需要判断其对错
2. 如果用户的观点正确：明确肯定，并补充为什么对、有什么支撑逻辑
3. 如果用户的观点有误：先点出其中有道理的部分（如果有），再指出错误之处，用数据/逻辑/历史案例解释为什么不对
4. 如果用户的策略有漏洞：指出风险在哪里、什么情况下会失效
5. 纠错态度：专业平等，不是居高临下说教，而是像同行讨论一样

你的原则：
- 不确定的事情要坦诚说"这不在我掌握的信息范围内"或"这取决于以下因素"
- 不做具体的买卖建议（"你应该买入"），但可以分析"从什么角度看，这是否合理"
- 涉及预测时，给出概率而非绝对判断
- 回答中不要使用 markdown 格式（不用 #、**、--- 等），用自然文字表达
- 数学表达使用 Unicode 符号（× ÷ ≈ ± ¥ € £），禁止使用 LaTeX 语法（如 \\times \\frac \\cdot）
- 不要提"根据提供的数据"、"注意"、"系统提示"等暴露提示词的字眼，直接引用数字即可
- 可以用「」或【】标注关键词，但不要用 markdown 标记"""


def _get_market_snapshot():
    """获取简洁的市场快照（用于注入对话上下文），同时返回结构化数据"""
    try:
        # 用 5d 避免周末/假日只有 1 天数据导致空值
        gold_df = ms.gold.history(period="5d")
        usd_df = ms.meiyuanzhishu.history(period="5d")
        bond_df = ms.meizhaililv.history(period="5d")
        vix_df = ms.vix.history(period="5d")
        oil_df = ms.yuanyoujiage.history(period="5d")
        sp500_df = ms.biaopu500.history(period="5d")
        cny_df = ms.usdcny.history(period="5d")
        eur_df = ms.eurusd.history(period="5d")

        data = {
            "gold_price": None, "gold_change": None,
            "cny_rate": None, "eur_rate": None,
            "text": ""
        }
        parts = []
        fx_parts = []

        if not gold_df.empty and len(gold_df['Close']) >= 1:
            g = gold_df['Close']
            price = float(g.iloc[-1])
            data["gold_price"] = round(price, 2)
            if len(g) >= 2:
                chg = round((price - float(g.iloc[-2])) / float(g.iloc[-2]) * 100, 2)
                data["gold_change"] = chg
                parts.append(f"黄金 ${price:.0f}（日涨跌 {chg:+.2f}%）")
            else:
                parts.append(f"黄金 ${price:.0f}")

        if not usd_df.empty and len(usd_df['Close']) >= 1:
            parts.append(f"美元指数 {float(usd_df['Close'].iloc[-1]):.2f}")

        if not bond_df.empty and len(bond_df['Close']) >= 1:
            parts.append(f"美债10Y {float(bond_df['Close'].iloc[-1]):.2f}%")

        if not vix_df.empty and len(vix_df['Close']) >= 1:
            parts.append(f"VIX {float(vix_df['Close'].iloc[-1]):.2f}")

        if not oil_df.empty and len(oil_df['Close']) >= 1:
            parts.append(f"原油 ${float(oil_df['Close'].iloc[-1]):.0f}")

        if not sp500_df.empty and len(sp500_df['Close']) >= 1:
            parts.append(f"标普500 {float(sp500_df['Close'].iloc[-1]):.0f}")

        # 汇率
        if not cny_df.empty and len(cny_df['Close']) >= 1:
            data["cny_rate"] = round(float(cny_df['Close'].iloc[-1]), 4)
            fx_parts.append(f"美元/人民币 {data['cny_rate']}")

        if not eur_df.empty and len(eur_df['Close']) >= 1:
            data["eur_rate"] = round(float(eur_df['Close'].iloc[-1]), 4)
            fx_parts.append(f"欧元/美元 {data['eur_rate']}")

        if parts:
            text = "当前市场数据：" + "，".join(parts)
            if fx_parts:
                text += "\n当前汇率：" + "，".join(fx_parts)
                text += "\n（用户可能要求用人民币或欧元报价，请用以上汇率换算。例如：黄金 $2000 × 6.77 = ¥13540/盎司）"
            data["text"] = text
        return data
    except Exception:
        return {"gold_price": None, "gold_change": None, "cny_rate": None, "eur_rate": None, "text": ""}


_PRICE_QUERY_KEYWORDS = [
    "金价", "黄金价格", "黄金多少钱", "现在黄金", "当前金价", "最新金价",
    "黄金期货价格", "gold price", "黄金报价", "金价多少",
]


def _is_price_query(query: str) -> bool:
    """判断用户是否在询问当前金价"""
    q = query.lower()
    return any(kw in q for kw in _PRICE_QUERY_KEYWORDS)


def _clean_for_chat(text: str) -> str:
    """清洗 markdown 和 LaTeX，转为纯文本"""
    # --- Markdown ---
    text = re.sub(r'#{1,6}\s+', '', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'\n---+\n', '\n', text)
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'`(.+?)`', r'\1', text)

    # --- LaTeX / 数学符号 ---
    # 分数 \frac{a}{b} → (a/b)
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1/\2)', text)
    # 文本模式 \text{...} → ...
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
    # 根号 \sqrt{x} → √x  (简单情况)
    text = re.sub(r'\\sqrt\{([^}]+)\}', r'√\1', text)
    # 下标 _{}  → 直接连写
    text = re.sub(r'_\{(.+?)\}', r'\1', text)
    # 上标 ^{}  → ^...
    text = re.sub(r'\^\{(.+?)\}', r'^\1', text)
    # 常用符号
    text = re.sub(r'\\times', '×', text)
    text = re.sub(r'\\div', '÷', text)
    text = re.sub(r'\\cdot', '·', text)
    text = re.sub(r'\\approx', '≈', text)
    text = re.sub(r'\\pm', '±', text)
    text = re.sub(r'\\mp', '∓', text)
    text = re.sub(r'\\leq', '≤', text)
    text = re.sub(r'\\geq', '≥', text)
    text = re.sub(r'\\neq', '≠', text)
    text = re.sub(r'\\infty', '∞', text)
    text = re.sub(r'\\to', '→', text)
    text = re.sub(r'\\rightarrow', '→', text)
    text = re.sub(r'\\leftarrow', '←', text)
    text = re.sub(r'\\Rightarrow', '⇒', text)
    text = re.sub(r'\\yen\s*', '¥', text)
    text = re.sub(r'\\euro\s*', '€', text)
    text = re.sub(r'\\pound\s*', '£', text)
    text = re.sub(r'\\left', '', text)
    text = re.sub(r'\\right', '', text)
    # 清除反斜杠转义的标点: \% \# \_ \& 等
    text = re.sub(r'\\([%#_&])', r'\1', text)
    # 移除剩余的反斜杠命令
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    # 移除 $$ 和 $ 数学定界符（成对出现时）
    text = re.sub(r'\$\$(.+?)\$\$', r'\1', text)
    text = re.sub(r'\$(.+?)\$', r'\1', text)
    # 移除 \[ \] 和 \( \) 数学定界符
    text = re.sub(r'\\\[(.+?)\\\]', r'\1', text)
    text = re.sub(r'\\\((.+?)\\\)', r'\1', text)
    # 清理残留的 { } 大括号（通常是 LaTeX 分组）
    text = re.sub(r'\{([^}]*)\}', r'\1', text)

    # --- 空白整理 ---
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def build_chat_messages(db, user_id: int, user_query: str) -> list:
    """构建标准 OpenAI 多轮对话 messages 数组"""
    snapshot = _get_market_snapshot()

    messages = [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
    ]

    # 注入实时市场数据
    if snapshot["text"]:
        messages.append({
            "role": "system",
            "content": (
                f"以下是今日实时市场数据，请在对话中直接引用：\n"
                f"{snapshot['text']}\n"
                f"（这些数据是权威的当前值，回答时直接使用，不要引用训练数据中的旧价格）"
            )
        })

    # 注入用户画像
    profile = get_user_profile_context(db, user_id)
    if profile and profile != "暂无用户画像":
        messages.append({
            "role": "system",
            "content": f"当前用户画像（供参考，用于个性化对话）：{profile}"
        })

    # 加载历史对话（最近30条）
    history = get_recent_memory(db, user_id, limit=30)
    for h in history:
        messages.append(h)

    # 价格类查询：作为系统数据注入，而非用户消息前缀
    if _is_price_query(user_query) and snapshot["gold_price"]:
        messages.append({
            "role": "system",
            "content": (
                f"用户正在询问当前金价。当前黄金价格为 ${snapshot['gold_price']:.0f}，"
                f"日涨跌 {snapshot['gold_change']:+.2f}%。"
                f"美元/人民币汇率为 {snapshot['cny_rate']}。"
                f"请直接使用这些数据回答，用自然语言融入数字即可，不要写「根据数据」「系统提示」等。"
            )
        })

    # 追加当前用户消息
    messages.append({"role": "user", "content": user_query})

    return messages


def _extract_conversation_insights(user_query: str, ai_reply: str) -> dict:
    """从对话中提取用户关注的关键词和话题"""
    prompt = f"""从以下对话中提取用户关注的关键词和观点倾向。返回JSON，不要解释。

用户: {user_query[:300]}
AI回复: {ai_reply[:500]}

返回格式:
{{"keywords": ["关键词1", "关键词2"], "bias": "bullish/bearish/neutral", "topics": ["话题1"]}}

keywords: 用户当前关注的具体概念或指标（如"非农数据"、"实际利率"、"ETF"），只提取明确提到的
bias: 用户表达的市场倾向，没有明确表达则为"neutral"
topics: 对话涉及的宏观话题"""

    try:
        response = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )
        text = response.choices[0].message.content.strip()
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass
    return {"keywords": [], "bias": "neutral", "topics": []}


def sync_profile_insights(db, user_id: int, user_query: str, ai_reply: str):
    """从对话中提取洞察，同步到用户画像"""
    from gold_ai.models.user import UserProfile

    insights = _extract_conversation_insights(user_query, ai_reply)

    profile = db.query(UserProfile).filter_by(id=user_id).first()
    if not profile:
        profile = UserProfile(id=user_id)
        db.add(profile)

    # 合并关键词
    new_kws = insights.get("keywords", [])
    if new_kws:
        existing = profile.keywords or []
        for kw in new_kws:
            if kw not in existing:
                existing.append(kw)
        profile.keywords = existing[-50:]  # 最多保留50个

    # 更新市场倾向（仅当有明显信号时）
    bias = insights.get("bias", "neutral")
    if bias != "neutral":
        profile.market_bias = bias

    db.commit()


def ai_chat_conversation(db, user_id: int, query: str) -> dict:
    """对话式 AI 入口：金融专家，多轮记忆，不生成报告"""
    # 1. 保存用户消息
    save_message(db, user_id, "user", query)

    # 2. 提取策略信号，更新画像
    strategy = extract_user_strategy(query)
    update_user_profile(db, user_id, strategy)

    # 3. 构建多轮对话 messages
    messages = build_chat_messages(db, user_id, query)

    # 4. 调用 LLM
    response = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )

    ai_reply = response.choices[0].message.content.strip()
    ai_reply = _clean_for_chat(ai_reply)

    # 5. 保存 AI 回复
    save_message(db, user_id, "assistant", ai_reply)

    # 6. 同步对话洞察到用户画像
    try:
        sync_profile_insights(db, user_id, query, ai_reply)
    except Exception:
        pass

    return {
        "reply": ai_reply,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat()
    }
