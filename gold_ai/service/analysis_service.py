import datetime
import os

from gold_ai.config import HTTP_PROXY, HTTPS_PROXY, MOONSHOT_API_KEY, MOONSHOT_BASE_URL, MARKET_CACHE_TTL, NEWS_CACHE_TTL

if HTTP_PROXY:
    os.environ['HTTP_PROXY'] = HTTP_PROXY
if HTTPS_PROXY:
    os.environ['HTTPS_PROXY'] = HTTPS_PROXY

from gold_ai.service import market_service as ms
from gold_ai.service import news_service as ns
from gold_ai.service import usernews_service as uns
from gold_ai.service.memory_service import (
    save_message,
    extract_user_strategy,
    update_user_profile,
    get_recent_memory,
    get_user_profile_context,
)
from gold_ai.service.learning_service import (
    record_prediction,
    get_weights_context,
    update_weights
)
from gold_ai.service.evaluation_service import evaluate_predictions
from gold_ai.service.cache import TTLCache
from openai import OpenAI

client = OpenAI(
    api_key=MOONSHOT_API_KEY,
    base_url=MOONSHOT_BASE_URL
)

# TTL caches for aggregate data
_market_cache = TTLCache(ttl_seconds=MARKET_CACHE_TTL)
_news_cache = TTLCache(ttl_seconds=NEWS_CACHE_TTL)


# ==================== Markdown 清洗 ==================

def clean_markdown(text: str) -> str:
    """去除 AI 回复中的 markdown 标记，便于 TTS 朗读"""
    import re
    # 去除 # 标题标记，保留标题文字
    text = re.sub(r'#{1,6}\s+', '', text)
    # 去除 ** 加粗标记
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    # 去除 * 斜体标记
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    # 去除 --- 分隔线
    text = re.sub(r'\n---+\n', '\n', text)
    # 去除 > 引用标记
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    # 去除 ` 代码标记
    text = re.sub(r'`(.+?)`', r'\1', text)
    # 去除多余空行（3个以上换行 → 2个）
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def aggregate_context():
    # Check cache first
    cached = _market_cache.get("aggregate")
    if cached:
        return cached

    gold_df = ms.gold.history(period="5d")
    usd_df = ms.meiyuanzhishu.history(period="5d")
    bond_df = ms.meizhaililv.history(period="5d")
    oil_df = ms.yuanyoujiage.history(period="5d")
    vix_df = ms.vix.history(period="5d")

    gold_close = gold_df['Close'].tolist()
    usd_close = usd_df['Close'].tolist()
    bond_close = bond_df['Close'].tolist()
    oil_close = oil_df['Close'].tolist()
    vix_close = vix_df['Close'].tolist()

    latest_gold = gold_close[-1] if gold_close else None

    gold_change_1d = round((gold_close[-1] - gold_close[-2]) / gold_close[-2] * 100, 2) if len(gold_close) >= 2 else None
    gold_change_5d = round((gold_close[-1] - gold_close[0]) / gold_close[0] * 100, 2) if len(gold_close) >= 2 else None
    gold_high_5d = round(max(gold_close), 2) if gold_close else None
    gold_low_5d = round(min(gold_close), 2) if gold_close else None

    usd_latest = usd_close[-1] if usd_close else None
    usd_change = round((usd_close[-1] - usd_close[-2]) / usd_close[-2] * 100, 2) if len(usd_close) >= 2 else None

    bond_latest = bond_close[-1] if bond_close else None
    bond_change = round((bond_close[-1] - bond_close[-2]) / bond_close[-2] * 100, 2) if len(bond_close) >= 2 else None

    oil_latest = oil_close[-1] if oil_close else None
    oil_change = round((oil_close[-1] - oil_close[-2]) / oil_close[-2] * 100, 2) if len(oil_close) >= 2 else None

    vix_latest = vix_close[-1] if vix_close else None
    vix_change = round((vix_close[-1] - vix_close[-2]) / vix_close[-2] * 100, 2) if len(vix_close) >= 2 else None

    market_summary = {
        "黄金期货": f"${latest_gold} (日涨跌: {gold_change_1d}%, 5日涨跌: {gold_change_5d}%, 5日最高: ${gold_high_5d}, 5日最低: ${gold_low_5d})",
        "美元指数": f"{usd_latest} (日涨跌: {usd_change}%)" if usd_latest else "N/A",
        "美债10年收益率": f"{bond_latest}% (日变动: {bond_change}%)" if bond_latest else "N/A",
        "原油价格": f"${oil_latest} (日涨跌: {oil_change}%)" if oil_latest else "N/A",
        "VIX恐慌指数": f"{vix_latest} (日变动: {vix_change}%)" if vix_latest else "N/A",
    }

    # News with separate cache key
    news_cached = _news_cache.get("jinshi")
    if news_cached:
        gold_related = news_cached
    else:
        all_news = ns.get_jinshi_news(limit=20)
        classified_news = ns.classify_news(all_news)
        gold_related = [f"[{n['time']}] {n['content']}" for n in classified_news if n['gold_related']]
        _news_cache.set("jinshi", gold_related)

    result = (market_summary, gold_related[:8], latest_gold)
    _market_cache.set("aggregate", result)
    return result


def build_prompt(user_query, market_context, news_context, memory, profile, weights, prediction_history=""):
    formatted_market = "\n".join([f"  - {k}: {v}" for k, v in market_context.items()])
    formatted_news = "\n".join([f"  {n}" for n in news_context]) if news_context else "  暂无相关新闻"

    prompt = f"""# 黄金宏观交易分析 AI（专业版 V3）

你是一名专业的黄金宏观交易分析 AI。

**重要：先判断用户问题的复杂度，据此决定回答方式。**

- 简单事实性问题（查询价格、询问指标含义、确认数据、概念解释）→ 直接简洁回答，2-4句话，不要展开。
- 深度分析问题（走势分析、预测、操作建议、开放式行情讨论）→ 按下方完整模板输出。

---

你具备：

* 宏观经济分析能力
* 黄金市场分析能力
* 多资产联动分析能力
* 技术分析能力
* 资金流分析能力
* 量化概率分析能力
* 历史行情复盘能力
* 交易策略设计能力

你的任务不是总结新闻。

你的任务是：

像专业黄金交易员一样，

利用市场数据、宏观信息、资金流信息和历史经验，

分析黄金未来走势，

帮助用户进行投资决策。

---

# 输入信息

## 实时市场数据

{formatted_market}

## 黄金相关新闻

{formatted_news}

## 用户画像

{profile}

## 历史记忆

{memory}

## 模型权重

{weights}

## 历史预测记录

{prediction_history}

## 用户问题

{user_query}

---

# 分析原则（必须遵守）

## 原则1：价格优先

当新闻与价格冲突时：

优先相信市场价格。

分析：

市场正在交易什么逻辑。

不要简单复述新闻。

---

## 原则2：预期优先

市场交易未来预期。

必须区分：

* 实际值
* 市场预期
* 前值

重点分析：

预期差。

---

## 原则3：实际利率优先

黄金长期核心驱动：

实际利率

实际利率 = 美债收益率 − 通胀预期

判断：

实际利率下降 → 利多黄金

实际利率上升 → 利空黄金

---

## 原则4：资金验证原则

任何方向判断都必须验证：

* ETF资金流
* COMEX持仓
* CFTC持仓
* 央行购金

价格与资金一致：

提高置信度。

价格与资金背离：

降低置信度。

---

## 原则5：趋势优先

上涨趋势：

优先寻找做多机会。

下跌趋势：

优先寻找做空机会。

震荡趋势：

优先区间交易。

禁止逆势预测。

---

## 原则6：多因子共振

至少3个核心因素同向：

才允许给出强烈观点。

例如：

黄金看涨共振：

✓ 美元下跌

✓ 利率下降

✓ ETF流入

✓ 技术面突破

✓ 避险情绪升温

满足3项以上：

看涨置信度提升。

---

## 原则7：大周期优先

分析顺序：

月线
→ 周线
→ 日线
→ 小时线

大周期决定方向。

小周期决定入场点。

---

## 原则8：新闻衰减机制

新闻权重：

24小时内：100%

3天内：70%

7天内：40%

14天以上：10%

过期新闻不得作为核心依据。

---

## 原则9：历史相似行情

必须寻找：

* 降息周期
* 高通胀周期
* 银行业危机
* 战争避险
* 美元牛市

历史相似场景。

评估：

当前市场与历史环境相似度。

---

## 原则10：概率预测

禁止：

“必涨”
“必跌”

必须输出：

上涨概率

震荡概率

下跌概率

总计100%。

---

# 综合评分模型

请构建：

Gold Score

评分范围：

-100 ~ +100

评分逻辑：

实际利率：25%

美元指数：15%

ETF资金流：15%

CFTC持仓：10%

央行购金：10%

地缘政治：10%

技术面：10%

新闻事件：5%

评分解释：

60以上：

强看涨

20~60：

偏多

-20~20：

震荡

-60~-20：

偏空

-60以下：

强看空

输出：

Gold Score = X

并解释原因。

---

# 预测校验关卡（输出结论前必须逐项检查）

在给出最终预测前，必须进行以下5项一致性校验：

1. 宏观方向 vs 资金流方向 → 判定：一致 / 背离
2. 资金流 vs 价格走势 → 判定：确认 / 背离
3. 技术面 vs 预测方向 → 判定：支持 / 不支持
4. 历史相似场景 vs 预测方向 → 判定：支持 / 不支持
5. 是否存在重大矛盾信号 → 判定：有 / 无

校验规则：
- 4~5项一致 → 置信度不变，可给出明确建议
- 2~3项一致 → 置信度 × 0.7，必须标注矛盾点，建议谨慎
- 0~1项一致 → 置信度 × 0.3，明确建议观望，不做方向性预测

输出格式：
```
🔍 预测校验结果：
  1. 宏观 vs 资金流：一致/背离
  2. 资金流 vs 价格：确认/背离
  3. 技术面支持：是/否
  4. 历史场景支持：是/否
  5. 重大矛盾信号：有/无
  校验通过: X/5
  原始置信度 → 校准后置信度
```

若校验不通过项 ≥ 3，必须在回答中明确告知用户当前市场矛盾较大，建议观望。

---

# 输出格式

---

## 【简略回答】（必须放在最前面）

在进入完整报告之前，先用2-4句话直接回答用户的问题。

要求：
* 直接回应用户问什么
* 给出明确的方向判断或核心观点
* 引用1-2个关键数据支撑
* 不展开，不列模板章节

示例：
用户问"黄金会涨吗？" →
"当前黄金处于上涨趋势，美元指数走弱至103.5、实际利率下行形成双重支撑。短期大概率维持偏强格局，关注2080阻力位能否突破。但需警惕周五非农数据可能带来的扰动。"

---

## 一、市场核心结论

3句话以内。

必须明确：

* 当前趋势
* 当前主导逻辑
* 当前最大风险

禁止模糊表达。

---

## 二、Gold Score 综合评分

输出：

Gold Score：

上涨概率：

震荡概率：

下跌概率：

并解释：

评分来源。

---

## 三、宏观驱动分析

分析：

### 美元指数 DXY

* 当前影响
* 利多因素
* 利空因素
* 主导方向

### 美债收益率

* 当前影响
* 利多因素
* 利空因素
* 主导方向

### 美联储政策

分析：

* 降息预期
* 点阵图
* 官员讲话

### 通胀数据

分析：

* CPI
* PCE
* 核心通胀

### 就业数据

分析：

* 非农
* 失业率
* 薪资增长

### 地缘政治

分析：

* 战争
* 制裁
* 全球风险事件

最后总结：

当前市场正在交易什么逻辑。

---

## 四、多资产联动分析

分析：

### 黄金 VS 美元

### 黄金 VS 美债收益率

### 黄金 VS 原油

### 黄金 VS 标普500

### 黄金 VS 比特币

### 黄金 VS 白银

判断：

是否存在背离现象。

若存在：

解释原因。

---

## 五、市场情绪与资金流

分析：

### ETF资金流

### COMEX持仓

### CFTC持仓

### 央行购金

### VIX恐慌指数

判断：

当前市场属于：

* Risk-On
* Risk-Off

资金主要在交易：

* 避险逻辑
* 通胀逻辑
* 降息逻辑
* 技术面逻辑

给出结论。

---

## 六、技术面分析

必须输出：

当前价格：

5日最高：

5日最低：

20日均价：

50日均价：

200日均价：

EMA趋势：

MACD方向：

RSI：

ATR：

布林带位置：

分析：

* 是否超买
* 是否超卖
* 是否突破
* 是否形成趋势

---

## 七、关键价格区

输出：

第一支撑位：

第二支撑位：

第三支撑位：

第一阻力位：

第二阻力位：

第三阻力位：

说明：

这些位置形成原因。

---

## 八、历史相似行情分析

寻找历史案例。

例如：

* 2019降息周期
* 2020疫情避险
* 2022高通胀周期
* 2023银行业危机

输出：

相似度评分：

历史走势：

当前参考意义。

---

## 九、情景预测

### 基准情景（最高概率）

概率：

走势：

目标位：

触发条件：

失效条件：

---

### 乐观情景

概率：

走势：

目标位：

触发条件：

失效条件：

---

### 悲观情景

概率：

走势：

目标位：

触发条件：

失效条件：

---

## 十、交易策略建议

### （1）短线（日内）

建议：

入场区：

止损：

第一目标：

第二目标：

风险收益比：

---

### （2）波段（3-10天）

建议：

建仓区：

止损：

目标位：

风险收益比：

---

### （3）中长期配置

是否适合定投：

是否适合避险配置：

建议仓位：

配置理由：

---

## 十一、用户个性化建议

结合：

{profile}

和

{memory}

分析：

* 当前策略是否适合用户
* 是否应减仓
* 是否应加仓
* 是否适合追涨
* 是否应等待回调

给出明确建议。

---

## 十二、预测校验层

请检查：

1. 宏观结论是否与资金流一致
2. 资金流是否与价格一致
3. 技术面是否支持预测
4. 历史案例是否支持预测
5. 是否存在重大矛盾信号

输出：

预测可信度：

★★★★★

并说明原因。

---

## 十三、风险提示

重点关注：

* 非农就业数据
* CPI
* PCE
* 美联储议息会议
* FOMC纪要
* 美债收益率变化
* 地缘政治风险

说明：

这些事件为何可能改变黄金走势。

---

# 输出风格要求

必须像：

宏观对冲基金分析师

机构黄金交易员

量化研究员

的综合体。

要求：

* 先给结论
* 数据驱动
* 明确方向
* 明确概率
* 明确风险
* 不要复述新闻
* 不要空泛表达
* 解释市场正在交易什么逻辑
* 给出可执行交易建议
* 所有预测必须附带触发条件与失效条件
"""
    return prompt


def extract_prediction(text):
    text = text.lower()
    if "上涨" in text or "看涨" in text:
        return "bullish", 0.7
    elif "下跌" in text or "看跌" in text:
        return "bearish", 0.7
    else:
        return "neutral", 0.5


def generate_ai_report(db, user_id: int, query: str):
    """生成结构化 AI 分析报告（13章节完整模板）"""
    # 0. 先评估历史预测（闭环）
    evaluate_predictions(db, hours_delay=1)

    # 1. 记录用户输入
    save_message(db, user_id, "user", query)

    # 2. 提取用户策略并更新画像
    strategy = extract_user_strategy(query)
    update_user_profile(db, user_id, strategy)

    # 获取上下文
    memory = get_recent_memory(db, user_id)
    profile = get_user_profile_context(db, user_id)
    weights = get_weights_context(db, user_id)

    # 3. 聚合市场与新闻数据
    market, news, gold_price = aggregate_context()
    user_news = uns.get_personalized_news(db, user_id, query, client)
    personal_news = [n["content"] for n in user_news["news"]]

    # 4. 获取历史预测记录
    from gold_ai.models.prediction import PredictionRecord
    past_records = db.query(PredictionRecord)\
        .filter_by(user_id=user_id)\
        .order_by(PredictionRecord.created_at.desc())\
        .limit(5).all()
    if past_records:
        lines = []
        for r in past_records:
            status = ""
            if r.is_correct is not None:
                status = "✓正确" if r.is_correct == 1 else "✗错误"
            lines.append(f"预测: {r.prediction} | 置信度: {r.confidence} | 预测价: {r.predicted_price} | 实际价: {r.actual_price} | {status}")
        prediction_history = "\n".join(lines)
    else:
        prediction_history = "暂无历史预测"

    # 5. 构建完整分析 prompt
    prompt = build_prompt(query, market, news + personal_news, memory, profile, weights, prediction_history)

    # 6. 调用 LLM（32k 模型，深度分析）
    response = client.chat.completions.create(
        model="moonshot-v1-32k",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    ai_reply = response.choices[0].message.content.strip()
    # 报告保留原始 markdown 格式，不调用 clean_markdown

    save_message(db, user_id, "assistant", ai_reply)
    prediction, confidence = extract_prediction(ai_reply)

    record_id = record_prediction(
        db,
        user_id=user_id,
        prediction=prediction,
        price=gold_price,
        confidence=confidence
    )

    update_weights(db, user_id)

    return {
        "type": "report",
        "reply": ai_reply,
        "prediction": prediction,
        "confidence": confidence,
        "record_id": record_id,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat()
    }
