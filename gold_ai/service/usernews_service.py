from sqlalchemy.orm import Session
from gold_ai.models import UserProfile
from gold_ai.service.news_service import classify_news
from gold_ai.config import (
    MOONSHOT_API_KEY, MOONSHOT_BASE_URL,
    TAVILY_API_KEY, SERPAPI_KEY,
    DEEPSEEK_BASE_URL
)
from openai import OpenAI
import requests
import json
import re
from bs4 import BeautifulSoup
from gold_ai.service import market_service as ms

# --- Shared Moonshot client ---
client = OpenAI(
    api_key=MOONSHOT_API_KEY,
    base_url=MOONSHOT_BASE_URL
)

AGENT_MAX_TURNS = 5


def extract_user_intent_llm(user_query: str, history_keywords: list = None, llm_client=None):
    if llm_client is None:
        llm_client = client

    history_context = ""
    if history_keywords and len(history_keywords) > 0:
        recent_keywords = history_keywords[-10:] if len(history_keywords) > 10 else history_keywords
        history_context = f"""
        用户历史关注的关键词：{', '.join(recent_keywords)}
        这些是用户之前感兴趣的金融话题，请结合这些历史兴趣和当前输入来理解用户意图。
        """

    prompt = f"""
    你是一个金融信息分析助手，专注于黄金市场及相关领域的信息检索。

    任务：
    从用户输入中提取与"黄金市场"或用户感兴趣的金融信息相关的检索意图。

    请提取以下内容并以JSON格式返回：
    1. keywords: 核心检索关键词（3-8个，覆盖用户关注的主要方面）
    2. categories: 内容类别（如：宏观经济、货币政策、地缘政治、市场情绪、供需关系、避险需求、技术分析等）
    3. intent: 用户意图（如：查找新闻、分析趋势、了解原因、获取数据等）
    4. query_expansion: 搜索引擎优化查询（2-3个扩展搜索短语，用于提高召回率）

    要求：
    - 不要过度限定在"金价影响因素"，保持开放性
    - 提取用户真正感兴趣的内容方向
    - 只返回JSON，不要解释

    用户输入：
    {user_query}
    """

    response = llm_client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {
            "keywords": user_query.split(),
            "query_expansion": [user_query]
        }


def update_user_keywords(db: Session, user_id, new_keywords):
    user = db.query(UserProfile).filter_by(id=user_id).first()

    if not user:
        user = UserProfile(id=user_id, keywords=new_keywords)
        db.add(user)
    else:
        existing = user.keywords or []
        user.keywords = list(set(existing + new_keywords))

    db.commit()
    return user.keywords


def search_tavily(query):
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "advanced"
    }
    response = requests.post(url, json=payload, timeout=15)
    data = response.json()

    results = []
    for item in data.get("results", []):
        results.append({
            "title": item.get("title"),
            "content": item.get("content"),
            "url": item.get("url")
        })
    return results


def search_serpapi(query):
    url = "https://serpapi.com/search.json"
    params = {
        "q": query,
        "api_key": SERPAPI_KEY,
        "engine": "google",
        "tbm": "nws"
    }
    response = requests.get(url, params=params, timeout=15)
    data = response.json()

    results = []
    for item in data.get("news_results", []):
        results.append({
            "title": item.get("title"),
            "content": item.get("snippet"),
            "url": item.get("link")
        })
    return results


def search_news_multi_engine(query):
    results = []

    try:
        results.extend(search_tavily(query))
    except Exception as e:
        print("Tavily error:", e)

    try:
        results.extend(search_serpapi(query))
    except Exception as e:
        print("SerpAPI error:", e)

    return results


def score_news(news, keywords):
    content = (news.get("content") or "").lower()
    score = 0
    for kw in keywords:
        if kw.lower() in content:
            score += 1
    return score


def get_personalized_news(db, user_id, query, llm_client=None):
    if llm_client is None:
        llm_client = client

    user = db.query(UserProfile).filter_by(id=user_id).first()
    history_keywords = user.keywords if user else []

    parsed = extract_user_intent_llm(query, history_keywords, llm_client)
    keywords = parsed.get("keywords", [])
    search_query = " ".join(parsed.get("query_expansion", [query]))

    all_keywords = update_user_keywords(db, user_id, keywords)

    news_list = search_news_multi_engine(search_query)

    scored = []
    for n in news_list:
        s = score_news(n, all_keywords)
        if s > 0:
            n["score"] = s
            scored.append(n)

    scored.sort(key=lambda x: x["score"], reverse=True)

    top_news = scored[:10]
    if top_news:
        top_news = classify_news(top_news)

    return {
        "parsed": parsed,
        "keywords": all_keywords,
        "news": top_news
    }


# ==================== Agent 搜索（Function Calling）====================

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": "搜索与黄金市场相关的新闻。可以搜索多轮，使用不同的查询词获取更全面的信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询词，中英文皆可，如 '美联储加息 黄金' 或 'gold price Fed rate'"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_article",
            "description": "获取指定 URL 的文章全文内容，用于深入了解某条新闻的细节。只在需要更多上下文时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "文章的完整 URL"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_snapshot",
            "description": "获取当前黄金、美元指数、美债、原油等实时市场数据，用于辅助分析新闻背景。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


def _tool_search_news(query: str) -> list:
    """Execute multi-engine search; returns (formatted_text, raw_results) to avoid double call."""
    results = search_news_multi_engine(query)
    if not results:
        return "未找到相关新闻。", []
    lines = []
    for i, r in enumerate(results[:8]):
        lines.append(f"[{i+1}] {r['title']}\n   {r['content'][:200]}\n   URL: {r['url']}")
    return "\n\n".join(lines), results


def _tool_fetch_article(url: str) -> str:
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        content = "\n".join(lines[:80])
        return content[:3000] if content else "无法提取文章内容"
    except Exception as e:
        return f"获取文章失败: {str(e)}"


def _tool_get_market_snapshot() -> str:
    try:
        gold = ms.gold.history(period="1d")['Close'].iloc[-1]
        usd = ms.meiyuanzhishu.history(period="1d")['Close'].iloc[-1]
        bond = ms.meizhaililv.history(period="1d")['Close'].iloc[-1]
        oil = ms.yuanyoujiage.history(period="1d")['Close'].iloc[-1]
        vix = ms.vix.history(period="1d")['Close'].iloc[-1]
        return f"黄金期货: ${gold:.2f}, 美元指数: {usd:.2f}, 美债10Y: {bond:.2f}%, 原油: ${oil:.2f}, VIX: {vix:.2f}"
    except Exception as e:
        return f"获取市场数据失败: {str(e)}"


def _execute_tool(name: str, args: dict):
    """Execute tool; for search_news returns (text, raw_results), otherwise returns str."""
    if name == "search_news":
        return _tool_search_news(args["query"])
    elif name == "fetch_article":
        return _tool_fetch_article(args["url"])
    elif name == "get_market_snapshot":
        return _tool_get_market_snapshot()
    return f"未知工具: {name}"


def _parse_agent_response(content: str, news_items: list) -> dict:
    summary = content
    sources = []
    url_pattern = re.findall(r'https?://[^\s\)\]】,，]+', content)
    seen_urls = set()
    for u in url_pattern:
        if u not in seen_urls:
            seen_urls.add(u)
            sources.append({"url": u, "title": "", "credibility": "medium"})

    for n in news_items[:5]:
        if n.get("url") and n["url"] not in {s["url"] for s in sources}:
            sources.append({
                "title": n.get("title", ""),
                "url": n["url"],
                "credibility": "medium"
            })

    if news_items and "categories" not in news_items[0]:
        news_items = classify_news(news_items)

    return {
        "summary": summary,
        "sources": sources[:8],
        "news": news_items[:10],
        "search_steps": []
    }


def agent_search_news(db, user_id: int, query: str, api_key: str) -> dict:
    agent_client = OpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL
    )

    user = db.query(UserProfile).filter_by(id=user_id).first()
    history_keywords = user.keywords if user else []

    system_prompt = f"""你是一个专业的金融新闻搜索助手，专注于黄金市场及相关领域。

当前用户的兴趣关键词: {', '.join(history_keywords[-10:]) if history_keywords else '无历史记录'}

你的工作流程:
1. 理解用户问题，确定需要搜索的方向
2. 使用 search_news 工具搜索相关信息（可以多次搜索不同角度）
3. 如果某条新闻的摘要信息不够，可以用 fetch_article 获取全文
4. 如需了解当前市场背景，使用 get_market_snapshot
5. 综合所有信息后，给出有引用标注的分析摘要

输出要求（当你不再需要调用工具时，按以下格式输出最终回答）:
- 用 3-5 段话总结关键发现
- 每条关键信息后面标注来源编号，如 [1]、[2]
- 在回答末尾列出所有引用来源的 URL
- 评估信息的整体可信度
- 如果搜索结果不相关或不足，诚实告知

用户问题: {query}"""

    messages = [{"role": "system", "content": system_prompt}]

    all_news = []
    search_steps = []

    for turn in range(AGENT_MAX_TURNS):
        response = agent_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=AGENT_TOOLS,
            tool_choice="auto",
            temperature=0.3
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                func_name = tc.function.name
                func_args = json.loads(tc.function.arguments)
                step_desc = f"调用 {func_name}({json.dumps(func_args, ensure_ascii=False)[:80]})"
                search_steps.append(step_desc)

                result = _execute_tool(func_name, func_args)

                if func_name == "search_news":
                    # _tool_search_news returns (text, raw_results) tuple
                    tool_text, raw_results = result
                    all_news.extend(raw_results)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_text
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result
                    })
        else:
            content = msg.content or ""
            result = _parse_agent_response(content, all_news)
            result["search_steps"] = search_steps
            return result

    return {
        "summary": "搜索轮次已达上限，以下是已收集的信息。请尝试更具体的搜索词。",
        "sources": [],
        "news": classify_news(all_news)[:10] if all_news else [],
        "search_steps": search_steps
    }
