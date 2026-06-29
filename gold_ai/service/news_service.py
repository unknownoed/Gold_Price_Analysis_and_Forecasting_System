import os
import re
import requests
import json
from datetime import datetime

from gold_ai.config import HTTP_PROXY, HTTPS_PROXY, MOONSHOT_API_KEY, MOONSHOT_BASE_URL

if HTTP_PROXY:
    os.environ['HTTP_PROXY'] = HTTP_PROXY
if HTTPS_PROXY:
    os.environ['HTTPS_PROXY'] = HTTPS_PROXY

# --- HTML 清洗 ---

def strip_html(text: str) -> str:
    """去除 HTML 标签，保留纯文本"""
    # 去除 <br/> <br> 等换行标签
    text = re.sub(r'<br\s*/?\s*>', '\n', text)
    # 去除所有其他 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # 去除 HTML 实体
    text = text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&quot;', '"').replace('&#39;', "'")
    # 合并多余空白和换行
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


# --- 金十数据获取 ---

def get_jinshi_news(limit=30):
    url = "https://flash-api.jin10.com/get_flash_list"
    headers = {
        "x-app-id": "SO1EJGmNgCtmpcPF",
        "x-version": "1.0.0"
    }
    params = {
        "max_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "channel": "-8200"
    }
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
    except Exception:
        return []

    news_list = []
    for item in data.get('data', [])[:limit]:
        content = item.get('data', {}).get('content', '')
        if content:
            content = strip_html(content)
            if not content.strip():
                continue
            news_list.append({
                'time': item.get('time'),
                'content': content,
                'important': item.get('important', 0)
            })
    return news_list


# --- LLM 新闻分类 ---

CLASSIFICATION_CATEGORIES = {
    '美联储_货币政策': '美联储利率决议、鲍威尔讲话、FOMC会议、加息/降息、缩表/QE、点阵图、鹰派鸽派表态',
    '通胀数据': 'CPI、PCE、PPI、通胀预期、物价指数、核心通胀',
    '就业数据': '非农就业、失业率、初请失业金、ADP就业、JOLTS职位空缺、薪资增长',
    '经济数据': 'GDP、PMI、零售销售、消费者信心、工业产出、房地产数据、贸易数据',
    '美元指数': '美元指数DXY、美元走强/走弱、汇率变动、去美元化',
    '地缘政治': '战争冲突、国际制裁、贸易战、关税、中东局势、俄乌、台海、避险事件',
    '央行政策': '央行购金/售金、外汇储备变动、各国央行政策（非美联储）、黄金储备',
    '市场情绪': 'VIX恐慌指数、避险情绪、风险偏好、资金流向、ETF持仓、多头空头',
    '大宗商品': '原油价格、铜价、白银、铂金、钯金、大宗商品走势',
    '股市相关': '美股、A股、港股、欧股、全球股市、企业财报',
    '加密货币': '比特币、以太坊、加密货币、数字资产',
    '其他': '不属于以上任何分类的内容'
}

_category_names = list(CLASSIFICATION_CATEGORIES.keys())
_category_descriptions = "\n".join([f"- {k}: {v}" for k, v in CLASSIFICATION_CATEGORIES.items()])


def _classify_with_llm(news_list: list) -> list:
    """使用 Moonshot 大模型对新闻进行批量分类"""
    if not news_list:
        return news_list

    # 构建批量分类请求
    items_text = []
    for i, news in enumerate(news_list):
        items_text.append(f"[{i}] {news['content'][:200]}")

    numbered_news = "\n".join(items_text)

    prompt = f"""你是一个金融新闻分类助手。请对以下每条新闻进行分类，从预定义类别中选择最匹配的标签。

预定义类别：
{_category_descriptions}

分类规则：
1. 每条新闻可以属于1-2个类别
2. 如果完全不属于前11个类别，则标记为"其他"
3. 只返回JSON数组，格式：[["类别1"], ["类别2"], ...]，按照新闻编号顺序

待分类新闻（共{len(news_list)}条）：
{numbered_news}

请直接返回JSON数组，不要任何解释："""

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=MOONSHOT_API_KEY,
            base_url=MOONSHOT_BASE_URL
        )
        response = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000
        )
        result_text = response.choices[0].message.content.strip()
        # 提取JSON数组
        json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
        if json_match:
            categories_list = json.loads(json_match.group())
            for i, cats in enumerate(categories_list):
                if i < len(news_list):
                    valid_cats = [c for c in cats if c in _category_names]
                    if not valid_cats:
                        valid_cats = ["其他"]
                    news_list[i]['categories'] = valid_cats
        else:
            raise ValueError("No JSON array found in response")

        # 确保所有新闻都有分类（遗漏的标记为"其他"）
        for news in news_list:
            if not news.get('categories'):
                news['categories'] = ['其他']
            news['gold_related'] = any(
                c in ['美联储_货币政策', '通胀数据', '就业数据', '经济数据',
                       '美元指数', '地缘政治', '央行政策', '大宗商品']
                for c in news['categories']
            )
    except Exception as e:
        print(f"LLM分类失败，回退到关键词匹配: {e}")
        _classify_with_keywords(news_list)

    return news_list


def _classify_with_keywords(news_list: list) -> list:
    """关键词匹配分类（作为LLM分类的fallback）"""
    categories = {
        '美联储_货币政策': [
            '美联储', 'Fed', '鲍威尔', '加息', '降息', '利率决议', '点阵图', '缩表',
            '量化宽松', 'QE', '量化紧缩', 'QT', '联邦基金利率', 'FOMC', '鹰派', '鸽派',
            '货币宽松', '货币紧缩', '利率不变', '加息预期', '降息预期'
        ],
        '通胀数据': [
            'CPI', 'PCE', '通胀', '消费者物价', 'PPI', '核心通胀', '物价指数',
            '通胀预期', '通胀数据', '物价上涨', '通胀压力'
        ],
        '就业数据': [
            '非农', '失业率', '初请', 'ADP', '就业人数', '劳动力', '薪资增长',
            '就业市场', '裁员', '招聘', '职位空缺', 'JOLTS'
        ],
        '经济数据': [
            'GDP', 'PMI', '零售销售', '消费者信心', '工业产出', '制造业',
            '服务业', '经济增速', '经济衰退', '贸易逆差', '进出口',
            '新屋开工', '成屋销售', '耐用品订单'
        ],
        '美元指数': [
            '美元指数', 'DXY', '美元走强', '美元走弱', '美元汇率',
            '强势美元', '弱势美元', '去美元化', '美元霸权'
        ],
        '地缘政治': [
            '战争', '冲突', '危机', '地缘', '袭击', '避险', '紧张局势', '制裁',
            '伊朗', '俄罗斯', '乌克兰', '中东', '以色列', '哈马斯',
            '朝鲜', '台海', '南海', '北约', '导弹', '军事', '轰炸', '入侵',
            '政变', '动荡', '谈判', '停火', '贸易战', '关税',
            '红海', '胡塞', '也门', '叙利亚'
        ],
        '央行政策': [
            '央行', '黄金储备', '欧洲央行', '日本央行', '中国人民银行', '英国央行',
            '购金', '抛售黄金', '外汇储备', '增持黄金', '减持美债'
        ],
        '市场情绪': [
            '恐慌', '避险', '风险偏好', 'VIX', '抛售', '熔断', '暴跌', '暴涨',
            '波动率', '资金流入', '资金流出', '持仓', 'ETF', '多头', '空头',
            '投机', '获利了结', '追涨', '抄底', '泡沫', '崩盘'
        ],
        '大宗商品': [
            '原油', '油价', '铜价', '白银', '铂金', '钯金', '大宗商品',
            'OPEC', '石油', '天然气', '能源'
        ],
        '股市相关': [
            '美股', 'A股', '港股', '欧股', '股市', '标普', '纳斯达克', '道指',
            '恒生', '上证', '深证', '个股', '板块', 'IPO', '财报', '盈利'
        ],
        '加密货币': [
            '比特币', 'BTC', '以太坊', 'ETH', '加密货币', '数字资产', '区块链',
            'DeFi', '稳定币', '狗狗币'
        ]
    }

    for news in news_list:
        content = news['content']
        content_lower = content.lower()
        news['categories'] = []
        for cat, keywords in categories.items():
            for kw in keywords:
                if kw.lower() in content_lower:
                    news['categories'].append(cat)
                    break
        if not news['categories']:
            news['categories'] = ['其他']
        news['gold_related'] = any(
            c in ['美联储_货币政策', '通胀数据', '就业数据', '经济数据',
                   '美元指数', '地缘政治', '央行政策', '大宗商品']
            for c in news['categories']
        )

    return news_list


def classify_news(news_list: list) -> list:
    """新闻分类主入口：优先使用LLM分类，失败则回退关键词匹配"""
    if not news_list:
        return news_list
    return _classify_with_llm(news_list)


def get_categories() -> list:
    """返回所有预定义分类（供前端筛选使用）"""
    return list(CLASSIFICATION_CATEGORIES.keys())
