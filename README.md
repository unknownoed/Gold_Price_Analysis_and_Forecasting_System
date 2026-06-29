# Gold Price Analysis and Forecasting System

黄金价格分析与预测系统，整合市场数据、新闻分析和 AI 大模型，提供宏观交易分析。

## 环境配置

### 1. 克隆项目

```bash
git clone https://github.com/unknownoed/Gold_Price_Analysis_and_Forecasting_System.git
cd Gold_Price_Analysis_and_Forecasting_System
```

### 2. 创建虚拟环境并安装依赖

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
# 或
venv\Scripts\activate      # Windows

pip install -r gold_ai/requirements.txt
```

### 3. 配置 API 密钥

```bash
cp gold_ai/.env.example gold_ai/.env
```

编辑 `gold_ai/.env`，填入你的 API 密钥：

| 变量 | 说明 | 获取地址 |
|------|------|----------|
| `MOONSHOT_API_KEY` | Kimi 大模型 API | https://platform.moonshot.cn |
| `TAVILY_API_KEY` | 新闻搜索 API | https://tavily.com |
| `SERPAPI_KEY` | 搜索引擎 API | https://serpapi.com |
| `DEEPSEEK_API_KEY` | DeepSeek 大模型（可选） | https://platform.deepseek.com |
| `FRED_API_KEY` | 美联储经济数据（可选） | https://fred.stlouisfed.org |

或者通过 Web 界面设置：启动后在 Settings 页面填入 API Key，保存到 `settings.json`。

### 4. 初始化数据库并启动

```bash
# 初始化数据库
cd gold_ai/db && python init_db.py && cd ../..

# 启动应用（必须在项目根目录运行）
python gold_ai/app.py
```

访问 http://localhost:5000

## 项目结构

```
gold_ai/
├── app.py              # Flask 主应用
├── config.py           # 配置加载（env / settings.json）
├── db/                 # 数据库层
├── models/             # SQLAlchemy 数据模型
├── service/            # 业务逻辑层
│   ├── analysis_service.py  # AI 分析
│   ├── chat_service.py      # 对话服务
│   ├── market_service.py    # 市场数据
│   ├── news_service.py      # 新闻服务
│   ├── tts_service.py       # 语音合成
│   └── ...
└── templates/          # Jinja2 页面模板
```
