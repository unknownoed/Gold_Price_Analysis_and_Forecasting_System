"""twelve_data 数据获取示例 —— 现已整合到 market_service 中。

直接从 market_service 导入 UltimateFinancialFetcher 即可使用。
"""

from gold_ai.service.market_service import UltimateFinancialFetcher, fetcher

# ==================== 运行测试 ====================
if __name__ == "__main__":
    # 使用模块级 fetcher（从 config/settings.json/env 读取 FRED_API_KEY）
    print("使用模块级 fetcher（FRED key 来自配置）:")
    gold = fetcher.fetch_via_yfinance("GC=F", "黄金期货")
    crude = fetcher.fetch_via_yfinance("CL=F", "原油期货")
    sp500 = fetcher.fetch_via_yfinance("^SPX", "标普500指数")
    dxy = fetcher.fetch_via_yfinance("DX-Y.NYB", "美元指数")
    vix = fetcher.fetch_via_yfinance("^VIX", "VIX恐慌指数")
    gld_etf = fetcher.fetch_via_yfinance("GLD", "GLD ETF")

    import time
    time.sleep(3)

    us10y = fetcher.fetch_fred("DGS10", "10年美债收益率")
    cpi = fetcher.fetch_fred("CPIAUCSL", "CPI数据")
    non_farm = fetcher.fetch_fred("PAYEMS", "非农数据")

    print("\n====== 抓取完成 ======")
    if gold is not None:
        print("黄金最新报价样例:\n", gold[['Close', 'Volume']].tail(2))
