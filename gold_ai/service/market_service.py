import os
import time
import requests
import pandas as pd
import yfinance as yf

from gold_ai.config import HTTP_PROXY, HTTPS_PROXY, FRED_API_KEY

if HTTP_PROXY:
    os.environ['HTTP_PROXY'] = HTTP_PROXY
if HTTPS_PROXY:
    os.environ['HTTPS_PROXY'] = HTTPS_PROXY

# ==================== Yahoo Finance Ticker 对象 ====================
gold = yf.Ticker("GC=F")
meiyuanzhishu = yf.Ticker("DX-Y.NYB")
meizhaililv = yf.Ticker("^TNX")
biaopu500 = yf.Ticker("^GSPC")
yuanyoujiage = yf.Ticker("CL=F")
vix = yf.Ticker("^VIX")
gld_etf = yf.Ticker("GLD")

# 常用汇率
usdcny = yf.Ticker("CNY=X")
eurusd = yf.Ticker("EUR=X")


def get_fx_rate(ticker, period="2d"):
    """获取汇率数据，返回 DataFrame"""
    return ticker.history(period=period)


def get_market_history(symbol, period="1mo", interval="1d"):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        return df
    df.index = df.index.strftime('%Y-%m-%d')
    return df.reset_index()


class UltimateFinancialFetcher:
    """多数据源金融数据获取器：yfinance + FRED"""

    def __init__(self, fred_key=None):
        self.fred_key = fred_key or FRED_API_KEY
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def fetch_via_yfinance(self, symbol, name, period="3mo"):
        print(f"[+] 正在从 Yahoo Finance 获取 {name} ({symbol})...")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            if df.empty:
                print(f"[-] {name} 数据返回为空")
                return None
            return df
        except Exception as e:
            print(f"[-] {name} 获取异常: {e}")
            return None

    def fetch_fred(self, series_id, name):
        if not self.fred_key:
            print(f"[-] 未提供 FRED Key，跳过 {name}")
            return None

        print(f"[+] 正在从 FRED 获取 {name} ({series_id})...")
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {"series_id": series_id, "api_key": self.fred_key, "file_type": "json"}
        try:
            res = requests.get(url, params=params).json()
            if "observations" in res:
                df = pd.DataFrame(res["observations"])
                df = df[df['value'] != '.']
                df['date'] = pd.to_datetime(df['date'])
                df['value'] = df['value'].astype(float)
                return df.set_index('date')['value']
        except Exception as e:
            print(f"[-] FRED {name} 请求异常: {e}")
        return None

    def get_all_market_data(self):
        """一次性获取所有市场数据（yfinance + FRED）"""
        results = {}

        results['gold'] = self.fetch_via_yfinance("GC=F", "黄金期货")
        time.sleep(1)
        results['crude_oil'] = self.fetch_via_yfinance("CL=F", "原油期货")
        time.sleep(1)
        results['sp500'] = self.fetch_via_yfinance("^SPX", "标普500指数")
        time.sleep(1)
        results['dxy'] = self.fetch_via_yfinance("DX-Y.NYB", "美元指数")
        time.sleep(1)
        results['vix'] = self.fetch_via_yfinance("^VIX", "VIX恐慌指数")
        time.sleep(1)
        results['gld_etf'] = self.fetch_via_yfinance("GLD", "GLD ETF")
        time.sleep(2)

        results['us10y'] = self.fetch_fred("DGS10", "10年美债收益率")
        results['cpi'] = self.fetch_fred("CPIAUCSL", "CPI数据")
        results['non_farm'] = self.fetch_fred("PAYEMS", "非农数据")

        return results


fetcher = UltimateFinancialFetcher()
