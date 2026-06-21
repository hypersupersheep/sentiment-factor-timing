# -*- coding: utf-8 -*-
"""
通达信(TDX)数据接口封装
========================
为什么单独封装一层：
1. pytdx 原始接口按"页"取数(每页最多800条)，需要手动翻页拼接，封装后一行拉全量；
2. 通达信免费行情服务器不稳定，封装自动遍历服务器列表选可用的；
3. 加本地 CSV 缓存(cache/ 目录)，回测反复运行时不重复请求，对服务器友好也更快。

四个 *_research 文件夹下的回测脚本统一 import 本模块取数。

数据范围说明(实测)：
- 指数日线：2005-01 至今全量可得(沪深300/中证500/上证综指/券商指数399975等)
- 指数1分钟线：标准行情服务器仅保留最近约4-5个月
- 个股日线：同样全量可得
"""

import os
import socket
import datetime as dt

import pandas as pd

socket.setdefaulttimeout(10)  # pytdx 底层 socket 不设超时会死等，必须全局兜底

from pytdx.hq import TdxHq_API

# 实测可用性较好的免费行情服务器，连接时按顺序尝试
TDX_SERVERS = [
    ("115.238.56.198", 7709),
    ("180.153.18.170", 7709),
    ("119.97.185.59", 7709),
    ("218.108.98.244", 7709),
    ("121.36.81.195", 7709),
    ("124.71.187.122", 7709),
    ("119.147.212.81", 7709),
]

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# K线周期编号(pytdx约定): 8=1分钟K线, 9=日K线
CATEGORY_MIN1 = 8
CATEGORY_DAY = 9

_API = None  # 模块级单例连接，避免每次取数重连


def get_api():
    """连接通达信行情服务器，返回已连接的 API 单例。"""
    global _API
    if _API is not None:
        return _API
    api = TdxHq_API()
    for ip, port in TDX_SERVERS:
        try:
            if api.connect(ip, port, time_out=5):
                print(f"[tdx_data] 已连接行情服务器 {ip}:{port}")
                _API = api
                return _API
        except Exception:
            continue
    raise ConnectionError("所有通达信行情服务器均连接失败，请检查网络")


def _fetch_bars(market, code, category, is_index, max_pages=60):
    """翻页拉取K线直到服务器无更多数据，返回按时间升序的 DataFrame。

    market: 0=深交所 1=上交所；is_index: True 走指数接口，False 走个股接口。
    每页800条是协议上限；max_pages=60 足够覆盖2005年以来的日线。
    """
    api = get_api()
    fetch = api.get_index_bars if is_index else api.get_security_bars
    chunks = []
    for page in range(max_pages):
        bars = fetch(category, market, code, page * 800, 800)
        if not bars:
            break
        chunks.append(api.to_df(bars))
        if len(bars) < 800:  # 不满一页说明已到最早数据
            break
    if not chunks:
        raise ValueError(f"未取到数据: market={market} code={code} category={category}")
    df = pd.concat(chunks[::-1], ignore_index=True)  # 翻页是从新到旧，倒序拼回时间升序
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = (
        df[["datetime", "open", "high", "low", "close", "vol", "amount"]]
        .drop_duplicates(subset="datetime")
        .sort_values("datetime")
        .set_index("datetime")
    )
    return df


def _cached(name, loader, refresh=False, staleness_days=3):
    """简单的 CSV 缓存层：缓存文件足够新就直接读，否则调用 loader 拉数并落盘。"""
    path = os.path.join(CACHE_DIR, name + ".csv")
    if not refresh and os.path.exists(path):
        age = dt.datetime.now() - dt.datetime.fromtimestamp(os.path.getmtime(path))
        if age.days < staleness_days:
            return pd.read_csv(path, index_col=0, parse_dates=True)
    df = loader()
    df.to_csv(path)
    return df


def index_daily(code, market, refresh=False):
    """指数日K线全量。例: index_daily('000300', 1) 沪深300。"""
    return _cached(
        f"idx_{market}_{code}_day",
        lambda: _fetch_bars(market, code, CATEGORY_DAY, is_index=True),
        refresh,
    )


def index_min1(code, market, refresh=False):
    """指数1分钟K线(服务器仅保留最近数月)。"""
    return _cached(
        f"idx_{market}_{code}_min1",
        lambda: _fetch_bars(market, code, CATEGORY_MIN1, is_index=True),
        refresh,
        staleness_days=1,
    )


def stock_daily(code, market, refresh=False):
    """个股日K线全量。market: 0=深市 1=沪市。"""
    return _cached(
        f"stk_{market}_{code}_day",
        lambda: _fetch_bars(market, code, CATEGORY_DAY, is_index=False),
        refresh,
    )


def to_weekly(df):
    """日线转周线(周五收盘对齐)，成交量/额按周求和。"""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last",
           "vol": "sum", "amount": "sum"}
    out = df.resample("W-FRI").agg(agg)
    return out.dropna(subset=["close"])


if __name__ == "__main__":
    # 自检：拉沪深300日线和分钟线，打印范围
    d = index_daily("000300", 1)
    print("沪深300日线:", len(d), d.index[0].date(), "~", d.index[-1].date())
    m = index_min1("000300", 1)
    print("沪深300分钟线:", len(m), m.index[0], "~", m.index[-1])
