# -*- coding: utf-8 -*-
"""
选股 baseline 组合构建
======================
用途：作为"既定选股策略"的基准持仓，用来检验情绪择时叠加在选股策略上的效果
(对照评估的 Version B)。

设计：
  universe : 上证50 成分股(akshare 取名单, 仅此一处用akshare; 价格仍走TDX)
  选股因子 : 低波动 low-volatility(过去120日收益率标准差最小的 top_n 只)
             —— 经典、稳健、纯价格可复现的 anomaly，不需要基本面数据
  调仓     : 月度等权再平衡

已知局限(报告中标注)：成分名单是 akshare 返回的"当前"成分，历史回测存在
survivorship/look-ahead bias；point-in-time 成分需付费源(中证指数/Wind)。
作为"选股策略 baseline"用于择时叠加的相对比较是可接受的，因为 baseline 与
baseline+择时 用的是同一组合，bias 在做差(Δ)时大部分抵消。
"""

import os

import numpy as np
import pandas as pd

import tdx_data

CACHE_DIR = tdx_data.CACHE_DIR


def infer_market(code):
    """从6位代码推断交易所：沪市=1, 深市=0。"""
    code = str(code).zfill(6)
    if code[0] in ("6", "9") or code[:3] in ("688", "605"):
        return 1
    return 0


def sse50_universe(refresh=False):
    """取上证50成分(代码列表)，缓存到 csv。仅此处依赖 akshare。"""
    path = os.path.join(CACHE_DIR, "universe_sse50.csv")
    if not refresh and os.path.exists(path):
        return pd.read_csv(path, dtype={"code": str})["code"].tolist()
    import akshare as ak
    df = ak.index_stock_cons_csindex(symbol="000016")
    codes = df["成分券代码"].astype(str).str.zfill(6).tolist()
    pd.DataFrame({"code": codes}).to_csv(path, index=False)
    return codes


def build_close_panel(codes, start="2005-01-01", refresh=False):
    """拉取个股**后复权**日线收盘价，拼成 panel(index=日期, columns=代码)。

    关键：必须用复权价。TDX 个股原始价不复权，除权除息/拆分当日会产生
    -50%~-90% 的假暴跌，污染组合(实测裸价组合最大回撤达-99.8%)。
    这里用 akshare 后复权(hfq)价，整张 panel 缓存为单个 csv 加速。
    指数策略不受此影响(指数本身连续)，仍全部走 TDX。
    """
    path = os.path.join(CACHE_DIR, "panel_sse50_hfq.csv")
    if not refresh and os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import akshare as ak
    series = {}
    for code in codes:
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                    start_date="20050101", adjust="hfq")
            df["日期"] = pd.to_datetime(df["日期"])
            series[code] = df.set_index("日期")["收盘"]
        except Exception:
            continue
    panel = pd.DataFrame(series).sort_index()
    panel.to_csv(path)
    return panel.loc[start:]


def low_vol_portfolio(top_n=20, vol_window=120, refresh_universe=False):
    """构建低波动选股组合，返回日频组合收益 Series。

    每月末按过去 vol_window 日收益波动率选最低的 top_n 只，下月等权持有。
    """
    codes = sse50_universe(refresh=refresh_universe)
    panel = build_close_panel(codes).replace(0, np.nan)   # 0价视为缺失
    rets = panel.pct_change()
    # 数据清洗：上证50全是主板，日涨跌停±10%。任何|日收益|>11%必是
    # 除权/停复牌/坏数据造成的假跳，置0剔除(否则单只假暴涨会拖垮整个组合)。
    rets = rets.where(rets.abs() <= 0.11, 0.0)
    vol = rets.rolling(vol_window).std()

    # 月末选股 -> 下月持仓权重(等权)，shift(1)避免用当月信息
    month_ends = rets.resample("ME").last().index
    weights = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    for me in month_ends:
        v = vol.loc[:me].iloc[-1] if len(vol.loc[:me]) else None
        if v is None:
            continue
        picks = v.dropna().nsmallest(top_n).index
        if len(picks) == 0:
            continue
        nxt = weights.index[weights.index > me]
        if len(nxt) == 0:
            continue
        weights.loc[nxt, picks] = 1.0 / len(picks)
    weights = weights.shift(1).fillna(0)             # 持仓滞后一日生效

    port_ret = (weights * rets).sum(axis=1)
    # 只在有持仓后开始计
    first = weights.sum(axis=1).gt(0).idxmax()
    return port_ret.loc[first:].rename("low_vol_portfolio")


if __name__ == "__main__":
    pr = low_vol_portfolio()
    nav = (1 + pr).cumprod()
    print("低波动组合:", pr.index[0].date(), "~", pr.index[-1].date(),
          "| 累计", f"{nav.iloc[-1]-1:.1%}")
