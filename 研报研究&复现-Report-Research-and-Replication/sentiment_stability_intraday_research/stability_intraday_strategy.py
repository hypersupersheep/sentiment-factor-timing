# -*- coding: utf-8 -*-
"""
市场情绪平稳度日内策略复现
==========================
研报：广发证券《基于市场情绪平稳度的股指期货日内交易策略》(2015-04-02)

核心指标(原文定义)：
  回撤      DD_i  = max_{j<=i}(p_j - p_i) / p_j   —— 从前高回落
  反向回撤  RDD_i = -min_{j<=i}(p_j - p_i) / p_j  —— 从前低回升
  平稳度 = min(mean(DD), mean(RDD))，越小趋势性越强

两套模型(参数全部为原文样本内优化值)：
  单次开仓：开盘50分钟观察 -> 平稳度<9/10000 开仓 -> 方向看50分钟价vs开盘价
            -> 止损0.5%，否则持有到收盘
  两次开仓：上午同上但11:30平仓；下午用11:12~13:32样本，13:33决定开仓，收盘平仓

本文件只含指标与单日交易逻辑，循环回测在 backtest/run_backtest.py。
"""

import numpy as np
import pandas as pd

# ---------------- 原文参数 ----------------
OBS_MINUTES = 50          # 观察窗口：开盘后50分钟
STABILITY_TH = 9 / 10000  # 平稳度开仓阈值
STOP_LOSS = 0.005         # 止损0.5%
COST_RT = 2 / 10000       # 双边交易成本


def stability(prices):
    """计算一段价格序列的市场情绪平稳度(原文公式)。

    prices: 一维 array/Series。
    实现：cummax/cummin 向量化，避免双重循环。
    """
    p = np.asarray(prices, dtype=float)
    if len(p) < 2:
        return np.nan
    dd = (np.maximum.accumulate(p) - p) / np.maximum.accumulate(p)   # 各点回撤
    rdd = (p - np.minimum.accumulate(p)) / np.minimum.accumulate(p)  # 各点反向回撤
    return min(dd.mean(), rdd.mean())


def _exit_with_stop(path, direction, stop=STOP_LOSS):
    """给定持仓期价格路径(含入场价为首元素)与方向，按止损/收盘平仓规则算收益。

    direction: +1 多 / -1 空。触发止损按止损价离场(忽略跳空劣化，原文同)。
    返回毛收益率。
    """
    entry = path[0]
    ret_path = direction * (path / entry - 1.0)
    breach = np.where(ret_path < -stop)[0]
    if len(breach):
        return -stop
    return ret_path[-1]


def trade_single(day_bars, obs=OBS_MINUTES, th=STABILITY_TH,
                 stop=STOP_LOSS, cost=COST_RT):
    """单次开仓模型：处理一天的1分钟K线，返回(收益, 方向)。不开仓返回(0,0)。

    day_bars: 当日1分钟 DataFrame(index为时间, 含close列)，按时间升序。
    """
    closes = day_bars["close"].values
    if len(closes) < obs + 10:          # 数据残缺的交易日跳过
        return 0.0, 0
    window = closes[:obs]
    if stability(window) >= th:         # 震荡日：不交易
        return 0.0, 0
    open_px = day_bars["open"].iloc[0]  # 当日开盘价
    direction = 1 if window[-1] > open_px else -1
    ret = _exit_with_stop(closes[obs - 1:], direction, stop)
    return ret - cost, direction


def trade_dual(day_bars, obs=OBS_MINUTES, th=STABILITY_TH,
               stop=STOP_LOSS, cost=COST_RT):
    """两次开仓模型：上午、下午各独立决策一次，返回[(时段,收益,方向),...]。

    时间约定(指数/期货通用)：上午收盘11:30，下午开盘13:00。
    下午观察样本取 11:12~13:32(跨午休，原文设定)，13:33决定开仓。
    """
    t = day_bars.index
    res = []

    # ---- 上午段：开盘50分钟观察，符合条件开仓，11:30平仓 ----
    am = day_bars.between_time("09:30", "11:30")
    closes_am = am["close"].values
    if len(closes_am) >= obs + 10:
        window = closes_am[:obs]
        if stability(window) < th:
            open_px = am["open"].iloc[0]
            d = 1 if window[-1] > open_px else -1
            ret = _exit_with_stop(closes_am[obs - 1:], d, stop)
            res.append(("AM", ret - cost, d))

    # ---- 下午段：11:12~13:32样本观察，13:33开仓，收盘平仓 ----
    obs_pm = day_bars.between_time("11:12", "13:32")["close"].values
    pm = day_bars.between_time("13:33", "15:00")
    closes_pm = pm["close"].values
    if len(obs_pm) >= 20 and len(closes_pm) >= 10:
        if stability(obs_pm) < th:
            d = 1 if obs_pm[-1] > obs_pm[0] else -1   # 方向看观察段首尾
            ret = _exit_with_stop(closes_pm, d, stop)
            res.append(("PM", ret - cost, d))
    return res
