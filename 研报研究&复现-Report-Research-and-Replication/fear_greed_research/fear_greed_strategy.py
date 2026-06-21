# -*- coding: utf-8 -*-
"""
贪婪恐惧择时模型复现
====================
研报：国泰君安《在众人恐惧时贪婪，在众人贪婪时恐惧——数量化专题之五十三》(2015-01-19)

规则严格按原文：
  情绪指数 上穿 [过去242日均值 + 1.5σ] -> 卖出信号(众人贪婪)
  情绪指数 下穿 [过去242日均值 - 1.5σ] -> 买入信号(众人恐惧)
  中间区域维持原信号；连续同向信号只取第一个。

与原文的差异：
  原文情绪指数来自论坛帖子文本情感分析(数据不可复现)；
  本复现用价量构造"日频情绪代理指数"，同样取20日移动平均(与原文平滑一致)：
    情绪代理 = z(PSY 12日上涨天数比例) + z(量比 5日/60日成交额) + z(20日动量)
  三者都是典型的散户情绪同步变量(涨得多、放量、有赚钱效应 -> 情绪高)。

行业选择模型：原文需要分行业文本情绪，复现保留同样的"买最悲观、卖最乐观"
分组框架，情绪输入改为各行业指数自身的情绪代理(接口允许注入任意情绪数据，
将来接上文本情绪数据即可无缝替换)。
"""

import numpy as np
import pandas as pd

# ---------------- 原文参数 ----------------
BAND_WINDOW = 242    # 阈值滚动窗口(交易日)，原文"过去242日"
SIGMA_MULT = 1.5     # 原文在1σ/1.5σ/2σ中按定性逻辑选1.5，拒绝精细优化
SMOOTH = 20          # 情绪指数20日移动平均，与原文一致


def _zscore(s, window=242):
    """滚动z-score，只用历史窗口，无前视。"""
    return (s - s.rolling(window).mean()) / s.rolling(window).std()


def build_sentiment_proxy(df_daily):
    """用日线价量构造情绪代理指数(替代原文论坛文本情绪)。

    df_daily: DataFrame[open/high/low/close/vol/amount]，日频。
    返回20日平滑后的情绪代理 Series。
    """
    close, amount = df_daily["close"], df_daily["amount"]
    ret = close.pct_change()

    psy = (ret > 0).rolling(12).mean()                    # PSY: 12日上涨天数占比
    vol_ratio = amount.rolling(5).mean() / amount.rolling(60).mean()  # 量比
    momentum = close.pct_change(20)                       # 20日动量(赚钱效应)

    raw = _zscore(psy) + _zscore(vol_ratio) + _zscore(momentum)
    return raw.rolling(SMOOTH).mean().rename("sentiment")


def generate_signals(sentiment, band_window=BAND_WINDOW, sigma_mult=SIGMA_MULT):
    """按原文阈值规则生成信号与仓位。

    返回 DataFrame[sentiment, upper, lower, signal, position]:
      signal: +1买入信号当日 / -1卖出信号当日 / 0无新信号
      position: 持续仓位，买入信号后为1，卖出信号后为0(纯多头版本)
                原文假设可做空，回测脚本中另算多空版本(卖出后-1)。
    """
    s = sentiment.dropna()
    mean = s.rolling(band_window).mean()
    std = s.rolling(band_window).std()
    upper, lower = mean + sigma_mult * std, mean - sigma_mult * std

    sig = pd.Series(0, index=s.index)
    state = 0  # 当前信号状态: +1多 / -1空 / 0未初始化
    for i in range(1, len(s)):
        if np.isnan(upper.iloc[i]):
            continue
        cross_up = s.iloc[i - 1] <= upper.iloc[i - 1] and s.iloc[i] > upper.iloc[i]
        cross_dn = s.iloc[i - 1] >= lower.iloc[i - 1] and s.iloc[i] < lower.iloc[i]
        # 连续相同信号维持原状态(原文规则)，只有状态翻转才记信号
        if cross_up and state != -1:
            sig.iloc[i], state = -1, -1
        elif cross_dn and state != 1:
            sig.iloc[i], state = +1, +1

    position = sig.replace(0, np.nan).ffill().fillna(0).clip(lower=0)  # 纯多头
    state_ls = sig.replace(0, np.nan).ffill().fillna(0)                # 多空状态
    return pd.DataFrame({
        "sentiment": s, "upper": upper, "lower": lower,
        "signal": sig, "position": position, "state_ls": state_ls,
    })


def industry_rotation_groups(sent_by_industry, n_groups=5):
    """贪婪恐惧行业选择：按上月行业情绪从低到高分组。

    sent_by_industry: DataFrame，index=月末日期，columns=行业，值=该月行业情绪。
    返回同形状的分组标签(1=最悲观组(买入)，n_groups=最乐观组(卖出))。
    下月持仓 = 上月标签为1的行业等权——调用方负责shift避免前视。
    """
    ranks = sent_by_industry.rank(axis=1, pct=True)
    groups = np.ceil(ranks * n_groups).clip(1, n_groups)
    return groups
