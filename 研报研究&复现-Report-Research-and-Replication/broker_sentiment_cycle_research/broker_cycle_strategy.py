# -*- coding: utf-8 -*-
"""
券商景气度(情绪引领周期)策略复现
================================
研报：光大证券《券商：盈利锚定估值，情绪引领周期——行业景气度系列之十》(2020-02-24)

原文模型：
  指标 = 三项业务收入增速 / PB        (剥离盈利后的"纯情绪估值动量")
  事件: 指标变化幅度 < -0.003 -> +1分(开仓)；> 0.001 -> -1分(平仓)
  月频，得分>0 持有券商板块(相对中证全指)，<=0 平仓

通达信数据可得性导致的代理(逻辑保持"收入增速/估值"不变)：
  三项业务收入增速 -> 全市场成交额TTM同比增速
      理由：经纪业务是三项业务中弹性最大的主导项，且成交额同时是
      两融交易额的强相关变量，原文表1中两项规模指标均由成交活跃度驱动。
  PB -> 券商指数收盘价 / 过去3年均价(相对估值水平)
      理由：指数净资产数据不可得；3年窗口对应原文Z-score的周期设定。
  阈值 -0.003/0.001 是原文指标量纲下的数值，代理指标量纲不同，
      按比例映射为变化量滚动标准差的 -0.3σ / +0.1σ(保持原文3:1的不对称度，
      即"开仓从严、平仓从宽"——负面信号优先的设计哲学)。
"""

import numpy as np
import pandas as pd

# ---------------- 参数(映射自原文) ----------------
TTM_MONTHS = 12        # 收入TTM平滑窗口(原文对流量数据滚动12月求和)
PB_WINDOW = 36         # 相对估值窗口(月)=3年，对应原文Z-score周期
OPEN_K = 0.3           # 开仓阈值 = -0.3σ(映射原文-0.003)
CLOSE_K = 0.1          # 平仓阈值 = +0.1σ(映射原文+0.001)
SIGMA_WINDOW = 36      # 阈值标准差的滚动估计窗口(月)


def build_indicator(amount_m, broker_close_m):
    """构造"收入增速/估值"指标(月频)。

    amount_m:       全市场月成交额 Series(月频, 沪深两市合计)
    broker_close_m: 券商指数月收盘价 Series
    返回 DataFrame[rev_growth, pb_proxy, indicator]
    """
    df = pd.DataFrame(index=amount_m.index)
    # 收入代理：成交额滚动12月求和(TTM) -> 同比增速
    amt_ttm = amount_m.rolling(TTM_MONTHS).sum()
    df["rev_growth"] = amt_ttm.pct_change(12)
    # 估值代理：券商指数价格相对3年均价
    df["pb_proxy"] = broker_close_m / broker_close_m.rolling(PB_WINDOW).mean()
    # 原文指标：收入增速 / PB —— 值下降代表乐观情绪推升估值
    df["indicator"] = df["rev_growth"] / df["pb_proxy"]
    return df.dropna()


def generate_signals(indicator):
    """事件化打分 -> 月频仓位。

    原文规则：指标变化幅度 < -0.003 记+1(乐观情绪推升估值, 开仓)，
              > 0.001 记-1(情绪退潮, 平仓)；得分>0持仓。
    本实现阈值为变化量的滚动σ比例(见文件头说明)，状态在两阈值间维持。
    返回 DataFrame[indicator, delta, score, position]。
    """
    s = indicator.dropna()
    delta = s.diff()
    sigma = delta.rolling(SIGMA_WINDOW, min_periods=12).std()

    score = pd.Series(0, index=s.index)
    state = 0
    for i in range(1, len(s)):
        if np.isnan(sigma.iloc[i]):
            continue
        if delta.iloc[i] < -OPEN_K * sigma.iloc[i]:
            state = +1            # 情绪动量启动：开仓
        elif delta.iloc[i] > CLOSE_K * sigma.iloc[i]:
            state = -1            # 情绪退潮：平仓
        score.iloc[i] = state

    return pd.DataFrame({
        "indicator": s, "delta": delta,
        "score": score,
        "position": (score > 0).astype(float),
    })
