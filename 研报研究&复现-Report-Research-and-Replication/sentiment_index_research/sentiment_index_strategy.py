# -*- coding: utf-8 -*-
"""
GMX/GMVX 市场情绪指数策略复现
================================
研报：国泰君安《市场情绪指数的建立及应用——数量化研究系列之十三》(2011-07-11)

方法论严格按原文：
  情绪代理变量 -> 移动平均平滑 -> z-score 标准化 -> 滚动 PCA
  -> GMX(第一主成分) / GMVX(第三主成分) -> 阈值/方向规则 -> 周频仓位

与原文的差异(数据可得性导致，见 策略使用指南.md 的代理变量映射表)：
  原文用 PE/PB/换手率/上涨家数占比/IPO首日涨幅/新增开户数 六个变量；
  通达信只有价量数据，本复现用六个价量代理变量替代，维度数保持一致。

本文件只负责"因子计算 + 信号生成"，不含取数与回测循环，
这样同一份代码既能被 backtest/run_backtest.py 调用，也能接实盘数据。
"""

import numpy as np
import pandas as pd

# ---------------- 默认参数(全部来自原文或样本内设定) ----------------
MA_WINDOW = 4        # 变量移动平均窗口(周)，原文对基础指标做平滑
TURNOVER_LAG = 4     # 换手率滞后期数，原文选"换手率(-4)"
PCA_WINDOW = 156     # 滚动PCA窗口(周)=3年，原文按周滚动更新
SIGMA_MULT = 2.0     # GMX阈值倍数：均值±2σ，原文图37
GMX_PC = 0           # GMX = 第一主成分
GMVX_PC = 2          # GMVX = 第三主成分


def build_proxy_features(df300_w, df500_w):
    """用通达信周线数据构造六个情绪代理变量(均为周频 Series)。

    入参为沪深300、中证500周线 DataFrame(open/high/low/close/vol/amount)。
    每个代理变量对应原文的一个维度，构造逻辑见使用指南映射表。
    """
    c300 = df300_w["close"]
    ret_w = c300.pct_change()

    feats = pd.DataFrame(index=df300_w.index)

    # 1) 估值代理(对应PE/PB)：价格相对52周均线的偏离——估值水平的纯价格近似
    feats["valuation"] = c300 / c300.rolling(52).mean() - 1.0

    # 2) 活跃度代理(对应换手率)：周成交额相对52周均额，并按原文滞后4期
    amt_ratio = df300_w["amount"] / df300_w["amount"].rolling(52).mean()
    feats["turnover"] = amt_ratio.shift(TURNOVER_LAG)

    # 3) 市场宽度代理(对应上涨家数占比)：近4周上涨周数占比(0~1)
    feats["breadth"] = (ret_w > 0).rolling(4).mean()

    # 4) 投机热度代理(对应IPO首日涨幅)：动量×波动——牛市打新热、波动大
    mom_4w = c300.pct_change(4)
    vol_4w = ret_w.rolling(4).std()
    feats["speculation"] = mom_4w * 10 + vol_4w * 20

    # 5) 场外资金代理(对应新增开户数)：26周动量——开户数在单边市后半段领先，
    #    本质是趋势确认变量，用中期动量近似
    feats["inflow"] = c300.pct_change(26)

    # 6) 风格情绪代理(对应小盘相对收益维度)：中证500/沪深300 相对强弱4周变化
    rel = (df500_w["close"] / c300).reindex(feats.index)
    feats["small_vs_big"] = rel.pct_change(4)

    # 原文对基础指标做移动平均平滑后再标准化
    feats = feats.rolling(MA_WINDOW).mean()
    return feats.dropna()


def _pca_components(X):
    """对标准化矩阵 X(样本×变量) 做PCA，返回(载荷矩阵, 解释度)。

    用相关矩阵的特征分解实现，避免引入 sklearn 依赖；
    X 已经 z-score 过，协方差矩阵即相关矩阵。
    """
    cov = np.cov(X, rowvar=False)
    eigval, eigvec = np.linalg.eigh(cov)        # eigh: 对称矩阵，特征值升序
    order = np.argsort(eigval)[::-1]            # 转为降序，第一主成分在前
    return eigvec[:, order], eigval[order] / eigval.sum()


def compute_gmx_gmvx(feats, window=PCA_WINDOW):
    """滚动PCA计算 GMX / GMVX(周频)。

    原文的滚动差分更新 m(t+1)=m(t)+[m'(t+1)-m'(t)] 等价于：
    每期用最近 window 周窗口重估PCA，只取"本期相对上期的增量"累加，
    从而剔除样本更替对载荷估计的跳变影响。这里直接按该等价形式实现。
    符号对齐：第一主成分与价格水平正相关、第三主成分与下周收益正相关，
    每个窗口内用与估值代理的相关性确定符号，避免特征向量符号随机翻转。
    """
    idx = feats.index
    gmx = pd.Series(np.nan, index=idx)
    gmvx = pd.Series(np.nan, index=idx)
    prev_score = {GMX_PC: None, GMVX_PC: None}
    level = {GMX_PC: 0.0, GMVX_PC: 0.0}

    vals = feats.values
    for t in range(window, len(feats)):
        Xwin = vals[t - window: t + 1]
        # 窗口内 z-score(用窗口自身均值方差，不引入未来信息)
        mu, sd = Xwin.mean(axis=0), Xwin.std(axis=0)
        sd[sd == 0] = 1.0
        Z = (Xwin - mu) / sd
        comps, _ = _pca_components(Z[:-1])      # 载荷用截至上期的窗口估计
        scores = Z @ comps                       # 全窗口得分

        for pc, store in ((GMX_PC, gmx), (GMVX_PC, gmvx)):
            s = scores[:, pc]
            # 符号对齐：与第一个代理变量(估值)正相关为正方向
            if np.corrcoef(s, Z[:, 0])[0, 1] < 0:
                s = -s
            cur, prev = s[-1], s[-2]
            if prev_score[pc] is None:
                level[pc] = cur                  # 首期直接用得分初始化
            else:
                level[pc] += cur - prev          # 差分累加，保持序列可比
            prev_score[pc] = cur
            store.iloc[t] = level[pc]

    return gmx.dropna(), gmvx.dropna()


def generate_signals(gmx, gmvx, sigma_mult=SIGMA_MULT, band_window=PCA_WINDOW):
    """按原文规则生成周频目标仓位(0/1)。

    规则优先级：
      1. GMX 首次上穿 均值+kσ -> 风险提示，强制空仓；
      2. GMX 首次下穿 均值-kσ -> 机会提示，强制满仓；
      3. 其余时间用 GMVX 方向：GMVX>0 看涨持仓，否则空仓。
    阈值用GMX滚动均值±kσ(滚动窗口与PCA一致)，全程无未来信息。
    返回 DataFrame[gmx, gmvx, upper, lower, position]。
    """
    df = pd.DataFrame({"gmx": gmx, "gmvx": gmvx}).dropna()
    mean = df["gmx"].rolling(band_window, min_periods=52).mean()
    std = df["gmx"].rolling(band_window, min_periods=52).std()
    df["upper"] = mean + sigma_mult * std
    df["lower"] = mean - sigma_mult * std

    pos = np.zeros(len(df))
    override = 0  # 阈值事件的覆盖状态: +1锁多 / -1锁空 / 0无
    for i in range(1, len(df)):
        g, g0 = df["gmx"].iloc[i], df["gmx"].iloc[i - 1]
        up, lo = df["upper"].iloc[i], df["lower"].iloc[i]
        if not np.isnan(up):
            if g0 <= df["upper"].iloc[i - 1] and g > up:
                override = -1            # 首次上穿上界：情绪过热，离场
            elif g0 >= df["lower"].iloc[i - 1] and g < lo:
                override = +1            # 首次下穿下界：情绪冰点，进场
            elif lo < g < up and override != 0:
                # 回到中性区域后解除覆盖，交还给GMVX方向信号
                override = 0
        if override != 0:
            pos[i] = 1.0 if override > 0 else 0.0
        else:
            pos[i] = 1.0 if df["gmvx"].iloc[i] > 0 else 0.0
    df["position"] = pos
    return df


def run_pipeline(df300_w, df500_w):
    """一条龙：代理变量 -> GMX/GMVX -> 信号。返回信号 DataFrame。"""
    feats = build_proxy_features(df300_w, df500_w)
    gmx, gmvx = compute_gmx_gmvx(feats)
    return generate_signals(gmx, gmvx)
