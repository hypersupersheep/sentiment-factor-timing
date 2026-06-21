# -*- coding: utf-8 -*-
"""
择时叠加对照评估模块
====================
回答"情绪择时到底贡献了什么"——把择时信号当作叠加在某个 baseline(基准持仓)
上的开关，干净地做 marginal contribution(边际贡献) 归因：

    baseline           : 一直满仓持有(buy & hold)
    baseline + 择时     : 情绪信号=1 时持有, =0 时空仓吃货币基金利率
    Δ                  : 后者 - 前者, 看择时是否 降回撤/提收益/提Sharpe/提胜率

统一在日频上做(周频/月频信号 ffill 到日频)，保证不同策略口径可比、可叠加。
绘图用英文标签避免中文字体缺失乱码；中文结论放在 md 报告里。
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # 无界面后端，直接存png
import matplotlib.pyplot as plt
# mac 系统自带的中文字体，避免标题中文显示为方框
for _f in ["PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS"]:
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

ANN_DAILY = 244
CASH_ANNUAL = 0.018              # 空仓期货币基金年化(原GTJA回测同口径)


def _stats(ret):
    """日频收益序列 -> 绩效字典。胜率按月度计(日频胜率噪音太大)。"""
    r = ret.dropna()
    if len(r) < 20:
        return {k: np.nan for k in
                ["年化收益", "年化波动", "Sharpe", "最大回撤", "Calmar", "月胜率"]}
    nav = (1 + r).cumprod()
    years = len(r) / ANN_DAILY
    ann = nav.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(ANN_DAILY)
    dd = (nav / nav.cummax() - 1).min()
    m_ret = (1 + r).resample("ME").prod() - 1     # 月度收益算胜率
    return {
        "年化收益": ann,
        "年化波动": vol,
        "Sharpe": ann / vol if vol > 0 else np.nan,
        "最大回撤": dd,
        "Calmar": ann / abs(dd) if dd < 0 else np.nan,
        "月胜率": (m_ret > 0).mean(),
    }


def evaluate(baseline_ret, position, cost_oneway=0.0006,
             cash_annual=CASH_ANNUAL, label="策略"):
    """核心评估：返回 (对照表DataFrame, 净值DataFrame)。

    baseline_ret: 基准持仓的日收益 Series(已对齐交易日)。
    position:     日频目标仓位 0/1，应已 shift 到"当日生效"(执行滞后由调用方处理)。
    cost_oneway:  单边交易成本，仓位变动时按变动量计。
    """
    idx = baseline_ret.dropna().index
    pos = position.reindex(idx).ffill().fillna(0).clip(0, 1)
    cash_daily = (1 + cash_annual) ** (1 / ANN_DAILY) - 1
    trade = pos.diff().abs().fillna(pos.abs())

    overlay_ret = pos * baseline_ret + (1 - pos) * cash_daily - trade * cost_oneway

    base_s, over_s = _stats(baseline_ret), _stats(overlay_ret)
    delta = {k: over_s[k] - base_s[k] for k in base_s}
    table = pd.DataFrame(
        {"baseline(满仓持有)": base_s, f"baseline+择时": over_s, "Δ(择时贡献)": delta}
    ).T

    nav = pd.DataFrame({
        "baseline": (1 + baseline_ret).cumprod(),
        "baseline+timing": (1 + overlay_ret).cumprod(),
    })
    nav.attrs["position"] = pos
    nav.attrs["持仓占比"] = pos.mean()
    nav.attrs["换手次数"] = int(trade.sum())
    return table, nav


def fmt_table(table):
    """对照表格式化为可读文本(百分比/两位小数)。"""
    out = table.copy()
    for c in ["年化收益", "年化波动", "最大回撤", "月胜率"]:
        if c in out.columns:
            out[c] = out[c].map(lambda x: f"{x:+.2%}" if pd.notna(x) else "-")
    for c in ["Sharpe", "Calmar"]:
        if c in out.columns:
            out[c] = out[c].map(lambda x: f"{x:+.2f}" if pd.notna(x) else "-")
    return out.to_string()


def plot_nav(nav, title, path):
    """画 baseline vs baseline+timing 净值 + 回撤对比，存png。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7),
                                   gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax1.plot(nav.index, nav["baseline"], label="Baseline (buy & hold)",
             color="#888", lw=1.3)
    ax1.plot(nav.index, nav["baseline+timing"], label="Baseline + Sentiment Timing",
             color="#c0392b", lw=1.5)
    ax1.set_yscale("log")
    ax1.set_ylabel("NAV (log)")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)
    for col, c in (("baseline", "#888"), ("baseline+timing", "#c0392b")):
        dd = nav[col] / nav[col].cummax() - 1
        ax2.fill_between(nav.index, dd, 0, color=c, alpha=0.35)
    ax2.set_ylabel("Drawdown")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def to_daily_position(weekly_or_monthly_pos, daily_index, exec_lag=1):
    """把低频(周/月)仓位映射到日频并施加执行滞后，杜绝前视。

    weekly_or_monthly_pos: index为信号日期的0/1仓位。
    daily_index: 目标日频交易日索引。
    exec_lag: 信号出现后延迟几个交易日生效(默认次日)。
    """
    s = weekly_or_monthly_pos.reindex(
        weekly_or_monthly_pos.index.union(daily_index)).ffill()
    s = s.reindex(daily_index).ffill().fillna(0)
    return s.shift(exec_lag).fillna(0)
