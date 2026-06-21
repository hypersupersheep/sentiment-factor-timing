# -*- coding: utf-8 -*-
"""
回测绩效统计公共模块
====================
四个研究文件夹的回测脚本共用，保证指标口径一致：
年化收益按 ann_factor 复利折算，Sharpe 用同口径年化，最大回撤基于净值高点。
"""

import numpy as np
import pandas as pd


def perf_stats(returns, ann_factor):
    """returns: 周期收益率 Series；ann_factor: 年化周期数(周频52/日频244)。"""
    r = returns.dropna()
    if len(r) == 0:
        return {}
    nav = (1 + r).cumprod()
    years = len(r) / ann_factor
    ann_ret = nav.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    ann_vol = r.std() * np.sqrt(ann_factor)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    dd = nav / nav.cummax() - 1
    # 胜率只统计有持仓的周期(收益恰为0的空仓期不计入)，避免低估择时策略胜率
    active = r[r != 0]
    win = (active > 0).mean() if len(active) else np.nan
    return {
        "区间收益": nav.iloc[-1] - 1,
        "年化收益": ann_ret,
        "年化波动": ann_vol,
        "Sharpe": sharpe,
        "最大回撤": dd.min(),
        "胜率": win,
        "期数": len(r),
    }


def split_report(strat_ret, bench_ret, split_date, ann_factor, label_is="样本内", label_oos="样本外"):
    """按 split_date 切分样本内/样本外，输出策略与基准的对比表(DataFrame)。"""
    rows = {}
    seg = {
        label_is: (strat_ret[:split_date], bench_ret[:split_date]),
        label_oos: (strat_ret[split_date:], bench_ret[split_date:]),
        "全区间": (strat_ret, bench_ret),
    }
    for name, (sr, br) in seg.items():
        s, b = perf_stats(sr, ann_factor), perf_stats(br, ann_factor)
        for k in s:
            rows.setdefault((name, "策略"), {})[k] = s[k]
            rows.setdefault((name, "基准"), {})[k] = b.get(k)
    df = pd.DataFrame(rows).T
    return df


def fmt_report(df):
    """把 split_report 的结果格式化成可读文本。"""
    out = df.copy()
    for col in ["区间收益", "年化收益", "年化波动", "最大回撤", "胜率"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
    if "Sharpe" in out.columns:
        out["Sharpe"] = out["Sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    if "期数" in out.columns:
        out["期数"] = out["期数"].map(lambda x: f"{int(x)}" if pd.notna(x) else "-")
    return out.to_string()
