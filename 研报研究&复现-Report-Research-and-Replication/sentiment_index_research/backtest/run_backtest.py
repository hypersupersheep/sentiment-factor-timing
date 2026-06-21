# -*- coding: utf-8 -*-
"""
GMX/GMVX 情绪指数策略回测
==========================
数据：通达信(pytdx) 沪深300 / 中证500 日线 -> 周线
执行：t周五收盘出信号 -> t+1周以周一开盘价(用下周open)成交，杜绝前视偏差
成本：单边 6bp(佣金1bp+滑点5bp，按ETF口径，免印花税)
切分：样本内 2005-04 ~ 2011-07-11(研报发布日)；样本外 2011-07-11 ~ 至今
      参数全部沿用研报原文设定，未在样本外调过任何参数。
输出：equity_curve.csv / signals.csv / report.txt(本目录下)
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # 情绪因子研究/
sys.path.insert(0, os.path.join(ROOT, "common"))
sys.path.insert(0, os.path.dirname(HERE))              # 研究文件夹本身

import tdx_data
import perf
from sentiment_index_strategy import run_pipeline

COST_ONEWAY = 0.0006          # 单边成本6bp
SPLIT_DATE = "2011-07-11"     # 研报发布日 = 样本内/外分界
ANN = 52                      # 周频年化因子


def main():
    print("[1/4] 拉取通达信数据...")
    df300 = tdx_data.index_daily("000300", 1)
    df500 = tdx_data.index_daily("000905", 1)
    w300, w500 = tdx_data.to_weekly(df300), tdx_data.to_weekly(df500)
    print(f"  沪深300周线 {len(w300)} 条: {w300.index[0].date()} ~ {w300.index[-1].date()}")

    print("[2/4] 计算 GMX/GMVX 与信号(滚动PCA，约1-2分钟)...")
    sig = run_pipeline(w300, w500)

    print("[3/4] 回测...")
    # t周信号 -> t+1周一开盘成交 -> 收益从 t+1周开盘到 t+2周开盘
    # 用周线 open 近似周一开盘价；持仓收益 = position(t) * [open(t+2)/open(t+1)-1]
    px_open = w300["open"].reindex(sig.index).ffill()
    fwd_ret = px_open.shift(-2) / px_open.shift(-1) - 1   # 下周开盘->下下周开盘
    pos = sig["position"]
    trade = pos.diff().abs().fillna(pos.abs())            # 仓位变动产生成本
    strat_ret = (pos * fwd_ret - trade * COST_ONEWAY).dropna()
    bench_ret = fwd_ret.reindex(strat_ret.index)          # 基准: 沪深300 buy&hold

    print("[4/4] 输出报告...")
    rep = perf.split_report(strat_ret, bench_ret, SPLIT_DATE, ANN)
    text = (
        "GMX/GMVX 情绪指数策略回测报告\n"
        f"样本内: {strat_ret.index[0].date()} ~ {SPLIT_DATE} (研报发布日)\n"
        f"样本外: {SPLIT_DATE} ~ {strat_ret.index[-1].date()}\n"
        f"成本: 单边{COST_ONEWAY*1e4:.0f}bp | 执行: 信号次周开盘成交\n\n"
        + perf.fmt_report(rep) + "\n\n"
        f"持仓周占比: {pos.mean():.1%} | 换仓次数: {int(trade.sum())}\n"
        "注: 代理变量与原文不同(见使用指南), 结果反映方法论而非原文精确复现。\n"
    )
    print("\n" + text)

    nav = pd.DataFrame({
        "strategy": (1 + strat_ret).cumprod(),
        "benchmark": (1 + bench_ret).cumprod(),
    })
    nav.to_csv(os.path.join(HERE, "equity_curve.csv"))
    sig.to_csv(os.path.join(HERE, "signals.csv"))
    with open(os.path.join(HERE, "report.txt"), "w") as f:
        f.write(text)
    print(f"已保存: equity_curve.csv / signals.csv / report.txt -> {HERE}")


if __name__ == "__main__":
    main()
