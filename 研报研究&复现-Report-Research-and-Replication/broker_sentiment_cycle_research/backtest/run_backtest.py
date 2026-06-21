# -*- coding: utf-8 -*-
"""
券商景气度策略回测
==================
数据：通达信(pytdx)
  券商板块: 中证全指证券公司指数 399975
  全市场成交额: 上证综指(999999) + 深证成指(399001) 成交额合计
  基准: 沪深300(中证全指TDX不可得，用沪深300替代相对收益基准)
频率：月频。t月末出信号 -> t+1月首个交易日开盘成交(月线open近似)
成本：单边15bp(行业指数ETF口径偏保守)
切分：样本内 数据起点 ~ 2020-02-24(研报发布日)；样本外 2020-02-24 ~ 至今
输出：equity_curve.csv / signals.csv / report.txt
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "common"))
sys.path.insert(0, os.path.dirname(HERE))

import tdx_data
import perf
from broker_cycle_strategy import build_indicator, generate_signals

COST_ONEWAY = 0.0015
SPLIT_DATE = "2020-02-24"   # 研报发布日
ANN = 12


def monthly(df):
    return df.resample("ME").agg(
        {"open": "first", "close": "last", "amount": "sum"}).dropna()


def main():
    print("[1/4] 拉取通达信数据...")
    broker = monthly(tdx_data.index_daily("399975", 0))
    sh = monthly(tdx_data.index_daily("999999", 1))
    sz = monthly(tdx_data.index_daily("399001", 0))
    hs300 = monthly(tdx_data.index_daily("000300", 1))
    print(f"  券商指数(399975): {broker.index[0].date()} ~ {broker.index[-1].date()}")

    amount = (sh["amount"] + sz["amount"]).reindex(broker.index)

    print("[2/4] 构造指标与信号...")
    ind = build_indicator(amount, broker["close"])
    sig = generate_signals(ind["indicator"])

    print("[3/4] 回测...")
    # t月末信号 -> t+1月初开盘建仓 -> 收益记 t+1月开盘到 t+2月开盘
    fwd_broker = broker["open"].shift(-2) / broker["open"].shift(-1) - 1
    fwd_300 = hs300["open"].shift(-2) / hs300["open"].shift(-1) - 1
    pos = sig["position"]
    trade = pos.diff().abs().fillna(pos.abs())

    # 绝对收益版: 持券商指数或空仓
    abs_ret = (pos * fwd_broker.reindex(pos.index) - trade * COST_ONEWAY).dropna()
    # 相对收益版(原文口径): 持仓期收益相对沪深300的超额
    rel_ret = (pos * (fwd_broker - fwd_300).reindex(pos.index)
               - trade * COST_ONEWAY).dropna()
    bench_abs = fwd_broker.reindex(abs_ret.index)           # 基准: 券商指数躺平
    bench_rel = (fwd_broker - fwd_300).reindex(rel_ret.index)

    print("[4/4] 输出报告...")
    lines = ["券商景气度策略回测报告", "=" * 60,
             f"样本内: {abs_ret.index[0].date()} ~ {SPLIT_DATE} (研报发布日)",
             f"样本外: {SPLIT_DATE} ~ {abs_ret.index[-1].date()}",
             f"成本: 单边{COST_ONEWAY*1e4:.0f}bp | 月频, 信号次月初开盘成交",
             f"开仓次数: {int((pos.diff() == 1).sum())} | 持仓月占比: {pos.mean():.1%}", ""]
    lines.append("[绝对收益版] 持券商指数 vs 券商指数buy&hold")
    lines.append(perf.fmt_report(perf.split_report(abs_ret, bench_abs, SPLIT_DATE, ANN)))
    lines.append("\n[相对收益版(原文口径)] 超额=券商-沪深300, 基准为始终持有的超额")
    lines.append(perf.fmt_report(perf.split_report(rel_ret, bench_rel, SPLIT_DATE, ANN)))
    lines.append("\n注: 收入增速用全市场成交额TTM同比代理, PB用3年相对价格代理(见使用指南)。")

    text = "\n".join(lines)
    print("\n" + text)

    pd.DataFrame({
        "abs_strategy": (1 + abs_ret).cumprod(),
        "abs_benchmark": (1 + bench_abs).cumprod(),
        "rel_strategy": (1 + rel_ret).cumprod(),
        "rel_benchmark": (1 + bench_rel).cumprod(),
    }).to_csv(os.path.join(HERE, "equity_curve.csv"))
    pd.concat([ind, sig[["score", "position"]]], axis=1).to_csv(
        os.path.join(HERE, "signals.csv"))
    with open(os.path.join(HERE, "report.txt"), "w") as f:
        f.write(text)
    print(f"\n已保存: equity_curve.csv / signals.csv / report.txt -> {HERE}")


if __name__ == "__main__":
    main()
