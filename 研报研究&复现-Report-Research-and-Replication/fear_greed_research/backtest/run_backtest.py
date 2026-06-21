# -*- coding: utf-8 -*-
"""
贪婪恐惧策略回测(择时 + 行业轮动)
==================================
数据：通达信(pytdx)
  择时：上证综指(999999)日线，2005年至今
  行业轮动：通达信行业指数(880xxx)日线
执行：t日收盘出信号 -> t+1日开盘成交
成本：单边10bp(指数化交易+行业切换从严计)
切分：样本内 2005 ~ 2015-01-19(研报发布日)；样本外 2015-01-19 ~ 至今
      参数(242日窗口、1.5σ、20日平滑)全部沿用原文，样本外未调参。
输出：equity_curve.csv / signals.csv / rotation_groups.csv / report.txt
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "common"))
sys.path.insert(0, os.path.dirname(HERE))

import tdx_data
import perf
from fear_greed_strategy import (build_sentiment_proxy, generate_signals,
                                 industry_rotation_groups)

COST_ONEWAY = 0.0010
SPLIT_DATE = "2015-01-19"   # 研报发布日
ANN = 244                   # 日频年化因子

# 通达信行业指数候选(880xxx, 上交所market=1)。名称以通达信客户端为准，
# 这里只按代码探测有效性；探测结果会缓存到 rotation_universe.csv。
INDUSTRY_CODE_RANGE = list(range(880301, 880500))
MAX_INDUSTRIES = 30         # 取前30个有效行业指数，覆盖面已接近申万一级


def timing_backtest():
    """模型一：贪婪恐惧择时(上证综指)。"""
    print("[择时] 拉取上证综指日线...")
    # TDX上证综指可回溯到1990年，但90年代涨跌停制度/微观结构与现今完全不同，
    # 统一从2004年起算(与其他研究文件夹口径一致)
    df = tdx_data.index_daily("999999", 1).loc["2004-01-01":]
    sent = build_sentiment_proxy(df)
    sig = generate_signals(sent)

    px_open = df["open"].reindex(sig.index)
    fwd_ret = px_open.shift(-2) / px_open.shift(-1) - 1   # t+1开盘 -> t+2开盘

    results = {}
    for name, pos in (("纯多头", sig["position"]),
                      ("多空", sig["state_ls"])):
        trade = pos.diff().abs().fillna(0)
        results[name] = (pos * fwd_ret - trade * COST_ONEWAY).dropna()
    bench = fwd_ret.reindex(results["纯多头"].index)

    # 原文图表7口径：统计信号后5日/22日指数收益
    close = df["close"]
    rows = []
    for d, s in sig.loc[sig["signal"] != 0, "signal"].items():
        loc = close.index.get_loc(d)
        for h in (5, 22):
            if loc + h < len(close):
                rows.append({"date": d.date(), "signal": "买入" if s > 0 else "卖出",
                             "horizon": h,
                             "ret": close.iloc[loc + h] / close.iloc[loc] - 1})
    ev = pd.DataFrame(rows)
    ev_stat = ev.groupby(["signal", "horizon"])["ret"].agg(["mean", "count"])

    return sig, results, bench, ev_stat


def discover_industries():
    """探测有效的通达信行业指数代码(有>=8年历史的才纳入)，结果缓存。"""
    cache = os.path.join(HERE, "rotation_universe.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)["code"].astype(str).tolist()
    api = tdx_data.get_api()
    codes = []
    for c in INDUSTRY_CODE_RANGE:
        try:
            bars = api.get_index_bars(9, 1, str(c), 0, 10)
        except Exception:
            continue
        if bars:
            codes.append(str(c))
        if len(codes) >= MAX_INDUSTRIES:
            break
    pd.DataFrame({"code": codes}).to_csv(cache, index=False)
    return codes


def rotation_backtest():
    """模型二：贪婪恐惧行业轮动(买最悲观组，对照最乐观组)。"""
    print("[轮动] 探测通达信行业指数...")
    codes = discover_industries()
    print(f"  纳入 {len(codes)} 个行业指数: {codes[:8]} ...")

    closes, sents = {}, {}
    for c in codes:
        try:
            df = tdx_data.index_daily(c, 1)
        except Exception:
            continue
        if len(df) < 244 * 8:        # 历史太短的行业剔除，保证回测覆盖度
            continue
        closes[c] = df["close"]
        sents[c] = build_sentiment_proxy(df)
    px = pd.DataFrame(closes).dropna(how="all")
    sent = pd.DataFrame(sents).reindex(px.index)
    print(f"  历史达标行业: {px.shape[1]} 个")

    # 月频：上月末情绪 -> 分5组 -> 下月等权持有
    sent_m = sent.resample("ME").last()
    px_m = px.resample("ME").last()
    ret_m = px_m.pct_change()
    groups = industry_rotation_groups(sent_m).shift(1)    # 用上月分组，避免前视

    group_ret = {}
    for g in range(1, 6):
        mask = groups == g
        group_ret[f"组{g}"] = ret_m.where(mask).mean(axis=1)
    gr = pd.DataFrame(group_ret).dropna(how="all")
    gr["多空(组1-组5)"] = gr["组1"] - gr["组5"]
    gr["全行业等权"] = ret_m.mean(axis=1).reindex(gr.index)
    return gr


def main():
    sig, results, bench, ev_stat = timing_backtest()
    gr = rotation_backtest()

    lines = ["贪婪恐惧策略回测报告", "=" * 60, ""]
    lines.append("一、择时模型(上证综指, 信号次日开盘成交, 单边10bp)")
    lines.append(f"样本内: ~{SPLIT_DATE} | 样本外: {SPLIT_DATE}~ (参数沿用原文未调)")
    for name, sr in results.items():
        lines.append(f"\n[{name}]")
        lines.append(perf.fmt_report(perf.split_report(sr, bench, SPLIT_DATE, ANN)))
    lines.append("\n信号事件统计(对照原文图表7口径: 信号后N日指数收益):")
    lines.append((ev_stat.rename(columns={"mean": "平均收益", "count": "次数"})
                  .assign(平均收益=lambda d: d["平均收益"].map("{:.2%}".format))
                  .to_string()))

    lines.append("\n\n二、行业轮动模型(月频, 买上月最悲观组)")
    sub = {}
    for col in ["组1", "组5", "多空(组1-组5)", "全行业等权"]:
        sub[col] = perf.perf_stats(gr[col], 12)
    stat = pd.DataFrame(sub).T
    for c in ["区间收益", "年化收益", "年化波动", "最大回撤", "胜率"]:
        stat[c] = stat[c].map("{:.2%}".format)
    stat["Sharpe"] = stat["Sharpe"].map("{:.2f}".format)
    lines.append(stat.to_string())
    lines.append("\n注: 情绪用价量代理(非原文论坛文本), 行业为通达信880xxx指数(非申万)。")

    text = "\n".join(str(x) for x in lines)
    print("\n" + text)

    pd.DataFrame({
        "long_only": (1 + results["纯多头"]).cumprod(),
        "long_short": (1 + results["多空"]).cumprod(),
        "benchmark": (1 + bench).cumprod(),
    }).to_csv(os.path.join(HERE, "equity_curve.csv"))
    sig.to_csv(os.path.join(HERE, "signals.csv"))
    gr.to_csv(os.path.join(HERE, "rotation_groups.csv"))
    with open(os.path.join(HERE, "report.txt"), "w") as f:
        f.write(text)
    print(f"\n已保存: equity_curve.csv / signals.csv / rotation_groups.csv / report.txt")


if __name__ == "__main__":
    main()
