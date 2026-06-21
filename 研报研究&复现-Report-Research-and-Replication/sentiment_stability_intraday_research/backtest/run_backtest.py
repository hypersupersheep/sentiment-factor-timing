# -*- coding: utf-8 -*-
"""
市场情绪平稳度日内策略回测
==========================
数据：通达信(pytdx) 沪深300指数 1分钟线
      ！免费行情服务器仅保留最近约4-5个月分钟数据，且无 IF 期货主力连续，
      本回测用指数1分钟线替代原文的 IF 当月合约——验证逻辑与近期表现，
      不构成对原文 2010-2015 结果的复刻(详见 策略使用指南.md)。
切分：可得区间前 60% 为"样本内"、后 40% 为"样本外"。
      注意：所有参数(50分钟/9bp阈值/0.5%止损)直接沿用原文，
      未在本地样本内重新优化，因此这里的 IS/OOS 实为两段独立验证窗口。
输出：trades.csv / daily_pnl.csv / report.txt
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
from stability_intraday_strategy import trade_single, trade_dual

ANN = 244


def run_model(bars_by_day, mode):
    """逐日运行模型，返回交易明细 DataFrame[date, session, ret, direction]。"""
    rows = []
    for day, bars in bars_by_day:
        if mode == "single":
            ret, d = trade_single(bars)
            if d != 0:
                rows.append({"date": day, "session": "DAY", "ret": ret, "dir": d})
        else:
            for session, ret, d in trade_dual(bars):
                rows.append({"date": day, "session": session, "ret": ret, "dir": d})
    return pd.DataFrame(rows)


def trade_stats(trades):
    """交易明细统计(对照原文表2/表3口径)。"""
    if trades.empty:
        return {}
    r = trades["ret"]
    win, lose = r[r > 0], r[r <= 0]
    odds = (win.mean() / abs(lose.mean())) if len(lose) and lose.mean() != 0 else np.nan
    return {
        "交易次数": len(r),
        "胜率": (r > 0).mean(),
        "单次平均盈利": win.mean() if len(win) else np.nan,
        "单次平均亏损": lose.mean() if len(lose) else np.nan,
        "赔率": odds,
        "累计收益(单利)": r.sum(),
    }


def main():
    print("[1/3] 拉取沪深300指数1分钟线(仅最近数月可得)...")
    m1 = tdx_data.index_min1("000300", 1)
    m1 = m1[~m1.index.duplicated()]
    days = list(m1.groupby(m1.index.date))
    print(f"  分钟线 {len(m1)} 条, {len(days)} 个交易日: "
          f"{days[0][0]} ~ {days[-1][0]}")

    split_i = int(len(days) * 0.6)
    split_date = pd.Timestamp(days[split_i][0])
    print(f"[2/3] 回测(前60%={split_date.date()}前 为验证窗口A, 后40%为窗口B)...")

    report_lines = ["市场情绪平稳度日内策略回测报告", "=" * 60,
                    f"数据: 沪深300指数1分钟线 {days[0][0]} ~ {days[-1][0]} "
                    f"(共{len(days)}个交易日)",
                    "参数: 50分钟观察 / 平稳度阈值9bp / 止损0.5% / 双边成本2bp",
                    "提示: 参数沿用原文(2010-2012样本内优化值)，本地未调参；",
                    "      指数不可直接交易，实盘需用IF/IM期货或ETF日内(受T+1限制)。", ""]

    all_trades = {}
    for mode, label in (("single", "单次开仓"), ("dual", "两次开仓")):
        trades = run_model(days, mode)
        all_trades[label] = trades
        report_lines.append(f"\n[{label}]")
        if trades.empty:
            report_lines.append("  区间内无满足开仓条件的交易日")
            continue
        trades["date"] = pd.to_datetime(trades["date"])
        seg = {
            "窗口A(前60%)": trades[trades["date"] < split_date],
            "窗口B(后40%)": trades[trades["date"] >= split_date],
            "全区间": trades,
        }
        for name, t in seg.items():
            s = trade_stats(t)
            if not s:
                report_lines.append(f"  {name}: 无交易")
                continue
            report_lines.append(
                f"  {name}: 交易{s['交易次数']}次 | 胜率{s['胜率']:.1%} | "
                f"赔率{s['赔率']:.2f} | 累计{s['累计收益(单利)']:.2%}")
        # 日频净值与回撤
        daily = trades.groupby("date")["ret"].sum()
        stats = perf.perf_stats(daily, ANN)
        report_lines.append(
            f"  日频口径: 年化{stats['年化收益']:.2%} | Sharpe {stats['Sharpe']:.2f} | "
            f"最大回撤{stats['最大回撤']:.2%}")

    text = "\n".join(report_lines)
    print("\n" + text)

    print("\n[3/3] 保存结果...")
    all_trades["两次开仓"].to_csv(os.path.join(HERE, "trades.csv"), index=False)
    pnl = pd.DataFrame({
        label: t.groupby("date")["ret"].sum()
        for label, t in all_trades.items() if not t.empty
    }).fillna(0)
    pnl.to_csv(os.path.join(HERE, "daily_pnl.csv"))
    with open(os.path.join(HERE, "report.txt"), "w") as f:
        f.write(text)
    print(f"已保存: trades.csv / daily_pnl.csv / report.txt -> {HERE}")


if __name__ == "__main__":
    main()
