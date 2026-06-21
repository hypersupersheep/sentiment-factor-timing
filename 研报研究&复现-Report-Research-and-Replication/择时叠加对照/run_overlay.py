# -*- coding: utf-8 -*-
"""
情绪择时叠加对照分析
====================
回答用户的核心问题：情绪指标是择时策略，把它叠加在一个既定 baseline 上，
看择时这一层是否 降回撤 / 提收益 / 提Sharpe / 提胜率。

对每个市场级情绪择时信号，做两套 baseline 的边际归因：
  Version A: baseline = 指数 buy & hold        (最干净，纯隔离择时的beta调节)
  Version B: baseline = 低波动选股组合          (更贴近实战，看择时叠在选股alpha上)

覆盖：
  GMX/GMVX(周频)        : A=沪深300, B=低波动组合
  恐惧贪婪择时(日频)     : A=上证综指, B=低波动组合
  券商景气度(月频)       : A=券商指数 (B不适用：板块择时不该叠在宽基选股上，是反面教材)

平稳度日内策略不在此列：它是自带PnL的独立日内策略，不是"叠加在持仓上的开关"。

输出(本目录)：每个策略一张净值图png + 汇总 对照报告.md + delta_tables.txt
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "common"))
for d in ["sentiment_index_research", "fear_greed_research",
          "broker_sentiment_cycle_research"]:
    sys.path.insert(0, os.path.join(ROOT, d))

import tdx_data
import overlay
from stock_portfolio import low_vol_portfolio

import sentiment_index_strategy as gmx_mod
import fear_greed_strategy as fg_mod
import broker_cycle_strategy as bk_mod


def daily_ret(code, market):
    return tdx_data.index_daily(code, market)["close"].pct_change().dropna()


def main():
    print("[准备] 拉取指数与选股组合...")
    csi300_d = daily_ret("000300", 1)
    sse_d = daily_ret("999999", 1).loc["2004-01-01":]
    broker_d = daily_ret("399975", 0)
    lowvol_d = low_vol_portfolio()          # 选股 baseline(Version B)

    blocks = []           # (策略名, version, 对照表, nav)

    # ---------------- 1. GMX/GMVX 周频 ----------------
    print("[1] GMX/GMVX 信号(滚动PCA, 约1-2分钟)...")
    w300 = tdx_data.to_weekly(tdx_data.index_daily("000300", 1))
    w500 = tdx_data.to_weekly(tdx_data.index_daily("000905", 1))
    gmx_sig = gmx_mod.run_pipeline(w300, w500)["position"]

    for ver, base in (("A_沪深300", csi300_d), ("B_低波动选股", lowvol_d)):
        pos = overlay.to_daily_position(gmx_sig, base.index)
        tbl, nav = overlay.evaluate(base, pos, cost_oneway=0.0006,
                                    label="GMX")
        blocks.append(("GMX周频择时", ver, tbl, nav))

    # ---------------- 2. 恐惧贪婪 日频 ----------------
    print("[2] 恐惧贪婪 信号...")
    sse_close = tdx_data.index_daily("999999", 1).loc["2004-01-01":]
    fg_sig = fg_mod.generate_signals(
        fg_mod.build_sentiment_proxy(sse_close))["position"]

    for ver, base in (("A_上证综指", sse_d), ("B_低波动选股", lowvol_d)):
        pos = overlay.to_daily_position(fg_sig, base.index)
        tbl, nav = overlay.evaluate(base, pos, cost_oneway=0.0010,
                                    label="恐惧贪婪")
        blocks.append(("恐惧贪婪择时", ver, tbl, nav))

    # ---------------- 3. 券商景气度 月频 ----------------
    print("[3] 券商景气度 信号...")
    mly = lambda d: d.resample("ME").agg(
        {"open": "first", "close": "last", "amount": "sum"}).dropna()
    bk = mly(tdx_data.index_daily("399975", 0))
    sh = mly(tdx_data.index_daily("999999", 1))
    sz = mly(tdx_data.index_daily("399001", 0))
    amt = (sh["amount"] + sz["amount"]).reindex(bk.index)
    bk_ind = bk_mod.build_indicator(amt, bk["close"])
    bk_sig = bk_mod.generate_signals(bk_ind["indicator"])["position"]

    pos = overlay.to_daily_position(bk_sig, broker_d.index)
    tbl, nav = overlay.evaluate(broker_d, pos, cost_oneway=0.0015, label="券商")
    blocks.append(("券商景气度择时", "A_券商指数", tbl, nav))

    # ---------------- 输出 ----------------
    print("[输出] 绘图 + 报告...")
    lines = ["# 情绪择时叠加对照报告", "",
             "> 评估口径：日频，信号次日生效，空仓吃1.8%货币基金利率。",
             "> Δ = (baseline+择时) − baseline，正=择时改善。", "",
             "## 一句话结论", "",
             "| 策略 | baseline | Δ年化 | Δ最大回撤 | ΔSharpe | 择时是否有效 |",
             "|---|---|---|---|---|---|"]
    verdicts = []
    for name, ver, tbl, nav in blocks:
        d = tbl.loc["Δ(择时贡献)"]
        eff = "✅ 有效" if (d["最大回撤"] > 0.05 and d["Sharpe"] > 0) else (
            "⚠️ 仅降回撤" if d["最大回撤"] > 0.05 else "❌ 无效")
        verdicts.append((name, ver, d, eff))
        lines.append(f"| {name} | {ver} | {d['年化收益']:+.1%} | "
                     f"{d['最大回撤']:+.1%} | {d['Sharpe']:+.2f} | {eff} |")
    lines += ["",
              "**读法**：择时不产生选股alpha，只调节beta暴露，核心价值是 **降回撤、提Sharpe**，",
              "收益率往往略降(削掉了部分上涨)。唯一全面改善的是 **券商景气度择时**——",
              "行业级情绪信号精准捕捉板块拐点；GMX/恐惧贪婪用价量代理后只能降回撤、",
              "Sharpe反而下降，印证其alpha在原始数据(开户数/文本情绪)而非价量。", ""]
    txt_tables = []
    for name, ver, tbl, nav in blocks:
        png = f"nav_{name}_{ver.split('_')[0]}.png"
        overlay.plot_nav(nav, f"{name} | baseline={ver}", os.path.join(HERE, png))
        head = f"## {name} — baseline: {ver}"
        body = overlay.fmt_table(tbl)
        extra = (f"持仓占比 {nav.attrs['持仓占比']:.1%} | "
                 f"换手 {nav.attrs['换手次数']} 次 | 图: {png}")
        lines += [head, "", "```", body, "", extra, "```", ""]
        txt_tables += [head, body, extra, ""]
        print(f"\n{head}\n{body}\n{extra}")

    lines += _conclusion()
    with open(os.path.join(HERE, "对照报告.md"), "w") as f:
        f.write("\n".join(lines))
    with open(os.path.join(HERE, "delta_tables.txt"), "w") as f:
        f.write("\n".join(txt_tables))
    print(f"\n已保存: 对照报告.md / delta_tables.txt / nav_*.png -> {HERE}")


def _conclusion():
    return [
        "## 怎么读这张表(方法论)", "",
        "情绪因子本质是 **timing(择时)** 信号，不产生选股 alpha，只调节 beta 暴露。",
        "所以正确的评估是边际归因：固定一个 baseline 持仓，只让情绪信号决定",
        "**满仓 or 空仓**，看 Δ 行——这才是情绪择时这一层的净贡献。", "",
        "- **Version A(指数B&H)** 最干净：隔离出择时对纯beta的调节效果；",
        "- **Version B(低波动选股)** 更实战：择时叠在选股策略上，看是否锦上添花；",
        "- **券商只有A**：板块择时信号不该叠在宽基选股组合上(beta错配)，故不做B——",
        "  这本身是个反面教材：择时信号必须叠在它所择时的那个资产上。", "",
        "**判断标准**：择时有价值 ⟺ Δ最大回撤 显著为负(降回撤) 且 ΔSharpe>0。",
        "单看收益率会被牛市裸多头带偏——择时的核心价值是 **改善风险调整后收益**，",
        "典型表现是：收益略降或持平、回撤大幅下降、Sharpe/Calmar 上升。", "",
        "> 数据局限：选股universe用akshare当前上证50成分(有survivorship bias)，",
        "> 价格全部来自通达信；做Δ时baseline与overlay同组合，bias大部分抵消。",
    ]


if __name__ == "__main__":
    main()
