# -*- coding: utf-8 -*-
"""GTJA 情绪指数研究 → 模拟盘择时策略 wrapper(包装器)

原研究文件 gtja_sentiment_strategy.py 一行不动；本文件把它接到
paper trading 平台的 on_bar(ctx, bar) 驱动上，作为 timing strategy(择时策略) 使用。

工作方式:
  1. 首次调用时加载研究模块, 读取同目录 data.csv(没有则用合成数据演示流程);
  2. 用研报第 4 节的滚动拼接 PCA 算 GMX/GMVX(无前视),
     再用 5.2+5.3 的组合规则得到周频仓位序列(0/1);
  3. 每个 bar 把 bar 时间映射到最近一个已发布的周仓位, 翻译成 TimingDecision:
       仓位 1 → 允许开仓(hold), 仓位 0 → 禁止开仓(close_all)。

用法: 在平台「择时」页导入本文件, 绑定到任意选股策略(如沪深300 ETF 510300 的策略),
情绪面风险时 BUY 会被 gate 拦截并写入审计链。
"""

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

RESEARCH_DIR = Path("/Users/chenyangsun/Documents/quant/CCF/情绪因子研究/GTJA_report")
DATA_CSV = RESEARCH_DIR / "data.csv"

_position = None      # 周频仓位序列缓存, 整个 run 只算一次
_data_mode = None


def _load_position() -> pd.Series:
    global _position, _data_mode
    if _position is not None:
        return _position
    if str(RESEARCH_DIR) not in sys.path:
        sys.path.insert(0, str(RESEARCH_DIR))
    import gtja_sentiment_strategy as gtja

    # 研究代码里的 print 必须拦住: worker 通过 stdout 回传 JSON, 混入文本会解析失败。
    with redirect_stdout(io.StringIO()):
        if DATA_CSV.exists():
            df = gtja.load_weekly_data(str(DATA_CSV))
            _data_mode = "data.csv"
        else:
            df = gtja.generate_demo_data()
            _data_mode = "demo(合成数据, 仅验证流程)"
        idx_roll = gtja.compute_indices_rolling(df)
        _position = gtja.combined_position(idx_roll["GMX"], idx_roll["GMVX"])
    return _position


def on_init(ctx):
    pos = _load_position()
    ctx.log(
        "INFO",
        f"GTJA sentiment timing ready: {len(pos)} weeks "
        f"({pos.index[0].date()} ~ {pos.index[-1].date()}), data={_data_mode}",
    )


def on_bar(ctx, bar):
    pos = _load_position()
    ts = pd.Timestamp(str(bar["timestamp"]))
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)

    # 取最近一个已发布的周仓位(<= 当前 bar 时间), 周频信号天然无前视。
    value = pos.asof(ts)
    if pd.isna(value):
        ctx.log("WARNING", f"no sentiment position before {ts.date()}, gate stays closed")
        return
    week = pos.index.asof(ts)
    weeks_stale = max(int((ts - week).days // 7), 0)

    allow = bool(value > 0)
    ctx.set_decision(
        allow_open=allow,
        position_policy="hold" if allow else "close_all",
        target_exposure=float(value),
        reason=(
            "情绪面允许做多(GMVX>0 或 GMX 超卖机会)"
            if allow
            else "情绪面风险(GMVX<0 或 GMX 过热), 禁止开仓"
        ),
        metadata={"signal_week": str(week.date()), "weeks_stale": weeks_stale, "data": _data_mode},
    )
