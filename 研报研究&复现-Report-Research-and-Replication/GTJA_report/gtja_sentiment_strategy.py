# -*- coding: utf-8 -*-
"""
国泰君安数量化研究系列之十三《市场情绪指数的建立及应用》(2011-07-11) 复现
================================================================================

复现内容(严格按研报, 不做优化):
  1. 情绪变量集(传统市场 6 变量):
       PE / PB / 换手率(-4, 即滞后 4 周) / 上涨家数占比 / IPO 首日涨幅 / 新增 A 股开户数
  2. 预处理: 移动平均(报告未给窗口, 取 4 周, 与"换手率(-4)"的 4 周尺度一致) -> z-score 标准化
  3. PCA(主成分分析):
       GMX  = 第一主成分 (市场情绪指数, 报告解释度 67.86%, 与沪深300相关 0.80)
       GMVX = 第三主成分 (市场情绪变动指数, 与沪深300周收益正相关)
  4. 指数更新(报告第 4 节): 滚动窗口 + 水平拼接
       m_{t+1} = m_t + (m'_{t+1} - m'_t)
  5. 交易规则(报告第 5 节):
       a) GMX 阈值: mean ± 2*sigma; 首次上穿上界 -> 风险(清仓), 首次下穿下界 -> 机会(做多)
       b) GMVX 方向: GMVX > 0 -> 判断下周上涨(做多), GMVX < 0 -> 判断下周下跌(空仓)
  6. 回测: 周频, t 周收盘出信号, 持有 t+1 周; 输出研报表 6 同口径的 GMVX 分区准确率表

数据接口:
  python gtja_sentiment_strategy.py data.csv     # 用真实周频数据
  python gtja_sentiment_strategy.py              # 无数据时用合成数据演示流程

data.csv 列要求(周频, 每行一个周五):
  date, hs300_close, pe, pb, turnover, advance_ratio, ipo_first_day_return, new_accounts
  - pe/pb:                  全市场(或沪深300)市盈率/市净率
  - turnover:               全市场周换手率(%)
  - advance_ratio:          当周上涨家数占比(0~1)
  - ipo_first_day_return:   当周 IPO 平均首日涨幅(无 IPO 周可向前填充)
  - new_accounts:           当周新增 A 股开户数
  数据源建议: Wind / tushare / akshare; 注意开户数、IPO 数据按真实发布时间对齐, 避免前视偏差。

依赖: numpy, pandas (PCA 用 numpy 特征分解实现, 不依赖 sklearn)
"""

import sys

import numpy as np
import pandas as pd

# ----------------------------- 参数(均来自研报) ------------------------------ #
MA_WINDOW = 4          # 指标移动平均窗口(周); 报告"对指标进行移动平均", 未给窗口, 取4周
TURNOVER_LAG = 4       # 换手率(-4): 报告表2中换手率 MA 后 t-4 期与沪深300相关性最高
SIGMA_MULT = 2.0       # GMX 阈值 = mean ± 2*sigma (报告图37)
ROLL_WINDOW = 156      # 滚动 PCA 窗口 N(周, 约3年); 报告未给 N, 仅给出拼接公式
GMX_PC, GMVX_PC = 0, 2 # GMX=第一主成分, GMVX=第三主成分(0-based)

FACTOR_COLS = ["new_accounts", "pe", "pb", "turnover",
               "ipo_first_day_return", "advance_ratio"]   # 报告表4的指标顺序


# ------------------------------- 数据 ---------------------------------------- #
def load_weekly_data(csv_path: str) -> pd.DataFrame:
    """读取周频数据 CSV, 校验列并按日期排序。"""
    df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date").sort_index()
    need = {"hs300_close", *FACTOR_COLS}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"data.csv 缺少列: {missing}")
    return df


def generate_demo_data(n_weeks: int = 320, seed: int = 42) -> pd.DataFrame:
    """
    合成数据, 仅用于在没有真实数据时跑通全流程。
    构造逻辑: 先生成一条带牛熊周期的指数, 再让各情绪变量 = 指数水平/动量 + 噪声,
    模拟报告中"情绪变量与大盘高度相关"的结构(PE/PB 跟水平, 换手率跟动量等)。
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-06-24", periods=n_weeks, freq="W-FRI")

    # 大盘: 趋势(牛熊切换的正弦) + 随机游走
    cycle = np.sin(np.linspace(0, 3 * np.pi, n_weeks)) * 0.9
    ret = 0.004 * np.gradient(cycle) * n_weeks / 10 + rng.normal(0, 0.03, n_weeks)
    close = 1000 * np.exp(np.cumsum(ret))
    level = (close - close.mean()) / close.std()                  # 标准化价格水平
    mom = pd.Series(ret).rolling(4).mean().fillna(0).to_numpy()   # 短期动量

    def proxy(base, w_level, w_mom, noise):
        return base * (1 + w_level * level + w_mom * mom * 30 + rng.normal(0, noise, n_weeks))

    df = pd.DataFrame({
        "hs300_close": close,
        "pe": proxy(20, 0.45, 0.05, 0.05),
        "pb": proxy(3, 0.45, 0.08, 0.05),
        "turnover": np.clip(proxy(3, 0.30, 0.60, 0.25), 0.3, None),     # 换手率更跟动量
        "advance_ratio": np.clip(0.5 + 0.05 * level + 3.0 * mom + rng.normal(0, 0.12, n_weeks), 0, 1),
        "ipo_first_day_return": np.clip(proxy(0.4, 0.50, 0.10, 0.30), -0.2, None),
        "new_accounts": np.clip(proxy(30, 0.55, 0.20, 0.15), 1, None),  # 万户
    }, index=pd.Index(dates, name="date"))
    return df


# ------------------------------- 指数构建 ------------------------------------ #
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """预处理: 移动平均 -> 换手率取滞后4期 -> 全部 z-score 标准化。"""
    x = df[FACTOR_COLS].rolling(MA_WINDOW).mean()
    # 换手率(-4): 用 4 周前的换手率(报告基于其领先性筛选的处理方式)
    x["turnover"] = x["turnover"].shift(TURNOVER_LAG)
    x = x.dropna()
    return (x - x.mean()) / x.std(ddof=1)


def pca_components(z: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    对标准化矩阵做 PCA: 相关阵特征分解, 按解释度降序返回 (载荷矩阵, 解释度)。
    用 numpy 实现以避免 sklearn 依赖; 数学上与报告做法一致。
    """
    corr = np.cov(z.to_numpy(), rowvar=False)
    eigval, eigvec = np.linalg.eigh(corr)          # eigh: 对称阵, 升序
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    return eigvec, eigval / eigval.sum()


def _fix_sign(score: pd.Series, ref: pd.Series) -> pd.Series:
    """
    PCA 主成分符号不唯一, 固定符号约定:
    令主成分与参考序列正相关(GMX 对沪深300水平, GMVX 对当周收益率), 与报告含义一致。
    """
    aligned = pd.concat([score, ref], axis=1).dropna()
    if aligned.iloc[:, 0].corr(aligned.iloc[:, 1]) < 0:
        return -score
    return score


def compute_indices_full_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    全样本 PCA(报告表4的原始做法, in-sample, 用于复现报告图表/表6统计)。
    返回含 GMX、GMVX 的 DataFrame。
    """
    z = preprocess(df)
    loadings, expl = pca_components(z)
    scores = z.to_numpy() @ loadings

    out = pd.DataFrame(index=z.index)
    out["GMX"] = scores[:, GMX_PC]
    out["GMVX"] = scores[:, GMVX_PC]

    close = df["hs300_close"].reindex(z.index)
    week_ret = close.pct_change()
    out["GMX"] = _fix_sign(out["GMX"], close)
    out["GMVX"] = _fix_sign(out["GMVX"], week_ret)

    print(f"[全样本PCA] 第一主成分解释度 {expl[GMX_PC]:.2%} (报告: 67.86%), "
          f"GMX 与沪深300相关 {out['GMX'].corr(close):.2f} (报告: 0.80)")
    print(f"[全样本PCA] GMVX 与当周收益相关 {out['GMVX'].corr(week_ret):.2%} (报告: 26.75%), "
          f"与下周收益相关 {out['GMVX'].corr(week_ret.shift(-1)):.2%} (报告: 24.64%)")
    return out


def compute_indices_rolling(df: pd.DataFrame, window: int = ROLL_WINDOW) -> pd.DataFrame:
    """
    报告第 4 节的可实盘更新方法(滚动窗口 + 水平拼接), 无前视偏差:
      1. 用最近 N 个样本做 PCA, 得到新窗口的主成分序列 m';
      2. 新一期指数 = 上一期已发布指数 + (m'_{t+1} - m'_t);
    每个窗口内重新固定符号(与窗口内沪深300/收益率正相关), 避免主成分符号翻转导致拼接错乱。
    """
    z = preprocess(df)
    close = df["hs300_close"].reindex(z.index)
    week_ret = close.pct_change()

    idx = z.index
    gmx = pd.Series(np.nan, index=idx)
    gmvx = pd.Series(np.nan, index=idx)

    for t in range(window, len(idx)):
        zw = z.iloc[t - window:t + 1]                       # 截至 t 的最近 window+1 个样本
        loadings, _ = pca_components(zw)
        scores = zw.to_numpy() @ loadings
        s_gmx = _fix_sign(pd.Series(scores[:, GMX_PC], index=zw.index),
                          close.loc[zw.index])
        s_gmvx = _fix_sign(pd.Series(scores[:, GMVX_PC], index=zw.index),
                           week_ret.loc[zw.index])
        if np.isnan(gmx.iloc[t - 1]):                       # 首期: 直接采用窗口内序列初始化
            gmx.iloc[t] = s_gmx.iloc[-1]
            gmvx.iloc[t] = s_gmvx.iloc[-1]
        else:                                               # 之后: 只取边际变化量拼接
            gmx.iloc[t] = gmx.iloc[t - 1] + (s_gmx.iloc[-1] - s_gmx.iloc[-2])
            gmvx.iloc[t] = gmvx.iloc[t - 1] + (s_gmvx.iloc[-1] - s_gmvx.iloc[-2])

    return pd.DataFrame({"GMX": gmx, "GMVX": gmvx}).dropna()


# ------------------------------- 交易信号 ------------------------------------ #
def gmx_threshold_signal(gmx: pd.Series, expanding: bool = True) -> pd.Series:
    """
    GMX 阈值信号(报告 5.2):
      首次上穿 mean+2σ -> -1 (情绪过热, 提示风险)
      首次下穿 mean-2σ -> +1 (情绪过冷, 提示机会)
      其余 -> 0
    expanding=True 时用截至当期的扩张均值/标准差(可实盘);
    False 时用全样本均值(严格复现报告图37, 但含前视)。
    """
    if expanding:
        mu = gmx.expanding(min_periods=26).mean()
        sd = gmx.expanding(min_periods=26).std()
    else:
        mu = pd.Series(gmx.mean(), index=gmx.index)
        sd = pd.Series(gmx.std(), index=gmx.index)
    upper, lower = mu + SIGMA_MULT * sd, mu - SIGMA_MULT * sd

    sig = pd.Series(0, index=gmx.index)
    prev = gmx.shift(1)
    cross_up = (prev <= upper.shift(1)) & (gmx > upper)     # "首次上穿": 上期在界内, 本期破界
    cross_dn = (prev >= lower.shift(1)) & (gmx < lower)
    sig[cross_up] = -1
    sig[cross_dn] = 1
    return sig


def gmvx_direction_signal(gmvx: pd.Series) -> pd.Series:
    """GMVX 方向信号(报告 5.3): >0 判断下周上涨(持仓1), <0 判断下跌(空仓0)。"""
    return (gmvx > 0).astype(int)


def combined_position(gmx: pd.Series, gmvx: pd.Series, hold_weeks: int = 4) -> pd.Series:
    """
    可交易组合仓位(报告 5.2 + 5.3 的直接组合, 不做参数优化):
      基础仓位 = GMVX 方向信号(1/0);
      GMX 首次下穿下界 -> 此后 hold_weeks 周强制满仓(均值回复机会);
      GMX 首次上穿上界 -> 此后 hold_weeks 周强制空仓(过热风险)。
    报告未给极端信号的持有期, 取 4 周(与报告"短期"口径一致), 仅为落地所需的最小补充。
    """
    pos = gmvx_direction_signal(gmvx).astype(float)
    thr = gmx_threshold_signal(gmx)
    for i, s in enumerate(thr):
        if s != 0:
            j = min(i + hold_weeks, len(pos))
            pos.iloc[i:j] = 1.0 if s > 0 else 0.0
    return pos


# ------------------------------- 回测与评估 ---------------------------------- #
def gmvx_accuracy_table(gmvx: pd.Series, next_week_ret: pd.Series) -> pd.DataFrame:
    """复现报告表 6: GMVX 不同区间对沪深300下周涨跌方向的预测准确率。"""
    df = pd.concat([gmvx.rename("gmvx"), next_week_ret.rename("ret")], axis=1).dropna()
    correct = np.sign(df["ret"]) == np.where(df["gmvx"] > 0, 1, -1)

    bins = [
        ("全样本",            pd.Series(True, index=df.index)),
        (">=0",               df["gmvx"] >= 0),
        ("<0",                df["gmvx"] < 0),
        (">=1.7",             df["gmvx"] >= 1.7),
        (">=0.75且<1.7",      (df["gmvx"] >= 0.75) & (df["gmvx"] < 1.7)),
        (">=0且<0.75",        (df["gmvx"] >= 0) & (df["gmvx"] < 0.75)),
        (">=-1且<0",          (df["gmvx"] >= -1) & (df["gmvx"] < 0)),
        ("<-1",               df["gmvx"] < -1),
    ]
    rows = []
    for name, mask in bins:
        n = int(mask.sum())
        rows.append({
            "区域": name, "样本数": n,
            "触发率": f"{n / len(df):.2%}" if len(df) else "-",
            "准确率": f"{correct[mask].mean():.2%}" if n else "-",
        })
    return pd.DataFrame(rows)


def backtest(position: pd.Series, close: pd.Series, name: str,
             cost: float = 0.001) -> pd.Series:
    """
    周频回测: t 周收盘确定仓位, 持有 t+1 周 -> 用 position.shift(1) 对齐下周收益,
    避免用当周信号吃当周收益的前视偏差。cost 为单边换仓成本(报告未计成本, 此处为可交易性保留)。
    """
    ret = close.pct_change().reindex(position.index)
    pos = position.shift(1).fillna(0)
    strat_ret = pos * ret - pos.diff().abs().fillna(0) * cost
    equity = (1 + strat_ret.fillna(0)).cumprod()

    n_years = len(strat_ret) / 52
    ann_ret = equity.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else np.nan
    ann_vol = strat_ret.std() * np.sqrt(52)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    mdd = (equity / equity.cummax() - 1).min()
    bench = (1 + ret.fillna(0)).cumprod()

    print(f"\n[{name}] 年化收益 {ann_ret:.2%} | 年化波动 {ann_vol:.2%} | "
          f"Sharpe {sharpe:.2f} | 最大回撤 {mdd:.2%} | 期末净值 {equity.iloc[-1]:.3f} "
          f"(同期基准净值 {bench.iloc[-1]:.3f})")
    return equity


# --------------------------------- 主流程 ------------------------------------ #
def main():
    if len(sys.argv) > 1:
        df = load_weekly_data(sys.argv[1])
        print(f"已加载真实数据: {sys.argv[1]}, {len(df)} 周")
    else:
        df = generate_demo_data()
        print("未提供数据文件, 使用合成数据演示流程 (结果无实际意义, 仅验证逻辑)。\n"
              "实盘请运行: python gtja_sentiment_strategy.py data.csv\n")

    close = df["hs300_close"]

    # --- 1. 全样本 PCA: 复现报告的指数与表6统计 (in-sample) --- #
    idx_full = compute_indices_full_sample(df)
    next_ret = close.pct_change().shift(-1).reindex(idx_full.index)
    print("\n表6复现 | GMVX 分区对下周涨跌的预测 (全样本PCA, in-sample):")
    print(gmvx_accuracy_table(idx_full["GMVX"], next_ret).to_string(index=False))

    # --- 2. 滚动拼接 PCA: 报告第4节的可实盘更新方法 (无前视) --- #
    idx_roll = compute_indices_rolling(df)
    print(f"\n滚动拼接法指数: {len(idx_roll)} 周 "
          f"(窗口 {ROLL_WINDOW} 周, 自 {idx_roll.index[0].date()} 起)")

    # --- 3. 回测三个策略 (均用滚动指数, 可交易口径) --- #
    c = close.reindex(idx_roll.index)
    backtest(gmvx_direction_signal(idx_roll["GMVX"]).astype(float), c,
             "策略A: GMVX>0 做多/否则空仓")

    # GMX 极端阈值: 下穿下界后做多4周, 其余空仓 (纯均值回复择时)
    thr = gmx_threshold_signal(idx_roll["GMX"])
    pos_b = pd.Series(0.0, index=thr.index)
    for i, s in enumerate(thr):
        if s == 1:
            pos_b.iloc[i:min(i + 4, len(pos_b))] = 1.0
    backtest(pos_b, c, "策略B: GMX 下穿下界做多4周")

    backtest(combined_position(idx_roll["GMX"], idx_roll["GMVX"]), c,
             "策略C: GMVX方向 + GMX极端覆盖")

    # --- 4. 落盘 --- #
    out = idx_roll.copy()
    out["GMX_full"] = idx_full["GMX"]
    out["GMVX_full"] = idx_full["GMVX"]
    out["position_combined"] = combined_position(idx_roll["GMX"], idx_roll["GMVX"])
    out.to_csv("sentiment_index_output.csv")
    print("\n指数与仓位已保存至 sentiment_index_output.csv")


if __name__ == "__main__":
    main()
