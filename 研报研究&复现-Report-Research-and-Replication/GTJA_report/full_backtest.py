# -*- coding: utf-8 -*-
"""
GTJA情绪指数策略完整回测系统
================================
严格按照用户要求执行：
- 成交时点: t周五收盘出信号 -> t+1周一开盘成交 (禁止前视偏差)
- 成本: 单边0.06% (佣金0.01%+滑点0.05%), ETF免印花税
- 空仓期: 货币基金利率1.8%年化
- 成本敏感性: 0/5/10/20bp四档
- 基准: 沪深300 buy-and-hold
- 分段报告: 样本内(2005-04~2011-07-11) / 样本外(2011-07-11~至今) / 全区间
- 参数只允许在样本内调试, 样本外只跑一次
"""

import numpy as np
import pandas as pd
import akshare as ak
import warnings
from datetime import datetime, date
warnings.filterwarnings('ignore')

# ============================================================
# 全局参数
# ============================================================
MA_WINDOW = 4          # 移动平均窗口(周)
TURNOVER_LAG = 4       # 换手率滞后
SIGMA_MULT = 2.0       # GMX阈值倍数
ROLL_WINDOW = 156      # 滚动PCA窗口(周)
GMX_PC, GMVX_PC = 0, 2 # GMX=第一主成分, GMVX=第三主成分

DATA_START = "2005-04-08"
IN_SAMPLE_END = "2011-07-11"  # 研报发表日期
BENCHMARK_START = "2005-04-08"

FACTOR_COLS = ["new_accounts", "pe", "pb", "turnover",
               "ipo_first_day_return", "advance_ratio"]

COST_SINGLE_BP = 6.0  # 默认单边成本(bp): 1bp佣金+5bp滑点
COST_SCENARIOS = [0.0, 5.0, 10.0, 20.0]  # 成本敏感性
MM_RATE = 0.018       # 货币基金年化1.8%

print("=" * 80)
print("GTJA情绪指数策略 - 完整回测系统")
print(f"数据起点: {DATA_START} | 样本内截止: {IN_SAMPLE_END}")
print(f"滚动PCA窗口: {ROLL_WINDOW}周 | MA窗口: {MA_WINDOW}周")
print("=" * 80)

# ============================================================
# Part 1: 数据获取
# ============================================================
print("\n[1/6] 获取数据...")

# 1a. CSI 300 日线价格
print("  获取沪深300日线...")
df_idx = ak.stock_zh_index_daily(symbol="sh000300")
df_idx["date"] = pd.to_datetime(df_idx["date"])
df_idx = df_idx.set_index("date").sort_index()
df_idx = df_idx[df_idx.index >= DATA_START]
print(f"    CSI300日线: {len(df_idx)}条, {df_idx.index[0].date()} ~ {df_idx.index[-1].date()}")

# 1b. CSI 300 PE (TTM滚动市盈率)
print("  获取沪深300 PE/PB...")
df_pe = ak.stock_index_pe_lg(symbol="沪深300")
df_pe["日期"] = pd.to_datetime(df_pe["日期"])
df_pe = df_pe.set_index("日期").sort_index()
# 使用滚动市盈率(TTM)
pe_series = df_pe["滚动市盈率"].rename("pe")

df_pb = ak.stock_index_pb_lg(symbol="沪深300")
df_pb["日期"] = pd.to_datetime(df_pb["日期"])
df_pb = df_pb.set_index("日期").sort_index()
pb_series = df_pb["市净率"].rename("pb")
print(f"    PE: {len(pe_series)}条, {pe_series.index[0].date()} ~ {pe_series.index[-1].date()}")
print(f"    PB: {len(pb_series)}条, {pb_series.index[0].date()} ~ {pb_series.index[-1].date()}")

# 1c. 换手率 - 用CSI300成交额作为活跃度代理
print("  计算市场换手率...")
# CSI300 akshare日线数据中volume字段是成交额(元)
# 用5日均成交额 / 252日均成交额 作为换手率活跃度代理
vol_5d = df_idx["volume"].rolling(5).mean()
vol_252d = df_idx["volume"].rolling(252).mean()
turnover_raw = (vol_5d / vol_252d).rename("turnover")
print(f"    换手率代理: 5日/252日均成交额比, 均值{turnover_raw.mean():.2f}")

# 1d. 上涨家数占比 - 基于CSI300日涨跌幅构建非对称代理
print("  估算上涨家数占比...")
# 原理: 当市场大涨时多数股票上涨, 大跌时多数下跌
# 用CSI300日收益率的符号+量级做加权代理
daily_ret_csi = df_idx["close"].pct_change()
# 上涨天数占比 + 收益量级调整 (更贴合实际分布)
adv_signal = (daily_ret_csi.rolling(5).mean() * 10).clip(-2, 2) * 0.125 + 0.5
advance_raw = adv_signal.clip(0.1, 0.9).rename("advance_ratio")
print(f"    上涨占比代理: 均值{advance_raw.mean():.2f}, 范围[{advance_raw.min():.2f}, {advance_raw.max():.2f}]")

# 1e. IPO首日涨幅 - 用市场情绪代理(牛市高、熊市低)
print("  获取IPO数据...")
try:
    df_ipo = ak.stock_ipo_ths()
    if len(df_ipo) > 100:  # 有足够历史数据
        df_ipo['date'] = pd.to_datetime(df_ipo['上市日期'], format='mixed', errors='coerce')
        df_ipo = df_ipo.dropna(subset=['date'])
        if '首日最高涨幅' in df_ipo.columns:
            df_ipo['ret'] = pd.to_numeric(df_ipo['首日最高涨幅'], errors='coerce')
            ipo_daily = df_ipo.set_index('date')['ret'].resample('W-FRI').mean()
        else:
            raise ValueError("No ipo return column")
    else:
        raise ValueError("IPO data too short")
except Exception as e:
    print(f"    Warning: IPO数据不足({e}), 使用市场趋势代理")
    # IPO热度 = 近期市场收益 × 波动率(牛市IPO更热)
    rolling_ret_20 = daily_ret_csi.rolling(20).mean()
    rolling_vol_20 = daily_ret_csi.rolling(20).std()
    # IPO首日涨幅 ≈ 市场趋势情绪 (20-300% range)
    ipo_arr = rolling_ret_20 * 1000 + rolling_vol_20 * 500
    ipo_daily = ipo_arr.clip(-20, 300).rename("ipo")
    print(f"    代理IPO首日涨幅: 均值{ipo_daily.mean():.1f}%, 范围[{ipo_daily.min():.0f}-{ipo_daily.max():.0f}]%")

# 1f. 新增开户数 (分两段: 2005-2015用代理, 2015-04起用真实数据)
print("  获取新增开户数...")
try:
    df_acc = ak.stock_account_statistics_em()
    # 日期格式: "2015-04" -> 解析为月初
    acc_dates = pd.to_datetime(df_acc["数据日期"].astype(str).str.strip() + "-01", format="mixed")
    acc_vals = pd.to_numeric(df_acc["新增投资者-数量"], errors='coerce')
    acc_real = pd.Series(acc_vals.values, index=acc_dates, name="new_accounts")
    acc_real = acc_real.dropna().sort_index()
    print(f"    真实开户数: {len(acc_real)}月, {acc_real.index[0].date()} ~ {acc_real.index[-1].date()}")
except Exception as e:
    print(f"    Warning: 开户数解析失败({e}), 全线使用代理")
    acc_real = pd.Series(dtype=float)

# 2005-2015代理: CSI300成交额活跃度
acc_proxy_arr = (vol_5d / vol_252d.clip(lower=vol_252d.quantile(0.01))) * 500
acc_proxy = acc_proxy_arr.rename("new_accounts")

# 拼接: 2015-04前用代理, 之后用真实(如果有)
if len(acc_real) > 20:
    # 将真实开户数扩展到日频 + 拼接
    acc_real_daily = acc_real.reindex(df_idx.index, method="ffill")
    switch_date = pd.Timestamp("2015-04-01")
    acc_daily = acc_proxy.copy()
    acc_daily[acc_daily.index >= switch_date] = acc_real_daily[acc_real_daily.index >= switch_date]
    acc_monthly = acc_daily
    print(f"    开户数：2005~2015-03用代理, 2015-04~至今用真实, 共{len(acc_monthly.dropna())}日")
else:
    acc_monthly = acc_proxy
    print(f"    开户数：全线使用CSI300成交量代理")

# ============================================================
# Part 2: 数据预处理 -> 周频
# ============================================================
print("\n[2/6] 转换为周频数据(周五收盘)...")

# 将所有日频数据对齐到CSI300指数日期
common_idx = df_idx.index

# 重采样到周五
def to_weekly_friday(daily_series, name):
    """将日频数据重采样到周频(周五收盘), 填入缺失值"""
    s = daily_series.reindex(common_idx)
    weekly = s.resample("W-FRI").last()
    weekly.name = name
    return weekly

close_w = to_weekly_friday(df_idx["close"], "hs300_close")
# 周一开盘价: resample到周五周标签，取周内第一个值(周一)
open_mon_w = df_idx["open"].reindex(common_idx).resample("W-FRI").first()
open_mon_w.name = "hs300_open_mon"
print(f"  周一开盘序列: {len(open_mon_w.dropna())}周")
pe_w = to_weekly_friday(pe_series, "pe")
pb_w = to_weekly_friday(pb_series, "pb")
turnover_w = to_weekly_friday(turnover_raw, "turnover")
advance_w = to_weekly_friday(advance_raw, "advance_ratio")

# IPO: forward fill within 4 weeks for missing
ipo_w = ipo_daily.reindex(common_idx).resample("W-FRI").last()
ipo_w = ipo_w.ffill(limit=4).rename("ipo_first_day_return")

# 开户数: 月频插值到周频
if len(acc_monthly) > 50:  # 有足够月频数据
    acc_daily = acc_monthly.reindex(common_idx, method="ffill")
    acc_w = acc_daily.resample("W-FRI").last().rename("new_accounts")
else:
    acc_w = to_weekly_friday(acc_monthly, "new_accounts")

# 合并
df_w = pd.concat([close_w, open_mon_w, pe_w, pb_w, turnover_w, advance_w, ipo_w, acc_w], axis=1)
# 重命名列以确保一致性
df_w.columns = ["hs300_close", "hs300_open", "pe", "pb", "turnover", "advance_ratio", "ipo_first_day_return", "new_accounts"]

# 前向填充缺失值
df_w = df_w.ffill(limit=4)
# 回填非常早期的缺失
df_w = df_w.bfill(limit=8)
df_w = df_w.dropna()

print(f"  周频数据: {len(df_w)}周, {df_w.index[0].date()} ~ {df_w.index[-1].date()}")
print(f"  各变量NaN数量:\n{df_w.isna().sum()}")

# 确保所有FACTOR_COLS存在
for col in FACTOR_COLS:
    if col not in df_w.columns:
        print(f"  Warning: {col} 缺失, 填充为0")
        df_w[col] = 0.0

# ============================================================
# Part 3: 情绪指数构建 (滚动PCA)
# ============================================================
print("\n[3/6] 构建情绪指数(滚动PCA)...")

def preprocess_raw(df_input):
    """预处理(无z-score): 移动平均 -> 换手率滞后 (不含标准化, 避免全样本前视偏差)"""
    x = df_input[FACTOR_COLS].copy()
    x_ma = x.rolling(MA_WINDOW).mean()
    x_ma["turnover"] = x_ma["turnover"].shift(TURNOVER_LAG)
    return x_ma.dropna()

def zscore_within(df):
    """窗口内z-score标准化"""
    mu = df.mean()
    sd = df.std(ddof=1)
    sd = sd.replace(0, 1)  # 防止除零
    return (df - mu) / sd

def pca_components(z):
    """PCA: 协方差阵特征分解"""
    corr = np.cov(z.to_numpy(), rowvar=False)
    eigval, eigvec = np.linalg.eigh(corr)
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    return eigvec, eigval / eigval.sum()

def fix_sign(score, ref):
    """固定PCA符号: 与参考序列正相关"""
    aligned = pd.concat([score, ref], axis=1).dropna()
    if len(aligned) < 2:
        return score
    if aligned.iloc[:, 0].corr(aligned.iloc[:, 1]) < 0:
        return -score
    return score

# === 全样本PCA (仅用于GMVX分区准确率复现, 不含交易信号) ===
x_full = preprocess_raw(df_w)
z_full = zscore_within(x_full)
loadings_full, expl_full = pca_components(z_full)
scores_full = z_full.to_numpy() @ loadings_full
gmx_full = pd.Series(scores_full[:, GMX_PC], index=z_full.index)
gmvx_full = pd.Series(scores_full[:, GMVX_PC], index=z_full.index)
gmx_full = fix_sign(gmx_full, df_w["hs300_close"].reindex(z_full.index))
gmvx_full = fix_sign(gmvx_full, df_w["hs300_close"].pct_change().reindex(z_full.index))

print(f"  全样本PCA: PC1解释度 {expl_full[GMX_PC]:.2%}, PC3解释度 {expl_full[GMVX_PC]:.2%}")
print(f"  GMX与沪深300相关: {gmx_full.corr(df_w['hs300_close'].reindex(z_full.index)):.3f}")

# === 滚动PCA (纯样本外可用, 窗口内独立z-score, 无前视偏差) ===
x_all = preprocess_raw(df_w)
close_all = df_w["hs300_close"].reindex(x_all.index)
ret_all = close_all.pct_change()

idx_all = x_all.index
gmx_roll = pd.Series(np.nan, index=idx_all)
gmvx_roll = pd.Series(np.nan, index=idx_all)

for t in range(ROLL_WINDOW, len(idx_all)):
    xw = x_all.iloc[t - ROLL_WINDOW:t + 1]
    zw = zscore_within(xw)  # 窗口内独立标准化
    loadings, _ = pca_components(zw)
    scores = zw.to_numpy() @ loadings
    s_gmx = fix_sign(pd.Series(scores[:, GMX_PC], index=zw.index), close_all.loc[zw.index])
    s_gmvx = fix_sign(pd.Series(scores[:, GMVX_PC], index=zw.index), ret_all.loc[zw.index])
    if np.isnan(gmx_roll.iloc[t - 1]):
        gmx_roll.iloc[t] = s_gmx.iloc[-1]
        gmvx_roll.iloc[t] = s_gmvx.iloc[-1]
    else:
        # 增量拼接: 用delta避免不同窗口尺度跳跃
        gmx_roll.iloc[t] = gmx_roll.iloc[t - 1] + (s_gmx.iloc[-1] - s_gmx.iloc[-2])
        gmvx_roll.iloc[t] = gmvx_roll.iloc[t - 1] + (s_gmvx.iloc[-1] - s_gmvx.iloc[-2])

idx_roll = gmx_roll.dropna()
print(f"  滚动PCA: {len(idx_roll)}个有效周(窗口内独立标准化), {idx_roll.index[0].date()} ~ {idx_roll.index[-1].date()}")

# ============================================================
# Part 4: 交易信号与仓位
# ============================================================
print("\n[4/6] 生成交易信号...")

def gmvx_signal(gmvx):
    """GMVX > 0 -> 做多, < 0 -> 空仓"""
    return (gmvx > 0).astype(int)

def gmx_threshold(gmx):
    """GMX极端阈值: 首次上穿mean+2sigma -> -1, 首次下穿mean-2sigma -> +1"""
    mu = gmx.expanding(min_periods=26).mean()
    sd = gmx.expanding(min_periods=26).std()
    upper, lower = mu + SIGMA_MULT * sd, mu - SIGMA_MULT * sd
    sig = pd.Series(0, index=gmx.index)
    prev = gmx.shift(1)
    cross_up = (prev <= upper.shift(1)) & (gmx > upper)
    cross_dn = (prev >= lower.shift(1)) & (gmx < lower)
    sig[cross_up] = -1
    sig[cross_dn] = 1
    return sig

def combined_position(gmx, gmvx, hold_weeks=4):
    """组合仓位: GMVX方向为基础, GMX极端值强制覆盖"""
    pos = gmvx_signal(gmvx).astype(float)
    thr = gmx_threshold(gmx)
    for i, s in enumerate(thr):
        if s != 0:
            j = min(i + hold_weeks, len(pos))
            pos.iloc[i:j] = 1.0 if s > 0 else 0.0
    return pos

# 生成仓位序列
position_raw = combined_position(
    gmx_roll.loc[idx_roll.index],
    gmvx_roll.loc[idx_roll.index]
)

# ============================================================
# Part 5: 回测引擎
# ============================================================
print("\n[5/6] 执行回测...")

def run_backtest(close_series, open_series, position_series, cost_bp=COST_SINGLE_BP,
                 start_date=None, end_date=None, label=""):
    """
    严格回测引擎:
    - t周五收盘出信号, t+1周一开盘成交
    - position.shift(1) + 使用open/close计算周收益
    - 成本按成交金额计算
    """
    # 切分时间段
    if start_date:
        mask = (position_series.index >= pd.Timestamp(start_date))
        position_series = position_series[mask]
    if end_date:
        mask = (position_series.index <= pd.Timestamp(end_date))
        position_series = position_series[mask]

    # 对齐数据
    idx = position_series.index
    c = close_series.reindex(idx)
    o = open_series.reindex(idx)

    # 前一周仓位
    pos_prev = position_series.shift(1).fillna(0)

    # 周收益: 本周五收盘价/上周五收盘价-1 (这是真实可得的)
    # 但用户要求"下周一开盘成交", 所以用 open(t+1)/open(t) 更准确
    # 策略: 空仓期按货币基金利率计息
    weekly_rf = MM_RATE / 52

    # 使用open-to-open收益 (周一开盘成交)
    o_next = o.shift(-1)
    o_prev = o.shift(1).fillna(o)

    # 持仓收益
    price_ret = o_next / o - 1

    # 换手成本(按成交金额)
    turnover_cost = pos_prev.diff().abs() * cost_bp / 10000

    # 策略收益: 持仓收益 + 空仓货币基金 - 成本
    strat_ret = pos_prev * price_ret.fillna(0) + (1 - pos_prev) * weekly_rf - turnover_cost.fillna(0)

    # 净值曲线
    equity = (1 + strat_ret.fillna(0)).cumprod()

    # 基准: 同期的CSI300 buy-and-hold (收盘价)
    b_ret_raw = c.pct_change().fillna(0)
    b_ret = b_ret_raw.reindex(strat_ret.index).fillna(0)
    bench_equity = (1 + b_ret).cumprod()

    # 绩效指标
    n_years = len(strat_ret.dropna()) / 52
    if n_years > 0 and equity.iloc[-1] > 0:
        ann_ret = equity.iloc[-1] ** (1 / n_years) - 1
    else:
        ann_ret = np.nan
    ann_vol = strat_ret.dropna().std() * np.sqrt(52)
    sharpe = (ann_ret - MM_RATE) / ann_vol if ann_vol > 0 else np.nan
    mdd = (equity / equity.cummax() - 1).min()
    if ann_ret is not None and not np.isnan(ann_ret) and mdd < 0:
        calmar = ann_ret / abs(mdd)
    else:
        calmar = np.nan

    # 基准指标
    b_n_years = len(b_ret) / 52
    b_ann_ret = bench_equity.iloc[-1] ** (1 / b_n_years) - 1
    b_ann_vol = b_ret.std() * np.sqrt(52)
    b_sharpe = (b_ann_ret - MM_RATE) / b_ann_vol if b_ann_vol > 0 else np.nan
    b_mdd = (bench_equity / bench_equity.cummax() - 1).min()

    # 年均换手率 (单边)
    n_periods = len(pos_prev)
    total_turnover = pos_prev.diff().abs().sum()
    annual_turnover = total_turnover / n_years if n_years > 0 else np.nan

    # 胜率
    win_rate = (strat_ret.dropna() > 0).mean()

    return {
        "label": label,
        "n_weeks": len(strat_ret.dropna()),
        "n_years": n_years,
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": mdd,
        "calmar": calmar,
        "final_equity": equity.iloc[-1],
        "annual_turnover": annual_turnover,
        "win_rate": win_rate,
        "bench_ann_ret": b_ann_ret,
        "bench_sharpe": b_sharpe,
        "bench_mdd": b_mdd,
        "bench_final": bench_equity.iloc[-1],
        "equity": equity,
        "bench_equity": bench_equity,
        "strat_ret": strat_ret,
    }

# 计算未扣除成本的基准回测(用于成本拖累对比)
result_no_cost = run_backtest(
    df_w["hs300_close"], df_w["hs300_open"], position_raw,
    cost_bp=0.0, label="费前(GMVX+GMX组合)"
)

# 默认成本回测
result_default = run_backtest(
    df_w["hs300_close"], df_w["hs300_open"], position_raw,
    cost_bp=COST_SINGLE_BP, label="费后(6bp单边)"
)

# GMVX单独策略
pos_gmvx = gmvx_signal(gmvx_roll.loc[idx_roll.index])
result_gmvx_only = run_backtest(
    df_w["hs300_close"], df_w["hs300_open"], pos_gmvx,
    cost_bp=COST_SINGLE_BP, label="GMVX单独"
)

# GMX阈值单独策略
thr_only = gmx_threshold(gmx_roll.loc[idx_roll.index])
pos_gmx_only = pd.Series(0.0, index=thr_only.index)
for i, s in enumerate(thr_only):
    if s == 1:
        pos_gmx_only.iloc[i:min(i + 4, len(pos_gmx_only))] = 1.0
result_gmx_only = run_backtest(
    df_w["hs300_close"], df_w["hs300_open"], pos_gmx_only,
    cost_bp=COST_SINGLE_BP, label="GMX阈值单独"
)

# ============================================================
# Part 6: 分段报告
# ============================================================
print("\n[6/6] 生成报告...")

# 切分时间段
in_sample_start = max(idx_roll.index[0], pd.Timestamp(DATA_START) + pd.Timedelta(weeks=ROLL_WINDOW + MA_WINDOW + 4))
in_sample_end = pd.Timestamp(IN_SAMPLE_END)
out_sample_start = pd.Timestamp(IN_SAMPLE_END)

# 分段回测
result_in_sample = run_backtest(
    df_w["hs300_close"], df_w["hs300_open"], position_raw,
    cost_bp=COST_SINGLE_BP,
    start_date=in_sample_start,
    end_date=in_sample_end,
    label="样本内(费后)"
)

result_out_sample = run_backtest(
    df_w["hs300_close"], df_w["hs300_open"], position_raw,
    cost_bp=COST_SINGLE_BP,
    start_date=out_sample_start,
    end_date=None,
    label="样本外(费后)"
)

result_full = run_backtest(
    df_w["hs300_close"], df_w["hs300_open"], position_raw,
    cost_bp=COST_SINGLE_BP,
    start_date=in_sample_start,
    end_date=None,
    label="全区间(费后)"
)

# 基准: 分段
bench_in = run_backtest(
    df_w["hs300_close"], df_w["hs300_open"],
    pd.Series(1.0, index=df_w.index), cost_bp=0,
    start_date=in_sample_start, end_date=in_sample_end,
    label="基准-样本内"
)
bench_out = run_backtest(
    df_w["hs300_close"], df_w["hs300_open"],
    pd.Series(1.0, index=df_w.index), cost_bp=0,
    start_date=out_sample_start, end_date=None,
    label="基准-样本外"
)
bench_full = run_backtest(
    df_w["hs300_close"], df_w["hs300_open"],
    pd.Series(1.0, index=df_w.index), cost_bp=0,
    start_date=in_sample_start, end_date=None,
    label="基准-全区间"
)

# ============================================================
# GMVX分区准确率(样本内, 用全样本PCA复现报告表6)
# ============================================================
gmvx_is = gmvx_full[gmvx_full.index <= in_sample_end]
next_ret = df_w["hs300_close"].pct_change().shift(-1).reindex(gmvx_is.index)

df_gmvx = pd.concat([gmvx_is.rename("gmvx"), next_ret.rename("ret")], axis=1).dropna()
correct = np.sign(df_gmvx["ret"]) == np.where(df_gmvx["gmvx"] > 0, 1, -1)

bins = [
    ("全样本", pd.Series(True, index=df_gmvx.index)),
    (">=0", df_gmvx["gmvx"] >= 0),
    ("<0", df_gmvx["gmvx"] < 0),
    (">=1.7", df_gmvx["gmvx"] >= 1.7),
    (">=0.75且<1.7", (df_gmvx["gmvx"] >= 0.75) & (df_gmvx["gmvx"] < 1.7)),
    (">=0且<0.75", (df_gmvx["gmvx"] >= 0) & (df_gmvx["gmvx"] < 0.75)),
    (">=-1且<0", (df_gmvx["gmvx"] >= -1) & (df_gmvx["gmvx"] < 0)),
    ("<-1", df_gmvx["gmvx"] < -1),
]

# 报告表6原始数据
report_table6 = {
    "全样本": (301, "100.00%", "65.78%"),
    ">=0": (136, "45.18%", "75.74%"),
    "<0": (165, "54.82%", "57.58%"),
    ">=1.7": (7, "2.33%", "28.57%"),
    ">=0.75且<1.7": (51, "16.94%", "80.39%"),
    ">=0且<0.75": (78, "25.91%", "76.92%"),
    ">=-1且<0": (153, "50.83%", "56.21%"),
    "<-1": (12, "3.99%", "75.00%"),
}

print("\n" + "=" * 80)
print("【表6对比】GMVX 分区下周涨跌方向预测准确率 (样本内)")
print("=" * 80)
print(f"{'区域':<18} {'报告N':>6} {'报告触发率':>10} {'报告准确率':>8} {'复现N':>6} {'复现触发率':>10} {'复现准确率':>8} {'差异':>8}")
print("-" * 80)

table6_data = []
for name, mask in bins:
    n = int(mask.sum())
    pct = n / len(df_gmvx) if len(df_gmvx) else 0
    acc = correct[mask].mean() if n else np.nan
    rpt_n, rpt_trigger, rpt_acc = report_table6.get(name, (0, "-", "-"))
    rpt_acc_val = float(rpt_acc.replace("%", "")) / 100 if rpt_acc != "-" else np.nan
    diff = acc - rpt_acc_val if not np.isnan(acc) and not np.isnan(rpt_acc_val) else np.nan
    diff_str = f"{diff:+.2%}" if not np.isnan(diff) else "-"
    print(f"{name:<18} {rpt_n:>6} {rpt_trigger:>10} {rpt_acc:>8} {n:>6} {pct:>9.2%} {acc:>8.2%} {diff_str:>8}")
    table6_data.append({
        "区域": name,
        "报告样本数": rpt_n,
        "报告准确率": rpt_acc,
        "复现样本数": n,
        "复现准确率": f"{acc:.2%}" if not np.isnan(acc) else "-",
        "差异": diff_str
    })

# ============================================================
# 逐年收益表 (样本外)
# ============================================================
print("\n" + "=" * 80)
print("【逐年收益】样本外 (2011-07-11 ~ 至今)")
print("=" * 80)

strat_ret_os = result_out_sample["strat_ret"].dropna()
bench_ret_os = result_out_sample["bench_equity"].pct_change().dropna()
# 对齐
common_os = strat_ret_os.index.intersection(bench_ret_os.index)
strat_ret_os = strat_ret_os.loc[common_os]
bench_ret_os = bench_ret_os.loc[common_os]

print(f"{'年份':<6} {'策略收益':>10} {'基准收益':>10} {'超额':>10} {'策略波动':>10} {'策略Sharpe':>12} {'策略MDD':>10}")
print("-" * 80)

annual_data = []
for yr in range(2011, 2027):
    mask = strat_ret_os.index.year == yr
    if mask.sum() < 10:
        continue
    sr = strat_ret_os[mask].sum()
    br = bench_ret_os[mask].dropna().sum()
    vol = strat_ret_os[mask].std() * np.sqrt(52)
    eq = (1 + strat_ret_os[mask]).cumprod()
    mdd = (eq / eq.cummax() - 1).min()
    sh = (strat_ret_os[mask].mean() * 52 - MM_RATE) / vol if vol > 0 else np.nan
    print(f"{yr:<6} {sr:>10.2%} {br:>10.2%} {sr-br:>+9.2%} {vol:>10.2%} {sh:>12.2f} {mdd:>10.2%}")
    annual_data.append({
        "年份": yr, "策略收益": sr, "基准收益": br, "超额": sr - br,
        "年化波动": vol, "Sharpe": sh, "最大回撤": mdd
    })

# ============================================================
# 成本敏感性分析
# ============================================================
print("\n" + "=" * 80)
print("【成本敏感性分析】")
print("=" * 80)
print(f"{'单边成本':<10} {'年化收益':>10} {'Sharpe':>10} {'最大回撤':>10} {'Calmar':>10} {'年均换手':>10} {'费后vs费前':>12}")
print("-" * 80)

cost_results = []
for c in COST_SCENARIOS:
    r = run_backtest(
        df_w["hs300_close"], df_w["hs300_open"], position_raw,
        cost_bp=c,
        start_date=in_sample_start, end_date=None,
        label=f"成本{c}bp"
    )
    drag = result_no_cost["ann_ret"] - r["ann_ret"] if not np.isnan(r["ann_ret"]) else np.nan
    print(f"{c:>6.0f}bp   {r['ann_ret']:>10.2%} {r['sharpe']:>10.2f} {r['max_dd']:>10.2%} "
          f"{r['calmar']:>10.2f} {r['annual_turnover']:>10.2f} {drag:>+11.2%}" if not np.isnan(r['ann_ret'])
          else f"{c:>6.0f}bp   {'N/A':>10}")
    cost_results.append({"单边成本(bp)": c, **{k: v for k, v in r.items() if k not in ["equity", "bench_equity", "strat_ret"]}})

# ============================================================
# 三段绩效汇总
# ============================================================
print("\n" + "=" * 80)
print("【三段绩效汇总】")
print("=" * 80)
print(f"{'指标':<20} {'样本内':>12} {'样本外':>12} {'全区间':>12} {'基准-样本外':>14}")
print("-" * 80)

metrics = [
    ("周数", "n_weeks", "{:.0f}"),
    ("年数", "n_years", "{:.2f}"),
    ("年化收益", "ann_ret", "{:.2%}"),
    ("年化波动", "ann_vol", "{:.2%}"),
    ("Sharpe", "sharpe", "{:.2f}"),
    ("最大回撤", "max_dd", "{:.2%}"),
    ("Calmar", "calmar", "{:.2f}"),
    ("年均换手", "annual_turnover", "{:.2f}"),
    ("胜率", "win_rate", "{:.2%}"),
    ("期末净值", "final_equity", "{:.3f}"),
]

for name, key, fmt in metrics:
    v_in = result_in_sample[key]
    v_out = result_out_sample[key]
    v_full = result_full[key]
    v_ben = bench_out[key] if key in bench_out else np.nan
    try:
        print(f"{name:<20} {fmt.format(v_in):>12} {fmt.format(v_out):>12} {fmt.format(v_full):>12} {fmt.format(v_ben):>14}")
    except:
        print(f"{name:<20} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>14}")

# 成本拖累
print(f"\n{'成本拖累(费前-费后)':<20} {'-':>12} "
      f"{result_no_cost['ann_ret']-result_out_sample['ann_ret']:>12.2%} "
      f"{result_no_cost['ann_ret']-result_full['ann_ret']:>12.2%} {'-':>14}")

# ============================================================
# 保存结果
# ============================================================
print("\n[保存] 结果落盘...")

# 保存净值曲线
output = pd.DataFrame({
    "date": idx_roll.index,
    "GMX_roll": gmx_roll.loc[idx_roll.index],
    "GMVX_roll": gmvx_roll.loc[idx_roll.index],
    "position": position_raw.loc[idx_roll.index],
    "close": df_w["hs300_close"].reindex(idx_roll.index),
    "equity_no_cost": result_no_cost["equity"].reindex(idx_roll.index),
    "equity_default": result_default["equity"].reindex(idx_roll.index),
    "bench_equity": result_default["bench_equity"].reindex(idx_roll.index),
}).set_index("date")

output.to_csv("backtest_equity_curve.csv")
print("  净值曲线: backtest_equity_curve.csv")

# 保存表6对比
pd.DataFrame(table6_data).to_csv("table6_comparison.csv", index=False)
print("  表6对比: table6_comparison.csv")

# 保存逐年收益
pd.DataFrame(annual_data).to_csv("annual_returns.csv", index=False)
print("  逐年收益: annual_returns.csv")

# 保存成本敏感性
pd.DataFrame(cost_results).to_csv("cost_sensitivity.csv", index=False)
print("  成本敏感性: cost_sensitivity.csv")

# ============================================================
# 最终总结
# ============================================================
print("\n" + "=" * 80)
print("回测完成! 关键结论:")
print("=" * 80)
print(f"\n样本内 (至2011-07-11):  年化 {result_in_sample['ann_ret']:.2%}, Sharpe {result_in_sample['sharpe']:.2f}, MDD {result_in_sample['max_dd']:.2%}")
print(f"样本外 (2011-07-11起):  年化 {result_out_sample['ann_ret']:.2%}, Sharpe {result_out_sample['sharpe']:.2f}, MDD {result_out_sample['max_dd']:.2%}")
print(f"全区间:                  年化 {result_full['ann_ret']:.2%}, Sharpe {result_full['sharpe']:.2f}, MDD {result_full['max_dd']:.2%}")
print(f"基准(样本外):            年化 {bench_out['ann_ret']:.2%}, Sharpe {bench_out['sharpe']:.2f}, MDD {bench_out['max_dd']:.2%}")
print(f"\n费前年化: {result_no_cost['ann_ret']:.2%} | 费后年化: {result_default['ann_ret']:.2%} | 成本拖累: {result_no_cost['ann_ret']-result_default['ann_ret']:.2%}")
print(f"年均换手率: {result_default['annual_turnover']:.2f}")

# 策略在多大成本下失效
print("\n成本失效阈值:")
for c, cr in zip(COST_SCENARIOS, cost_results):
    is_dead = cr.get("sharpe", 999) < 0 or (cr.get("ann_ret", -1) or -1) <= 0
    status = "⚠ 失效" if is_dead else "✓ 有效"
    print(f"  {c:>5.0f}bp 单边: 年化 {(cr.get('ann_ret') or np.nan):.2%}, Sharpe {(cr.get('sharpe') or np.nan):.2f} [{status}]")

print("\nDONE. 所有输出文件已保存。")
