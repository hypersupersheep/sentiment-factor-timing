# Sentiment Factor Timing · 情绪因子择时研究与复现

> A 股市场情绪因子择时策略的研报复现、回测与优化研究。数据通过通达信(pytdx)拉取，回测严格区分样本内/样本外。
>
> Replication, backtesting and optimization of sentiment-factor timing strategies in the A-share market. Data via TDX (pytdx), with strict in-sample / out-of-sample separation.

---

## 中文

### 项目简介

本项目复现了 **国泰君安、广发证券、光大证券** 三家券商共四篇情绪相关因子择时策略研报，用通达信(pytdx)实盘数据做了含成本、区分样本内外的回测，并在此基础上搭建了"裸策略 vs 叠加情绪择时"的边际归因框架，定量评估情绪择时到底贡献了什么。

仓库分为两大部分：

- **`研报研究&复现 / Report-Research-and-Replication`**：四篇研报的总结、策略复现代码、使用指南、回测结果，以及把情绪信号当择时开关叠加在不同 baseline 上的对照分析(`择时叠加对照/`)。
- **`情绪择时优化 / Sentiment-Timing-Optimization`**：基于复现暴露的问题，下一阶段的优化路线图与实现(进行中)。

### 复现的四篇研报

| 研报 | 券商 / 年份 | 情绪载体 | 频率 / 标的 | 核心方法 |
|---|---|---|---|---|
| 市场情绪指数(GMX/GMVX) | 国泰君安 2011 | 估值/换手/开户数等宏观变量 | 周频 / 沪深300 | PCA 合成情绪指数 + 阈值择时 |
| 在众人恐惧时贪婪 | 国泰君安 2015 | 论坛文本情感分析 | 日频 / 上证综指 + 行业轮动 | 均值±1.5σ 逆向反转 |
| 市场情绪平稳度日内策略 | 广发证券 2015 | 日内价格回撤/反向回撤 | 1分钟 / 股指期货 | 平稳度过滤震荡日 + 趋势跟随 |
| 券商：盈利锚定估值，情绪引领周期 | 光大证券 2020 | 行业估值中的情绪残差 | 月频 / 券商板块 | 收入 nowcast + "收入增速/PB" 估值动量 |

### 策略回测信息

所有回测：参数沿用研报原文、样本外零调参、信号次期开盘成交、含交易成本。原始情绪数据(开户数、论坛文本、两融)不可得处，统一用价量代理变量替代，方法论保持与原文一致。

**情绪择时边际归因表**(Δ = 叠加择时后 − 裸策略 baseline；正值代表择时带来改善)：

| 策略 | baseline | Δ 年化收益 | Δ 最大回撤 | Δ Sharpe | 择时是否有效 |
|---|---|---|---|---|---|
| GMX 周频择时 | 沪深300 buy&hold | −6.3% | **+25.6%(回撤变浅)** | −0.22 | 仅降回撤 |
| GMX 周频择时 | 低波动选股组合 | −10.3% | **+24.2%(回撤变浅)** | −0.32 | 仅降回撤 |
| 恐惧贪婪择时 | 上证综指 buy&hold | −3.1% | −0.1% | −0.11 | 无效 |
| 恐惧贪婪择时 | 低波动选股组合 | −10.8% | −0.0% | −0.41 | 无效 |
| 券商景气度择时 | 券商指数 buy&hold | **+9.4%** | **+31.1%(回撤变浅)** | **+0.48** | 全面有效 |

补充结果：券商景气度策略样本内年化超额 8.8%(原文 11%)、样本外仍有效，最大相对回撤从持有的 −56.8% 压缩到 −13.0%；恐惧贪婪买入信号后 22 日上证平均 +5.1%；平稳度日内策略两次开仓版近 92 个交易日累计 +3.4%(数据过短，仅作通路验证)。

### 主要发现：情绪择时效果并不尽人意

从上表可以看到，**除券商景气度外，情绪择时的效果并不尽人意**——多数情况下只能降低回撤，却以牺牲收益率和 Sharpe 比率为代价。原因可能有以下几点：

1. **情绪指标存在市况依赖性**。不同情绪指标天然适用于不同的策略与市场状态(牛市、熊市、震荡市)。研报自己就指出：估值类指标适用单边市、换手率类适用震荡市与顶部、开户数类适用单边市后半段。用一套固定阈值规则横跨所有市况，必然在不匹配的市场里失效。

2. **代理变量失真**。GMX 的开户数/IPO、恐惧贪婪的论坛文本、券商的两融/股押等原始情绪数据通达信无法获取，复现时用价量数据代理。价量本身已部分反映在价格里，丢失了情绪数据的增量信息——这正是 GMX 与恐惧贪婪复现明显弱于原文、而券商(情绪信号相对独立)复现成功的根本原因。

3. **策略本身的局限性**。阈值类规则的有效信号偏少(GMX 的阈值穿越事件、券商 11 年仅 22 次开仓)，统计显著性不足；恐惧贪婪存在"信号锁死"(买入后情绪未触上界无法翻空，被动穿越下跌)且卖出端基本无效；平稳度日内策略对交易成本极度敏感，股指期货平今手续费足以吞掉全部利润。

4. **算法与参数有待优化**。固定阈值(±2σ / ±1.5σ)、固定窗口长度均为原文样本内的经验设定，没有随市场自适应；0/1 满仓-空仓的仓位映射过于粗暴，在削掉下跌的同时也削掉了大量上涨，是 Sharpe 不升反降的直接原因。

5. **数据与回测的局限**。通达信免费行情的分钟数据仅保留最近数月，日内策略无法做长样本检验；选股 universe 采用当前成分股，存在 survivorship bias；多数研报的样本外区间偏向牛市后段，未经历完整的熊市与震荡市检验。

### 优化思路与下一阶段目标

基于以上问题，优化分两层推进，先夯实基础对照、再针对局限逐项改进。

**第一层 · 系统化回测(先做)**

- **多组参数扫描**：对窗口长度、阈值倍数(σ)、平滑周期做网格遍历，绘制参数平原(parameter plateau)，检验稳健性而非寻找单一最优点；
- **多标的 / 多基数回测**：把每个情绪信号分别叠加到沪深300、中证500、中证1000、创业板指及各行业指数上，看信号对 beta 基数的依赖；
- **分市况拆分回测**：用趋势/波动率把历史切成牛市、熊市、震荡市，分别统计择时效果，定量验证"情绪指标的市况依赖性"假设；
- **连续仓位映射**：用 logistic 或分档映射把情绪强度转成 0~100% 连续仓位，替代 0/1 开关，缓解削峰过度的问题。

**第二层 · 针对局限的改进(待优化)**

- **接入真实情绪数据**：用 akshare / Tushare 获取两融余额、北向资金、真实 PE/PB；用中文金融情感模型(FinBERT 等)从股吧/雪球重建文本情绪，替换价量代理——这是预期改善最大的一笔投入；
- **多因子情绪合成**：将不同时间尺度(日内 / 周频 / 行业周期)的情绪信号视为近似正交的来源，做加权合成或风险平价叠加；
- **自适应阈值与 regime-switching**：引入随波动率自适应的阈值，或用隐马尔可夫/状态切换模型显式建模市况，解决固定规则跨市况失效的问题；
- **修补已知缺陷**：为恐惧贪婪加入时间止损与趋势过滤以解除信号锁死；为日内策略改用隔日平仓或迁移到中证1000股指期货以规避平今成本；
- **组合层叠加**：四个策略时间尺度互补，在单策略成熟后做组合层的风险平价配置，并搭建统一的情绪监控面板。

### 仓库结构

```
sentiment-factor-timing/
├── README.md
├── 研报研究&复现-Report-Research-and-Replication/
│   ├── common/                          # 公共模块：tdx_data(取数) / perf(绩效) / overlay(叠加评估) / stock_portfolio(选股baseline)
│   ├── sentiment_index_research/        # 国泰君安 GMX/GMVX
│   ├── fear_greed_research/             # 国泰君安 恐惧贪婪
│   ├── sentiment_stability_intraday_research/  # 广发 平稳度日内
│   ├── broker_sentiment_cycle_research/ # 光大 券商景气度
│   ├── GTJA_report/                     # GMX 的 akshare 版早期实现(参考)
│   ├── 择时叠加对照/                     # 裸策略 vs 叠加择时 的边际归因 + 净值图
│   └── 情绪因子研究概览.md              # 四篇研报的横向对比与统一框架
└── 情绪择时优化-Sentiment-Timing-Optimization/
    └── ROADMAP.md                       # 下一阶段优化路线图
```

每个研报文件夹下固定四件套：`研报总结.md`(方法论与结论) / `*_strategy.py`(策略核心) / `策略使用指南.md`(回测与实盘) / `backtest/`(回测脚本与结果)。

### 环境与运行

```bash
pip install pytdx pandas numpy matplotlib akshare
# 任一策略回测
cd 研报研究&复现-Report-Research-and-Replication/sentiment_index_research/backtest && python3 run_backtest.py
# 情绪择时叠加对照
cd ../../择时叠加对照 && python3 run_overlay.py
```

数据通过 `common/tdx_data.py` 从通达信免费行情服务器拉取并本地缓存。指数与回测数据全部走通达信；选股组合的成分名单与个股后复权价用 akshare(通达信个股价不复权，除权日会产生假跌)。

### 数据与免责说明

本项目仅用于量化研究与学习。所复现的研报版权归各券商所有，本仓库不含研报 PDF 原文，仅含基于公开方法论的独立实现。回测结果基于价量代理变量，不等同于研报原文精确复现，更不构成任何投资建议。市场有风险。

---

## English

### Overview

This project replicates four sentiment-factor timing research reports from three Chinese brokers — **Guotai Junan, GF Securities, and Everbright Securities** — backtests them on live TDX (pytdx) data with transaction costs and strict in-sample / out-of-sample separation, and builds a marginal-attribution framework ("baseline vs. baseline + sentiment timing") to quantify what the timing layer actually contributes.

The repository has two parts:

- **`研报研究&复现 / Report-Research-and-Replication`**: report summaries, replication code, usage guides, backtest results, and the overlay-attribution analysis (`择时叠加对照/`) that treats each sentiment signal as an on/off switch over different baselines.
- **`情绪择时优化 / Sentiment-Timing-Optimization`**: the optimization roadmap and implementation (work in progress) driven by the weaknesses the replication exposed.

### The Four Reports

| Report | Broker / Year | Sentiment source | Frequency / Asset | Core method |
|---|---|---|---|---|
| Market Sentiment Index (GMX/GMVX) | Guotai Junan 2011 | Macro variables (valuation, turnover, new accounts) | Weekly / CSI 300 | PCA-composited index + threshold timing |
| Be Greedy When Others Are Fearful | Guotai Junan 2015 | Forum-text sentiment analysis | Daily / SSE Composite + sector rotation | Mean ±1.5σ contrarian reversal |
| Intraday Sentiment-Stability Strategy | GF Securities 2015 | Intraday drawdown / reverse-drawdown | 1-min / index futures | Stability filter for choppy days + trend following |
| Brokers: Earnings Anchor Valuation, Sentiment Leads the Cycle | Everbright 2020 | Sentiment residual in sector valuation | Monthly / broker sector | Revenue nowcast + "revenue-growth / PB" valuation momentum |

### Backtest Results

All backtests keep the original report parameters, do zero re-tuning out-of-sample, execute at the next period's open, and include transaction costs. Where raw sentiment data (new accounts, forum text, margin balances) is unavailable, price-volume proxies are used while keeping the methodology faithful to the original.

**Sentiment-timing marginal-attribution table** (Δ = with-timing − baseline; positive means the timing layer improves the metric):

| Strategy | Baseline | Δ Annual Return | Δ Max Drawdown | Δ Sharpe | Timing effective? |
|---|---|---|---|---|---|
| GMX weekly timing | CSI 300 buy & hold | −6.3% | **+25.6% (shallower)** | −0.22 | Only cuts drawdown |
| GMX weekly timing | Low-vol stock portfolio | −10.3% | **+24.2% (shallower)** | −0.32 | Only cuts drawdown |
| Fear-Greed timing | SSE Composite buy & hold | −3.1% | −0.1% | −0.11 | Ineffective |
| Fear-Greed timing | Low-vol stock portfolio | −10.8% | −0.0% | −0.41 | Ineffective |
| Broker-cycle timing | Broker index buy & hold | **+9.4%** | **+31.1% (shallower)** | **+0.48** | Fully effective |

Additional results: the broker-cycle strategy delivers 8.8% in-sample annualized excess return (11% in the original), remains effective out-of-sample, and compresses max relative drawdown from −56.8% (buy & hold) to −13.0%; the fear-greed buy signal is followed by an average +5.1% SSE return over 22 days; the intraday-stability two-entry version returns +3.4% over ~92 trading days (too short a window, used only as a sanity check).

### Key Finding: Sentiment Timing Underperforms

As the table shows, **apart from the broker-cycle signal, sentiment timing underwhelms** — in most cases it only reduces drawdown at the cost of return and Sharpe ratio. Possible reasons:

1. **Sentiment indicators are regime-dependent.** Different indicators naturally suit different strategies and market states (bull, bear, range-bound). The reports themselves note that valuation indicators suit trending markets, turnover suits choppy markets and tops, and new-account data suits the back half of trending markets. A single fixed-threshold rule applied across all regimes is bound to fail in the mismatched ones.

2. **Proxy-variable distortion.** GMX's new-account/IPO data, fear-greed's forum text, and the broker strategy's margin/pledge data are unavailable through TDX, so price-volume proxies are substituted. Price-volume is already partly embedded in prices and loses the incremental information of true sentiment data — exactly why GMX and fear-greed replicate far weaker than the originals, while the broker strategy (whose sentiment signal is relatively independent) replicates successfully.

3. **Strategy-level limitations.** Threshold rules generate few valid signals (GMX threshold crossings; only 22 broker entries in 11 years), so statistical significance is weak; fear-greed suffers "signal lock-in" (after a buy, if sentiment never hits the upper band it cannot flip short, passively riding declines) and its sell side is essentially useless; the intraday-stability strategy is extremely cost-sensitive — index-futures same-day-close fees alone can wipe out all profit.

4. **Algorithm and parameters need optimization.** Fixed thresholds (±2σ / ±1.5σ) and fixed window lengths are in-sample empirical choices that do not adapt to the market; the 0/1 full-position-vs-cash mapping is too blunt — it cuts upside along with downside, directly causing Sharpe to fall rather than rise.

5. **Data and backtest constraints.** TDX free-feed minute data only covers recent months, so intraday strategies cannot be tested on long samples; the stock universe uses current constituents, introducing survivorship bias; most reports' out-of-sample periods skew toward late bull markets and have not faced full bear and range-bound regimes.

### Optimization Plan and Next-Stage Goals

Given these issues, optimization proceeds in two layers — first solidify systematic comparison, then address each limitation.

**Layer 1 · Systematic backtesting (first)**

- **Multi-parameter sweeps**: grid-search window length, threshold multiplier (σ), and smoothing period; plot the parameter plateau to test robustness rather than chase a single optimum;
- **Multi-asset / multi-baseline backtests**: overlay each sentiment signal on CSI 300, CSI 500, CSI 1000, ChiNext, and sector indices to see its dependence on the beta base;
- **Regime-split backtests**: slice history into bull / bear / range-bound regimes via trend and volatility, and measure timing performance in each to quantitatively test the regime-dependence hypothesis;
- **Continuous position mapping**: map sentiment strength to a 0–100% continuous position via logistic or tiered mapping, replacing the 0/1 switch to relieve over-trimming.

**Layer 2 · Limitation-driven improvements (to do)**

- **Bring in real sentiment data**: use akshare / Tushare for margin balances, northbound flows, and true PE/PB; rebuild text sentiment from forums (Guba / Xueqiu) with a Chinese financial sentiment model (e.g., FinBERT) to replace price-volume proxies — the highest-expected-payoff investment;
- **Multi-factor sentiment compositing**: treat signals at different time scales (intraday / weekly / sector-cycle) as near-orthogonal sources and combine them by weighting or risk parity;
- **Adaptive thresholds and regime-switching**: introduce volatility-adaptive thresholds, or model regimes explicitly with hidden-Markov / regime-switching models, to fix fixed-rule failure across regimes;
- **Patch known defects**: add a time stop and trend filter to fear-greed to break signal lock-in; switch the intraday strategy to next-day close or migrate to CSI 1000 index futures to avoid same-day-close costs;
- **Portfolio-layer overlay**: the four strategies are complementary in time scale — once each matures, combine them with risk parity and build a unified sentiment dashboard.

### Repository Structure

```
sentiment-factor-timing/
├── README.md
├── 研报研究&复现-Report-Research-and-Replication/
│   ├── common/                          # Shared: tdx_data / perf / overlay / stock_portfolio
│   ├── sentiment_index_research/        # Guotai Junan GMX/GMVX
│   ├── fear_greed_research/             # Guotai Junan Fear-Greed
│   ├── sentiment_stability_intraday_research/  # GF intraday stability
│   ├── broker_sentiment_cycle_research/ # Everbright broker cycle
│   ├── GTJA_report/                     # Early akshare-based GMX implementation (reference)
│   ├── 择时叠加对照/                     # Baseline vs. timing marginal attribution + NAV charts
│   └── 情绪因子研究概览.md              # Cross-report comparison and unified framework
└── 情绪择时优化-Sentiment-Timing-Optimization/
    └── ROADMAP.md                       # Next-stage optimization roadmap
```

Each report folder follows a fixed four-file layout: `研报总结.md` (methodology and conclusions) / `*_strategy.py` (strategy core) / `策略使用指南.md` (backtest and live usage) / `backtest/` (scripts and results).

### Environment and Usage

```bash
pip install pytdx pandas numpy matplotlib akshare
# Run any strategy backtest
cd 研报研究&复现-Report-Research-and-Replication/sentiment_index_research/backtest && python3 run_backtest.py
# Run the sentiment-timing overlay comparison
cd ../../择时叠加对照 && python3 run_overlay.py
```

Data is pulled from TDX free quote servers via `common/tdx_data.py` and cached locally. Indices and all backtest data come from TDX; the stock-portfolio constituent list and back-adjusted stock prices use akshare (TDX raw stock prices are unadjusted and show fake drops on ex-dividend days).

### Data and Disclaimer

This project is for quantitative research and learning only. Copyright of the replicated reports belongs to the respective brokers; this repository contains no original report PDFs, only independent implementations based on publicly described methodology. Backtest results rely on price-volume proxies, do not constitute an exact replication of the originals, and are not investment advice. Markets carry risk.
