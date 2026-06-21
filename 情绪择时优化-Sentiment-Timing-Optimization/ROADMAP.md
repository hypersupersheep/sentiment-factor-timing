# 情绪择时优化路线图 · Sentiment-Timing Optimization Roadmap

> 本文件夹承接 `研报研究&复现` 暴露的问题，规划并实现下一阶段的优化。当前为规划阶段，代码逐步填充。
>
> This folder continues from the issues exposed in `Report-Research-and-Replication`, planning and implementing the next-stage optimization. Currently at the planning stage; code will be filled in progressively.

---

## 中文

### 出发点

复现结果显示：除券商景气度外，情绪择时多数只能降回撤、却牺牲收益与 Sharpe(详见根目录 README 的边际归因表)。本阶段目标不是急于提升单一收益数字，而是先**系统性地搞清楚每个情绪信号在什么条件下有效**，再针对性改进。

### 第一层 · 系统化回测(优先)

- [ ] **参数平原扫描**：对窗口、阈值倍数(σ)、平滑周期做网格遍历，输出热力图，找稳健区而非单点最优。
- [ ] **多标的 / 多基数**：每个信号叠加到沪深300 / 中证500 / 中证1000 / 创业板指 / 行业指数，量化对 beta 基数的依赖。
- [ ] **分市况拆分**：用趋势 + 波动率把历史切成牛 / 熊 / 震荡，分别统计择时效果，验证"市况依赖性"假设。
- [ ] **连续仓位映射**：logistic / 分档把情绪强度映射成 0~100% 仓位，替代 0/1 开关，缓解削峰过度。

### 第二层 · 针对局限的改进(待优化)

- [ ] **真实情绪数据**：akshare / Tushare 取两融、北向、真实 PE/PB；FinBERT 类模型从股吧 / 雪球重建文本情绪。
- [ ] **多因子情绪合成**：不同时间尺度(日内 / 周 / 行业周期)信号近似正交，做加权或风险平价合成。
- [ ] **自适应阈值 / regime-switching**：波动率自适应阈值，或隐马尔可夫显式建模市况。
- [ ] **缺陷修补**：恐惧贪婪加时间止损 + 趋势过滤解除信号锁死；日内策略改隔日平仓或迁中证1000期指避平今成本。
- [ ] **组合层叠加**：四策略时间尺度互补，单策略成熟后做组合层风险平价 + 统一情绪监控面板。

### 验收标准

每项改进都回到根目录的**边际归因口径**评估：择时有价值 ⟺ Δ最大回撤显著为负(降回撤) **且** ΔSharpe / ΔCalmar > 0。只降回撤、不升风险调整后收益的改动不计为成功。

---

## English

### Motivation

Replication shows that, except for the broker-cycle signal, sentiment timing mostly only cuts drawdown at the cost of return and Sharpe (see the marginal-attribution table in the root README). This stage's goal is not to rush a single return number higher, but first to **systematically establish under what conditions each sentiment signal works**, then improve accordingly.

### Layer 1 · Systematic Backtesting (priority)

- [ ] **Parameter-plateau sweep**: grid-search window, threshold multiplier (σ), and smoothing period; output heatmaps; find robust regions, not single optima.
- [ ] **Multi-asset / multi-baseline**: overlay each signal on CSI 300 / CSI 500 / CSI 1000 / ChiNext / sector indices to quantify dependence on the beta base.
- [ ] **Regime split**: slice history into bull / bear / range-bound via trend + volatility; measure timing per regime to test the regime-dependence hypothesis.
- [ ] **Continuous position mapping**: map sentiment strength to a 0–100% position via logistic / tiered mapping, replacing the 0/1 switch to relieve over-trimming.

### Layer 2 · Limitation-Driven Improvements (to do)

- [ ] **Real sentiment data**: margin balances, northbound flows, true PE/PB via akshare / Tushare; rebuild text sentiment from Guba / Xueqiu with a FinBERT-style model.
- [ ] **Multi-factor sentiment compositing**: treat near-orthogonal signals across time scales (intraday / weekly / sector-cycle) and combine by weighting or risk parity.
- [ ] **Adaptive thresholds / regime-switching**: volatility-adaptive thresholds, or explicit regime modeling via hidden-Markov models.
- [ ] **Defect patches**: add time stop + trend filter to fear-greed to break signal lock-in; switch intraday to next-day close or migrate to CSI 1000 futures to avoid same-day-close costs.
- [ ] **Portfolio-layer overlay**: the four strategies are complementary in time scale — combine with risk parity once each matures, plus a unified sentiment dashboard.

### Acceptance Criterion

Every improvement is evaluated under the root README's **marginal-attribution** convention: timing adds value iff Δ max-drawdown is meaningfully negative (shallower drawdown) **and** Δ Sharpe / Δ Calmar > 0. Changes that only cut drawdown without improving risk-adjusted return do not count as success.
