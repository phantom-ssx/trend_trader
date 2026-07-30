# trend_trader

基于 `uv` 和 `nautilus-trader` 的可扩展趋势交易项目，面向多交易所设计，当前首先支持 OKX。

## 结构

- `src/trend_trader/data/okx_candles.py`：下载 OKX 合约 K 线、清洗、保存 Parquet。
- `src/trend_trader/strategies/demo_ema_cross.py`：一个简单 EMA 交叉 demo 策略。
- `src/trend_trader/backtest/run_backtest.py`：读取 Parquet 并启动 Nautilus 回测。
- `src/trend_trader/live/paper.py`：模拟盘配置骨架。
- `src/trend_trader/live/live.py`：实盘配置骨架，默认不允许直接下单。
- `configs/`：回测、模拟盘、实盘示例配置。

## 安装

```bash
uv sync --extra dev
```

## 下载并清洗 OKX 合约 K 线

示例下载 `BTC-USDT-SWAP` 1 分钟 K 线：

```bash
uv run trend-trader-download \
  --inst-id BTC-USDT-SWAP \
  --bar 1m \
  --start 2024-01-01T00:00:00Z \
  --end 2024-01-03T00:00:00Z
```

不指定 `--out` 时会自动生成文件名：

`data/clean/okx/BTC-USDT-SWAP/BTC-USDT-SWAP_1m_20240101T000000Z_20240103T000000Z.parquet`

如果目标文件已存在，命令会提示是否覆盖。需要非交互覆盖时加 `--overwrite`。

下载源默认是 OKX 专用 REST：

```bash
uv run trend-trader-download \
  --source okx-rest \
  --inst-id ETH-USDT-SWAP \
  --bar 1m \
  --start 2026-01-01T00:00:00Z \
  --end 2026-07-07T12:23:02Z
```

也可以使用 `ccxt` 统一接口：

```bash
uv run trend-trader-download \
  --source ccxt \
  --inst-id ETH-USDT-SWAP \
  --bar 1m \
  --start 2026-07-06T00:00:00Z \
  --end 2026-07-07T00:00:00Z
```

当前实测 OKX 专用 REST 更快；`ccxt` 更适合未来扩展其它交易所或做通用 fallback。

输出字段：

`venue, instrument_id, bar_type, timestamp, open, high, low, close, volume, volume_ccy, volume_quote, confirm`

## 下载 OKX 历史资金费率

```bash
uv run trend-trader-funding-download \
  --inst-id ETH-USDT-SWAP \
  --start 2021-01-01T00:00:00Z \
  --end 2026-07-16T16:00:00Z
```

默认输出到 `data/clean/okx/<inst-id>/<inst-id>_funding_rates.parquet`。OKX 公共接口
目前仅返回最近约三个月的历史记录；即使指定更早的开始时间，输出仍受该保留窗口限制。
输出字段为：

`venue, instrument_id, timestamp, funding_rate, realized_rate, method, formula_type`

## 实时采集 OKX 资金费率

`trend-trader-funding-collector` 同时维护两类 Parquet 数据：

- `funding_snapshot`：通过公共 WebSocket 更新内存状态，在每个 UTC 整分钟写入一次；
- `funding_history`：通过历史接口确认结算后的 `realizedRate`，并将它保存为
  `funding_rate`。

启动时通过 REST 初始化所有 `state=live` 的永续合约，WebSocket 断线会自动重连。
内存状态超过 120 秒未更新时标记为 `stale`，每 5 分钟通过 REST 补偿；合约列表
每小时刷新。历史结算会在预计结算时间后确认，并每小时重新核对默认最近 10 个
UTC 日。历史窗口从“当前 UTC 当日 00:00 减去指定天数”开始计算。

本地单合约冒烟测试：

```bash
uv run trend-trader-funding-collector \
  --once \
  --instrument-id BTC-USDT-SWAP \
  --data-root data/market/v1
```

云服务器常驻运行：

```bash
uv run trend-trader-funding-collector \
  --data-root /data/market/v1
```

不传 `--instrument-id` 时采集全部 live SWAP。`--instrument-id` 仅用于测试。所有
时间列均为带 `UTC` 时区的毫秒时间戳。需要改变默认 10 日窗口时传入
`--history-days T`。文件结构为：

```text
/data/market/v1/
├── funding_snapshot/
│   └── year=2026/
│       └── date=2026-07-28/
│           └── funding_snapshot-2026-07-28.parquet
└── funding_history/
    └── year=2026/
        └── date=2026-07-28/
            └── funding_history-2026-07-28.parquet
```

分钟快照字段：

`venue, instrument_id, snapshot_time, exchange_ts, received_at, funding_rate,`
`next_funding_rate, funding_time, next_funding_time, interest_rate, premium,`
`method, formula_type, data_source, data_status`

实际结算字段：

`venue, instrument_id, funding_time, funding_rate, received_at, method, formula_type`

## 实时采集 OKX 合约持仓量

`trend-trader-open-interest-collector` 通过公共 WebSocket `open-interest` 维护
单合约最新状态，并在每个 UTC 整分钟写入 `open_interest_snapshot`。启动时使用
REST 初始化，状态超过 120 秒未更新时标记为 `stale`，每 5 分钟通过 REST
批量补偿；live 合约列表每小时刷新。

默认同时采集 `SWAP`（永续合约）和 `FUTURES`（交割合约）：

```bash
uv run trend-trader-open-interest-collector \
  --data-root /data/market/v1
```

单合约 REST 冒烟测试：

```bash
uv run trend-trader-open-interest-collector \
  --once \
  --instrument-id BTC-USDT-SWAP \
  --data-root data/market/v1
```

只采集永续合约时使用：

```bash
uv run trend-trader-open-interest-collector \
  --instrument-type SWAP \
  --data-root /data/market/v1
```

文件结构为：

```text
/data/market/v1/
└── open_interest_snapshot/
    └── year=2026/
        └── date=2026-07-28/
            └── open_interest_snapshot-2026-07-28.parquet
```

分钟快照字段：

`venue, instrument_id, instrument_type, instrument_family, base_currency,`
`settle_currency, contract_type, expiration_time, snapshot_time, exchange_ts,`
`received_at, open_interest, open_interest_ccy, open_interest_usd, data_source,`
`data_status`

`open_interest_snapshot` 是单合约当前截面的本地分钟历史。统一查询 API 中已有的
`open_interest` 则来自 OKX Rubik，是币种级 `SWAP + FUTURES` 的 5 分钟汇总，
两者不会自动混合。合约元数据来自同一次 instruments 列表刷新，因此可以直接按
`instrument_family`、`base_currency`、到期时间或合约类型筛选，不依赖解析
`instrument_id`。

## 实时记录 OKX 合约多空比

`trend-trader-long-short-ratio-collector` 通过 OKX Rubik 公共 REST 接口读取
单合约的三类多空比，按交易所原生 5 分钟周期写入
`long_short_ratio_snapshot`：

| `ratio_type` | OKX endpoint | 含义 |
| --- | --- | --- |
| `all_account` | `long-short-account-ratio-contract` | 全市场多头账户数 / 空头账户数 |
| `top_trader_account` | `long-short-account-ratio-contract-top-trader` | 大户多头账户数 / 空头账户数 |
| `top_trader_position` | `long-short-position-ratio-contract-top-trader` | 大户多头持仓价值 / 空头持仓价值 |

采集器在每个 UTC 5 分钟边界轮询；live 合约列表每小时刷新。数据时效按 OKX 返回的
`exchange_ts` 判断，而不是按本地请求时间判断，避免接口停止更新时仍被误标为
`fresh`。

默认同时记录 `SWAP`、`FUTURES` 和全部三类指标：

```bash
uv run trend-trader-long-short-ratio-collector \
  --data-root /data/market/v1
```

单合约 REST 冒烟测试：

```bash
uv run trend-trader-long-short-ratio-collector \
  --once \
  --instrument-id BTC-USDT-SWAP \
  --data-root data/market/v1
```

可用重复的 `--ratio-type` 只采集指定类别：

```bash
uv run trend-trader-long-short-ratio-collector \
  --ratio-type all_account \
  --ratio-type top_trader_position \
  --data-root /data/market/v1
```

文件结构为：

```text
/data/market/v1/
└── long_short_ratio_snapshot/
    └── year=2026/
        └── date=2026-07-29/
            └── long_short_ratio_snapshot-2026-07-29.parquet
```

5 分钟快照字段：

`venue, instrument_id, instrument_type, instrument_family, base_currency,`
`settle_currency, contract_type, expiration_time, snapshot_time, exchange_ts,`
`received_at, bar_type, ratio_type, long_short_ratio, data_source, data_status`

主键为 `venue, instrument_id, ratio_type, snapshot_time`；同一类别的同一截面重复
写入会保留最后一次结果。旧版文件没有 `ratio_type` 时会自动按 `all_account`
读取和迁移。
`long_short_ratio_snapshot` 保存完整采集截面和审计元数据。统一查询 API 中已有的
`long_short_ratio` 仍是精简的研究数据集，两者不会自动混合。

## 统一数据查询 API

上层代码可以通过同一个 `MarketDataClient` 查询不同数据集。时间范围统一为 UTC
的左闭右开区间 `[start, end)`，返回值统一为 Polars `DataFrame`：

```python
from trend_trader.data import MarketDataClient

data = MarketDataClient()

# 查询层优先读取本地数据，缺失区间自动从 OKX 下载并保存
candles = data.candles(
    "ETH-USDT-SWAP",
    "1h",
    "2026-07-01T00:00:00Z",
    "2026-07-02T00:00:00Z",
    venue="OKX",
)

# 使用方不需要指定数据源或文件路径
funding = data.funding_rates(
    "ETH-USDT-SWAP",
    "2026-07-01T00:00:00Z",
    "2026-07-02T00:00:00Z",
    venue="OKX",
)
```

查询层只持久化 1 分钟 K 线；查询 `15m`、`1h`、`1d` 等周期时，会在完整的
本地 1 分钟数据上聚合。K 线按月分区，资金费率按年分区，默认存储根目录为
`data/market/v1`，覆盖区间记录在 `catalog.sqlite`。Parquet 文件本身包含
`venue`、`instrument_id`、`bar_type` 和 `timestamp` 主键列。

异步程序使用 `await data.query_async(DataQuery(...))`。新增交易所或数据库时，
实现 `DataSource` 协议并通过 `data.register(source)` 注册即可，上层 API 不需要改变。

### 衍生品与市场指标

查询层还支持以下数据，仍然遵循“本地优先、缺口下载、自动持久化”：

```python
# 指定合约的标记价相对指数价基差；底层保存 1m
basis = data.contract_basis(
    "ETH-USDT-SWAP", "1h",
    "2026-07-01T00:00:00Z", "2026-07-02T00:00:00Z",
)

# OKX 可回溯接口提供币种级合约持仓量（USD）和成交量（USD）；底层保存 5m
open_interest = data.open_interest(
    "ETH-USDT-SWAP", "1h",
    "2026-07-01T00:00:00Z", "2026-07-02T00:00:00Z",
)

long_short = data.long_short_ratio(
    "ETH-USDT-SWAP", "1h",
    "2026-07-01T00:00:00Z", "2026-07-02T00:00:00Z",
)

taker = data.taker_volume(
    "ETH-USDT-SWAP", "1h",
    "2026-07-01T00:00:00Z", "2026-07-02T00:00:00Z",
)

# 强平是事件数据，没有 bar_type。OKX 公共接口仅保留最近约 3 天。
liquidations = data.liquidations(
    "ETH-USDT-SWAP",
    "2026-07-17T00:00:00Z", "2026-07-18T00:00:00Z",
)
```

市值是全市场资产指标，使用 `venue="GLOBAL"`，按日保存。CoinGecko 当前要求
Demo 或 Pro API key，使用前设置：

```bash
# Demo key
export COINGECKO_API_KEY="..."

# 或 Pro key
export COINGECKO_PRO_API_KEY="..."
```

```python
market_cap = data.market_cap(
    "ETH",
    "2026-07-01T00:00:00Z", "2026-07-10T00:00:00Z",
)
```

新增数据的主要字段：

| 数据类型 | 数值字段 | 本地基础粒度 |
|---|---|---:|
| `contract_basis` | `mark_price, index_price, basis, basis_rate` | `1m` |
| `open_interest` | `open_interest_usd, volume_usd` | `5m` |
| `long_short_ratio` | `long_short_ratio` | `5m` |
| `market_cap` | `market_cap_usd, price_usd, volume_24h_usd` | `1d` |
| `liquidations` | `liquidation_id, side, position_side, bankruptcy_price, size, bankruptcy_loss` | 事件级 |
| `taker_volume` | `buy_volume, sell_volume, net_buy_volume` | `5m` |

周期型指标查询更高周期时，主动买卖量使用求和聚合，其余指标取窗口内最后值。
强平数据使用稳定生成的 `liquidation_id` 去重。受上游历史保留窗口限制且本地没有
覆盖时，查询会明确报出缺失区间，不会静默返回不完整数据。

## 多币种标的池与历史快照

标的池维护独立于单合约行情查询。一次维护会读取 OKX 当前的合约配置、ticker
和持仓量，保存完整截面，重建合约生命周期，然后使用同一时点的信息筛选并保存
交易标的池：

```python
from trend_trader.data import MarketDataClient

data = MarketDataClient()

universe = data.maintain_universe(
    venue="OKX",
    instrument_type="SWAP",
    settle_currency="USDT",
    contract_type="linear",
    min_listing_days=30,
    min_volume_usd_24h=20_000_000,
    min_open_interest_usd=10_000_000,
    max_spread_bps=30,
    top_n=30,
)

instrument_ids = universe["instrument_id"].to_list()
```

也可以通过命令定时维护，推荐每天运行一次：

```bash
uv run trend-trader-universe-update \
  --min-listing-days 30 \
  --min-volume-usd-24h 20000000 \
  --top-n 30
```

本地目录为：

```text
data/market/v1/
├── instruments/             # 每次抓取的完整交易所截面，按日期分区
├── instrument_lifecycle/    # 上线、下线有效区间及推导来源
└── universes/               # 最终可交易集合，按名称、交易所和日期分区
```

查询历史时点只会读取该时点之前最近的快照，不会使用未来快照：

```python
historical = data.trading_universe(
    "2024-01-01T00:00:00Z",
    min_listing_days=30,
    top_n=30,
)

# 直接读取已经固化的标的池，不重新执行筛选
saved = data.saved_universe(
    "2024-01-01T00:00:00Z",
    name="okx_usdt_linear_swaps",
)
```

OKX 合约接口只提供当前截面，不能把今天的数据写成历史快照。因此只有当前时间
可以执行在线刷新；过去缺少快照时，可用已有 1 分钟 K 线的首末时间重建近似
生命周期：

```python
lifecycle = data.instrument_lifecycle(venue="OKX", rebuild=True)
```

生命周期中的 `valid_from_source`、`valid_to_source` 和 `confidence` 会明确标记
时间来自交易所字段、每日快照还是首末 K 线。由 K 线推导的数据只能证明本地
观测到的交易区间，精度不等同于交易所官方上下线时间。

### 批量维护成交量前 N 个合约

下面的命令刷新当前仍在交易的 USDT 线性永续合约，按 24 小时美元成交额选出
前 50，并从各合约的 `max(2020-01-01, 上市时间)` 开始维护全部已支持数据：

```bash
uv run trend-trader-history-download \
  --start 2020-01-01T00:00:00Z \
  --top-n 50
```

任务按数据类型、合约和月份独立保存，可安全中断并重复执行。已完整覆盖的区间
会直接跳过，接口只能返回部分记录时也会保存实际取得的数据。执行状态保存在
`data/market/v1/maintenance/top_volume_history.json`。资金费率只请求最近约 92 天、
强平只请求最近 3 天；其他指标会尽量向前回补。市值数据需要 CoinGecko API key，
且尚未配置 symbol 到 CoinGecko coin id 的币种会记录失败，不会中止其他下载。

## 回测

```bash
uv run trend-trader-backtest --config configs/backtest.example.toml
```

对比 MA 交叉过滤策略的整体或月度表现：

```bash
uv run python scripts/evaluate_filters.py \
  --config configs/backtest.eth-2026.toml \
  --mode monthly \
  --format table
```

直接比较 ETH 小时线上的过滤条件：

```bash
uv run python scripts/evaluate_eth_filters.py --limit 30
```

也可以通过 `--ma-pairs` 同时比较多组均线。均线对使用 `快线:慢线` 格式，
每一组都会计算裸交叉策略和 `spread > 0.35% + ATR >= 0.50%` 过滤策略：

```bash
uv run python scripts/evaluate_eth_filters.py \
  --ma-pairs 5:20,6:24,8:20,8:24,10:30 \
  --limit 20
```

结果按收益回撤比排序，并输出收益、最大回撤、手续费、交易次数、胜率、
盈亏比、平均盈利/亏损，以及盈利和亏损的最大值、最小值和总体方差。
单笔交易盈亏包含开仓费、平仓费及持仓期间同方向再平衡产生的手续费；
亏损以负数表示，因此 `min_loss` 是最严重的单笔亏损，方差单位为 `USDT²`。

在 2020–2026 年度数据上比较小时级 MA8/20 固定过滤和自适应过滤：

```bash
uv run python scripts/evaluate_eth_hourly_adaptive.py
```

自适应版本使用 `abs(MA8 - MA20) / ATR14` 归一化均线距离，以过去 90 天
ATR 的滚动分位数描述相对波动状态，并用过去 24 小时的价格趋势效率排除部分
无方向波动。只有市场状态合格时才接受新的多空信号；所有指标只使用当时及
之前的数据，信号在当前小时收盘确认后，于下一根小时 K 线开盘成交。脚本默认
输出稳定参数区域内的三套相邻模板，方便观察结果是否依赖单一参数点。
此外还会输出 `adaptive_balanced_direction_30d_60d`：只有价格同时高于
30 日和 60 日均线时才允许做多，同时低于两条均线时才允许做空，位于两者
之间或均线尚未完成预热时保持空仓。长期方向也只使用当前及历史收盘价。

检验成交量能否改善当前小时级 MA5/20 独立退出策略：

```bash
uv run python scripts/evaluate_eth_hourly_volume_filters.py
```

脚本比较相对历史中位成交量、滚动成交量分位和方向性量流等因果特征。
参数按 2020–2023 年训练期表现排序，2024–2026 年作为样本外区间，并默认计入
单边 `0.05%` 手续费和 `5 bps` 滑点。完整参数网格写入
`outputs/eth_hourly_volume_filters.csv`；这些结果用于研究，不会自动改变实盘策略。

比较固定 10 小时冷却与成交量驱动的再入场解锁机制：

```bash
uv run python scripts/evaluate_eth_hourly_volume_cooldown.py
```

该实验包含累计相对成交量“成交量时钟”、放量交叉、缩量解锁以及先缩量后放量
四类规则，并额外输出逐年扩展窗口选择结果。成交量规则仅替换退出后的冷却机制，
首次入场、均线、ATR、退出和仓位规则保持不变。

比较小时级 MA5/20 全仓与分批加仓策略：

```bash
uv run python scripts/evaluate_eth_hourly_pyramiding.py
```

分批版本在均线距离首次突破 0.35% 且 ATR/价格不低于 0.5% 时建立首仓，
随后只在同方向均线距离继续扩大时加仓，总敞口不超过 1 倍权益；反向信号前
不会因距离回落而减仓。所有目标仓位都在 K 线收盘后确定，并于下一小时开盘
成交。输出包含 2020–2026 连续复利结果及逐年独立结果，手续费默认按每次
成交 0.05% 计算。

在 15 分钟 K 线上比较同一批均线组合：

```bash
uv run python scripts/evaluate_eth_15m_filters.py \
  --ma-pairs 5:20,6:24,8:20,8:24,10:30 \
  --include-time-equivalent \
  --limit 20
```

`--include-time-equivalent` 会额外测试周期乘以 4 的组合，例如小时线的
MA5/20 对应 15 分钟线的 MA20/80，使两者覆盖相同的实际时间长度。输出指标与
小时线脚本一致，便于直接比较。15 分钟线交易更频繁，默认手续费仍为每次成交
`0.05%`，可用 `--fee-rate` 修改。

可以围绕指定均线组合搜索 spread 和 ATR 阈值组合：

```bash
uv run python scripts/evaluate_eth_15m_filters.py \
  --ma-pairs 20:72,20:80,24:80,24:96,28:80,28:96,32:80,32:96,40:120 \
  --filter-grid \
  --spread-thresholds 0.001,0.0015,0.002,0.0025,0.003,0.0035 \
  --atr-thresholds 0.0015,0.002,0.0025,0.003,0.004,0.005 \
  --limit 30
```

使用 `--start` 和 `--end` 可以进行分段或样本外验证；`start` 包含、`end` 不包含。

按自然月详细对比筛选出的三套 15 分钟候选策略：

```bash
uv run python scripts/evaluate_eth_15m_monthly.py
```

月度脚本分别计算每个月的收益、回撤、手续费、胜率、盈亏比，以及单笔盈利和
亏损的均值、极值与总体方差。每个月独立从 `10,000 USDT` 开始，便于比较不同
月份的策略适应性。

为三套候选策略比较不同的固定止损比例：

```bash
uv run python scripts/evaluate_eth_15m_stop_losses.py \
  --stop-losses 0.005,0.01,0.015,0.02,0.03,0.04,0.05,0.06,0.08,0.10,0.12
```

止损从开仓后的下一根 K 线开始按 `high/low` 触发；发生跳空时使用开盘价和
止损价中更差的价格成交，并计入平仓手续费。

用 NautilusTrader 执行当前效果最好的 `spread_0.35%+ATR` 小时级全仓策略：

```bash
uv run trend-trader-backtest \
  --config configs/backtest.eth-2026.toml \
  --resample 1h \
  --strategy best-filter \
  --sizing all-in
```

执行带独立退出和冷却规则的小时级 MA5/20 策略：

```bash
uv run trend-trader-backtest \
  --config configs/backtest.eth-hourly-ma5-ma20-exit.toml \
  --resample 1h \
  --strategy hourly-exit-filter \
  --fast-period 5 \
  --slow-period 20 \
  --spread-threshold 0.0025 \
  --exit-threshold 0 \
  --atr-pct-min 0.005 \
  --cooldown-bars 10 \
  --sizing all-in \
  --orders-csv outputs/hourly_ma5_20_exit
```

该策略只用 ATR 过滤开仓；多仓在 spread 回落到零、空仓在 spread 回升到零时
退出，随后等待 10 根小时 K 线再接受新的入场信号。

默认会跳过名义金额低于 `50 USDT` 的小额调仓订单；可用
`--min-order-notional 0` 关闭过滤，或传其它数值调整阈值。

如需查看 Nautilus 回测的订单明细，可导出 CSV：

```bash
uv run trend-trader-backtest \
  --config configs/backtest.eth-2026.toml \
  --resample 1h \
  --strategy best-filter \
  --sizing all-in \
  --orders-csv outputs
```

订单文件会自动命名为 `orders_<执行时间>_<策略名>_<交易标的>.csv`。

不基于 Nautilus 的本地计算回测放在 `scripts/` 下，例如：

```bash
uv run python scripts/local_backtest.py \
  --config configs/backtest.eth-2026.toml \
  --engine sma-cross \
  --resample 1h \
  --sizing all-in
```

## 绘制 K 线图

从 Parquet 文件生成 HTML K 线图：

```bash
uv run trend-trader-chart \
  --parquet data/clean/okx/ETH-USDT-SWAP/ETH-USDT-SWAP_1m_20260101T000000Z_20260707T124753Z.parquet \
  --start 2026-07-06T00:00:00Z \
  --end 2026-07-07T00:00:00Z \
  --ma-periods 5,10,20 \
  --title "ETH-USDT-SWAP 1m 2026-07-06" \
  --out outputs/eth_usdt_swap_1m_20260706_chart.html
```

默认会叠加 MA5、MA10、MA20；可用 `--ma-periods 7,25,99` 自定义，或传 `--ma-periods ""` 关闭均线。

长区间建议重采样后绘图：

```bash
uv run trend-trader-chart \
  --parquet data/clean/okx/ETH-USDT-SWAP/ETH-USDT-SWAP_1m_20260101T000000Z_20260707T124753Z.parquet \
  --start 2026-01-01T00:00:00Z \
  --end 2026-07-07T12:47:53Z \
  --resample 1h \
  --title "ETH-USDT-SWAP 1h 2026 YTD" \
  --out outputs/eth_usdt_swap_1h_2026_ytd_chart.html
```

## 模拟盘 / 实盘

模拟盘和实盘可通过 Bark 接收每根实时 K 线与每笔实际成交的推送。启动前设置
设备推送地址（不要提交包含 device key 的 `.env`）：

```bash
export BARK_URL="https://api.day.app/your-device-key"
```

推送标题会明确标注 `模拟盘` 或 `实盘`。历史 K 线预热和回测不会发送推送；未设置
`BARK_URL` 时推送功能保持关闭。

先复制环境变量：

```bash
cp .env.example .env
```

模拟盘：

```bash
uv run trend-trader-paper --config configs/paper.example.toml
```

默认只做 dry-run：读取配置、检查密钥、构造 Nautilus 的 `OKXDataClientConfig` 和
`OKXExecClientConfig`。确认无误后再启动真实 Nautilus node：

```bash
uv run trend-trader-paper --config configs/paper.example.toml --start
```

启动后，模拟盘策略会先请求 `warmup_bars` 根历史 K 线初始化 MA/ATR。在历史数据
达到指标所需的最小数量之前，策略仍接收实时行情，但不会提交订单；预热失败时会
保持禁用下单并输出错误日志。

实盘入口默认要求显式确认：

```bash
uv run trend-trader-live --config configs/live.example.toml --i-understand-this-is-live
```

实盘同样默认 dry-run。确认配置无误后才加 `--start`：

```bash
uv run trend-trader-live \
  --config configs/live.example.toml \
  --i-understand-this-is-live \
  --start
```

当前项目已按 `nautilus-trader==1.230.0` 验证 OKX adapter 类名：
`OKXDataClientConfig`、`OKXExecClientConfig`、`OKXLiveDataClientFactory`、
`OKXLiveExecClientFactory`。

## 统一因子层

`trend_trader.factors` 基于统一数据查询层按需计算因子，自动加载预热区间，并将
Funding、市值等低频数据向后 as-of 对齐到 K 线时间轴。强平事件按左闭右开 K 线
窗口聚合。任何低频数据都不会向前匹配未来值。结果中的 `timestamp` 是因子的
可用时刻：例如使用 `[00:00, 01:00)` 小时 K 线计算出的值标记为 `01:00`。

```python
from trend_trader.factors import (
    FactorClient,
    FactorRequest,
    FactorSpec,
    OutlierConfig,
    ProcessingConfig,
    StandardizeConfig,
)

factors = FactorClient()
result = factors.query(
    FactorRequest(
        factors=(
            FactorSpec("momentum", {"lookback": "24h"}),
            FactorSpec("ma_spread", {"fast_period": 5, "slow_period": 20}),
            FactorSpec("atr", {"period": 14}),
            FactorSpec("taker_imbalance"),
        ),
        instrument_ids=("BTC-USDT-SWAP", "ETH-USDT-SWAP"),
        bar_type="1h",
        start="2026-07-01T00:00:00Z",
        end="2026-07-10T00:00:00Z",
        processing=ProcessingConfig(
            outlier=OutlierConfig(method="mad", threshold=5),
            standardize=StandardizeConfig(
                method="zscore",
                scope="cross_sectional",
                min_cross_section=2,
            ),
        ),
    )
)

long_frame = result.frame
wide_frame = result.to_wide()
```

当前注册的因子包括：

- 价格趋势：`momentum`、`ma_spread`、`trend_slope`、`breakout`、
  `mean_reversion`。
- 波动率：`historical_volatility`、`atr`、`volatility_change`、
  `up_down_volatility_asymmetry`、`realized_skewness`、`realized_kurtosis`。
- 成交和流动性：`volume_change`、`turnover`、`amihud`、
  `volume_price_divergence`。
- 分钟微观结构：`quarter_hour_volume_pressure`。它用 K 线收盘位置和相对成交额构造
  每 15 分钟边界的成交方向压力；这是 OHLCV 代理，并不等同于逐笔成交的真实订单失衡。
- 衍生品和规模：`funding_rate`、`basis`、`open_interest`、`market_cap`、
  `long_short_ratio`、`liquidation_imbalance`、`taker_imbalance`。

新增分钟因子的研究依据包括 2026 年的
[The Quarter-Hour Effect](https://arxiv.org/abs/2607.09426)（永续合约 15 分钟边界订单流与
中期收益预测）以及 2024 年的
[Time-varying expected returns, conditional skewness and Bitcoin return predictability](https://doi.org/10.1016/j.qref.2024.101868)。
实现只使用当前及历史 K 线；`FactorClient` 仍会把结果时间平移到下一根 K 线开盘，避免
把尚未完成的分钟 K 线用于交易。

处理流水线依次执行非有限值过滤、异常值处理、标准化和可选截面中性化。
中性化暴露使用注册因子的原始值，例如：

```python
from trend_trader.factors import NeutralizeConfig

processing = ProcessingConfig(
    neutralize=NeutralizeConfig(exposures=("market_cap",)),
)
```

结果同时保留 `raw_value` 和最终 `value`。预热不足、截面样本不足或中性化失败时，
`is_valid` 为 false，并通过 `quality_flags` 给出原因；因子层不会默认使用零填补无效值。

## 因子研究层

`trend_trader.research` 将因子与未来可执行收益连接。标签遵循下一根 K 线开盘入场、
持有指定 bar 后按开盘价退出的语义。若信号来自第 `t` 根 K 线，持有 `h` 根的标签为：

```text
open(t + 1 + h) / open(t + 1) - 1
```

`FactorClient` 的结果时间已经是因子可用时刻，因此标签实现中等价为
`open(T + h) / open(T) - 1`，不会再次额外偏移一根 K 线。

```python
from trend_trader.factors import FactorRequest, FactorSpec
from trend_trader.research import (
    ExecutionReturnSpec,
    FactorAnalyzer,
    FactorResearchClient,
)

request = FactorRequest(
    factors=(
        FactorSpec("momentum", {"lookback": "24h"}),
        FactorSpec("atr", {"period": 14}),
        FactorSpec("basis"),
    ),
    instrument_ids=("BTC-USDT-SWAP", "ETH-USDT-SWAP"),
    bar_type="1h",
    start="2026-01-01T00:00:00Z",
    end="2026-04-01T00:00:00Z",
)

research = FactorResearchClient()
dataset = research.build(
    request,
    labels=(
        ExecutionReturnSpec(horizon_bars=1, round_trip_cost_bps=8),
        ExecutionReturnSpec(horizon_bars=4, round_trip_cost_bps=8),
        ExecutionReturnSpec(horizon_bars=24, round_trip_cost_bps=8),
    ),
)

analyzer = FactorAnalyzer(dataset)
report = analyzer.run(
    method="spearman",
    min_cross_section=2,
    quantiles=2,
    stability_period="1mo",
)
```

标准报告包含：

- `summary`：覆盖率、有效样本数、因子和标签描述统计。
- `overall_ic`：全样本 Pearson IC 或 Rank IC。
- `ic_series` / `ic_summary`：逐时点截面 IC、ICIR、t 值和正 IC 比例。
- `factor_returns` / `factor_return_summary`：逐时点单因子截面回归收益及显著性。
- `quantile_returns` / `quantile_spread`：分组收益、胜率、单调性和多空收益。
- `periodic_ic`：按月或其它周期检查稳定性。
- `decay`：比较不同预测周期的 IC 和多空收益衰减。
- `factor_correlation`：因子间相关性，帮助识别冗余因子。
- `autocorrelation`：因子持续性，可作为换手率的近似指标。

单独调用分析方法时可以选择截面或时间序列分组：

```python
rank_ic = analyzer.overall_ic(method="spearman")
monthly_ic = analyzer.periodic_ic(every="1mo")
time_series_groups = analyzer.quantile_returns(
    quantiles=5,
    scope="time_series",
)
```

标签结果同时保存毛收益、扣除预估双边成本后的净收益、入场/退出时间和价格。研究
数据的 `label_value` 默认使用净收益。数据缺口导致退出时间不连续时，标签会标记为
`NON_CONTIGUOUS_HORIZON`，不会跨缺失 K 线错误地执行 `shift`。

训练和验证切分时，可以清除所有退出时间跨越验证起点的训练标签，并设置 embargo：

```python
from datetime import UTC, datetime, timedelta

training, validation = dataset.purged_time_split(
    datetime(2026, 3, 1, tzinfo=UTC),
    embargo=timedelta(hours=24),
)
```

### 因子冗余和独特贡献

两两相关性只能发现直接重复。需要判断一个因子在控制其它因子后是否仍然提供信息时，
显式运行冗余报告：

```python
redundancy = analyzer.redundancy_report(
    method="spearman",
    min_observations=20,
    cluster_threshold=0.8,
)

vif = redundancy.vif
unique = redundancy.unique_contribution
unique_summary = redundancy.unique_contribution_summary
clusters = redundancy.clusters
```

报告包含：

- `vif`：把每个因子对其余因子回归，计算方差膨胀系数。默认将 `VIF >= 5`
  标记为 `MODERATE`，`VIF >= 10` 标记为 `HIGH`。
- `unique_contribution`：每个时点先用控制因子解释目标因子，再计算残差与未来收益的
  条件 IC；同时比较加入目标因子前后的 `R²`，输出 `incremental_r_squared`。
- `unique_contribution_summary`：汇总条件 IC、条件 IC t 值、正 IC 比例以及平均增量
  `R²`。若原始 IC 较高但条件 IC和增量 `R²` 接近零，说明预测能力很可能已被其它
  因子覆盖。
- `clusters`：对绝对相关性超过阈值的因子构建连通分组，并按平均绝对 IC 从每组选择
  一个代表因子。

也可以只检验一个新因子相对于指定基线因子的增量贡献：

```python
incremental = analyzer.unique_contribution(
    target_factors=("momentum[lookback=24h]",),
    control_factors=(
        "ma_spread[fast_period=5,slow_period=20]",
        "trend_slope[period=20]",
    ),
    method="spearman",
    min_observations=20,
)
```

高级冗余分析按时点执行多次截面回归，计算成本高于普通 IC，因此不会在
`FactorAnalyzer.run()` 中默认执行，需要显式调用 `redundancy_report()`。

## 配置驱动的实验

实验明确分成两条独立管线：

- `trend_trader.experiments.factor`：一次只研究一个因子，判断预测能力、覆盖率、稳定性、
  分组单调性和周期衰减。标签使用毛收益，不计算策略 Sharpe、回撤或换手。
- `trend_trader.experiments.strategy`：必须组合至少两个因子，判断扣除交易成本后的组合
  收益、Sharpe、最大回撤和换手率。组合信号的 IC 只作为辅助诊断。

两类配置和命令分别为：

```bash
uv run trend-trader-factor-experiment \
  configs/experiments/factor/momentum_24h_v1.yaml

uv run trend-trader-strategy-experiment \
  configs/experiments/strategy/multi_factor_linear_v1.yaml
```

原 `trend-trader-experiment` 命令保留为兼容入口，会根据 YAML 是 `factor` 还是
`factors + combination` 自动分发。

默认要求 Git 工作区干净；这样数据库中的 commit 能唯一定位代码。临时调试可在配置中
设置 `experiment.allow_dirty_git: true`，此时仍会记录 dirty 状态和因子源码哈希，但不适合
作为正式可复现实验。只有策略实验接受成本配置；`fee_bps` 和 `slippage_bps` 均按单边
解释，完整持有期成本为 `2 * (fee_bps + slippage_bps)`。

`start_snapshot` 标的池只使用实验开始时已知的快照并在整个实验期间固定，避免存活者偏差
和未来信息；若本地没有对应历史快照会直接失败。需要精确复跑指定品种时，可使用
`mode: explicit` 和 `instruments` 固定列表。

每次成功实验会写入 `experiments/experiments.sqlite`，其中 `experiment_type` 明确标识
`factor` 或 `strategy`。两类实验分别原子发布不同产物：

```text
factor_<experiment_id>/
├── ic.csv / ic_summary.csv / overall_ic.csv
├── factor_returns.csv / periodic_ic.csv / decay.csv
├── quantile_returns.csv / quantile_spread.csv
└── report.html

strategy_<experiment_id>/
├── component_factor_summary.csv
├── signal_ic.csv / signal_ic_summary.csv
├── combination_weights.csv / combination_diagnostics.json
├── portfolio_returns.csv / portfolio_metrics.csv
├── model.pkl                         # 仅模型组合
└── report.html
```

两个目录都会包含 `config.yaml`、`summary.json`、`data_manifest.json` 和 `universe.csv`。
数据版本是实际查询范围（包含因子预热和标签退出区间）内目录元数据的 SHA-256。摘要和
SQLite 同时保存 Git commit、因子声明版本与源码哈希、完整配置、标的池规则和标签定义。
策略实验另外记录成本、年化收益、Sharpe、最大回撤和换手率；组合成本按每次权重变化
对应的换手收取，持仓不变时不会重复收取完整双边成本。

单品种分钟策略可使用 `time_series_threshold` 在每根分钟 K 线结束后更新信号，并用
滚动标准化后的阈值控制交易频率。例如下面的配置会先平滑 60 根 K 线，再用只包含当前
和历史信号的 90 日窗口计算 z-score；只有信号超过 3.5 个标准差才在下一根 K 线开盘
入场：

```yaml
portfolio:
  mode: time_series_threshold
  signal_smoothing_periods: 60
  signal_standardization_periods: 129600
  signal_standardization_min_periods: 129600
  long_threshold_zscore: 3.5
  short_threshold_zscore: null
```

完整 1 分钟示例见
`configs/experiments/strategy/eth_1m_linear_predict_60m_skew_extreme_long_v3.yaml`。该版本在
2020–2024 开发段筛选因子、并用 2025–2026 样本外区间复核方向稳定性，加入 4 小时
`realized_skewness` 后再提高极端信号门槛。因子时间戳代表
信号完成后的第一个可交易开盘；标签和组合收益均按 next-open 语义计算。零成交量的入场
或退出分钟不参与因子评价，也不允许换仓；已有持仓仍会连续盯市。策略摘要分别记录
`prediction_horizon`（组合训练/预测周期）和 `execution_horizon`（组合收益记账与调仓周期），
避免把“预测未来 60 分钟”误写成“只在第 60 分钟才能交易”。

## 多因子组合层

`trend_trader.combinations` 把多个已完成异常值处理、标准化和可选中性化的因子合成
一个统一信号。组合结果仍是 `ResearchDataset`，因此可以直接复用 IC、分组收益、组合
回测和实验报告。完整线性组合配置见
`configs/experiments/strategy/multi_factor_linear_v1.yaml`。

当前内置方法覆盖常用的主要组合范式：

| `combination.method` | 逻辑 |
|---|---|
| `rule` | 有优先级的多组 AND/OR 条件，可输出多头、空头、过滤或指定因子的分数 |
| `linear` | 固定线性权重、截距、权重归一化和缺失值策略 |
| `rank` | 每个时点先做横截面百分位排名，再按方向和权重合成 |
| `ic_weighted` | 使用已经成熟的历史 Rank IC/Pearson IC 做滚动或半衰期加权 |
| `machine_learning` | Ridge、Elastic Net、随机森林、GBDT、Histogram GBDT 的 walk-forward 预测 |
| `deep_learning` | 可配置多层隐藏层的 MLP walk-forward 预测 |

规则组合可以按顺序定义多个规则，第一条匹配规则生效：

```yaml
combination:
  method: rule
  name: trend_rule_signal
  params:
    rules:
      - conditions:
          - {factor: momentum_24h, operator: gt, value: 0}
          - {factor: ma_trend, operator: gt, value: 0}
        logic: all
        score: 1
      - conditions:
          - {factor: momentum_24h, operator: lt, value: 0}
          - {factor: ma_trend, operator: lt, value: 0}
        logic: all
        score: -1
    default_score: null
```

横截面排名和滚动 IC 权重示例：

```yaml
combination:
  method: rank
  name: rank_score
  params:
    weights: {momentum_24h: 0.5, ma_trend: 0.3, liquidity: -0.2}
    missing: drop
```

或使用动态 IC 权重：

```yaml
combination:
  method: ic_weighted
  name: rolling_ic_score
  training_horizon: 4
  params:
    window: 60
    min_periods: 20
    min_cross_section: 20
    ic_method: spearman
    normalization: sum_abs  # 也支持 equal_sign、long_only、softmax
    half_life: 20
    missing: renormalize
```

机器学习和神经网络必须设置 walk-forward 训练门槛。默认在标签退出价格出现后再等待
1 根 K 线，每个预测时刻只使用满足
`exit_time + label_lag + embargo <= prediction_time` 的样本。这避免观察某根开盘价后，
又假设能够按同一个开盘价成交；`label_lag_bars` 至少为 1：

```yaml
combination:
  method: machine_learning
  name: gbdt_score
  training_horizon: 4
  params:
    model: random_forest  # ridge/elastic_net/gradient_boosting/hist_gradient_boosting
    model_params: {n_estimators: 300, max_depth: 6, random_state: 42}
    min_train_observations: 5000
    min_train_periods: 240
    train_window_periods: 8760
    retrain_every: 168
    embargo_bars: 1
    target_transform: demean  # none/demean/zscore；仅用当次训练窗口估计
```

多层感知机配置：

```yaml
combination:
  method: deep_learning
  name: mlp_score
  training_horizon: 4
  params:
    model_params:
      hidden_layer_sizes: [128, 64, 32]
      activation: relu
      early_stopping: true
      max_iter: 500
    min_train_observations: 5000
    min_train_periods: 240
    retrain_every: 168
```

实验产物会额外保存 `combination_weights.csv`、`combination_diagnostics.json`，模型类
方法还会保存最终的 `model.pkl`。Pickle 只应从可信的本地实验目录加载。组合方式通过
`FactorCombiner` 和 `FactorCombinationRegistry` 注册，便于继续接入 XGBoost、LightGBM、
PyTorch 或自定义优化器，而不改变实验和评价层。
