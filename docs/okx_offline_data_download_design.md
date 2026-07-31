# OKX 全合约离线数据下载方案

状态：第一阶段代码已实现，尚未部署、尚未执行生产数据下载
范围：OKX 加密货币类 `SWAP`（永续）与 `FUTURES`（交割），包含历史到期/下架
合约；不含 `OPTION`、非加密资产和 pre-market 阶段
时间基准：规范化数据统一使用 UTC，时间区间统一使用左闭右开 `[start, end)`

## 1. 结论摘要

1. 离线数据与实时 snapshot 必须物理隔离：
   - 新离线数据只写 `/data/market/v1/offline/`。
   - 现有 `open_interest_snapshot`、`funding_snapshot`、
     `funding_history`、`long_short_ratio_snapshot` 保持原路径，不迁移、不与离线
     Parquet 合并。
   - 查询层以后如需提供“最佳可用数据”，只能在读取时合并，并且必须返回
     `dataset_kind` 和 `source_name`。
   - Normalized 层默认按“每个数据集、每个 UTC 日一个 Parquet”保存；同一文件
     包含当日全部 `SWAP + FUTURES` 合约或全部币种，便于做全市场横截面分析。
   - 普通逐笔成交和 L2 是大数据集，不合并成全市场单文件；它们在同一日期目录下
     按完整 `instrument_id` 分文件，合约信息体现在文件名中，不增加合约路径层级。
2. OKX 历史数据中心可批量下载的模块只有：
   - 公共逐笔成交；
   - 普通 1 分钟 K 线；
   - 实际结算资金费率；
   - L2 订单簿；
   - 借币利率（本项目暂不需要）。
3. 标记价格 K 线、指数价格 K 线、聚合 OI、taker volume、多空比不在离线文件
   模块内，只能每天通过公共 REST API 固化为本地离线分区。
4. “每天下载前一天”不能作为完成标准。OKX 官方说明：
   - 逐笔、K 线、资金费率通常 T+2 发布；
   - L2 订单簿通常 T+3 发布。
   因此任务应每天扫描“已经成熟但尚未落盘”的日期，而不是只请求 T-1。
5. 公共历史回补不能使用一个统一起点，必须按数据类型分别回补。最早官方离线
   文件范围为：逐笔 2021-09、K 线 2023-07、资金费率 2022-03、L2 2023-03。
6. 第一阶段不下载普通逐笔成交和 L2：
   - 历史回补和 timer 日常增量均设置为 `enabled = false`；
   - 先完成其他公共数据和私有数据；
   - 第一阶段结束后只做容量测算，由用户再次确认磁盘预算、启用时间和回补起点；
   - 本文仍保留其 schema、路径和校验设计，供第二阶段直接启用。
7. 历史区间采用“各数据集最大可得历史”：
   - 第一阶段每个已启用数据集从各自最早可得时间开始回补，不裁剪到统一起点；
   - 普通逐笔和 L2 继续禁用，本条决策不会隐式启用它们；
   - 分析层根据实际使用的特征集合计算共同有效区间。
8. 其他第一阶段范围已经确认：
   - 聚合 OI 同时保存 `5m`、`1H`、`1D`；
   - 私有数据覆盖主账户及实际参与交易的子账户；
   - 第一阶段 Raw 全部永久保留，不设置自动清理任务。

官方依据：

- [OKX Historical Market Data](https://www.okx.com/historical-data)
- [OKX API 文档](https://www.okx.com/docs-v5/en/)
- [OKX 个人历史报表说明](https://www.okx.com/en-gb/help/how-to-check-download-order-history-position-history-and-trading-history)

## 2. 数据域隔离

### 2.1 现有实时/近实时数据

以下路径继续代表 collector 采集结果，不改变语义：

```text
/data/market/v1/
├── open_interest_snapshot/
├── funding_snapshot/
├── funding_history/
└── long_short_ratio_snapshot/
```

其中 `funding_history` 虽然是最终结算值，但它由实时 collector 通过 REST 补偿取得，
仍属于 collector 数据域；交易所历史文件解析出的结算值写入新的 offline 数据域。

### 2.2 新离线数据

```text
/data/market/v1/offline/
├── raw/                 # 交易所原始 ZIP/CSV/JSON.GZ，不做内容修改
├── normalized/          # 统一字段、统一 UTC 后的 Parquet
├── manifests/           # 每次下载与转换的 JSON manifest
├── quarantine/          # 校验失败或 schema 未识别的文件
└── catalog.sqlite       # 离线文件、覆盖区间、校验和与任务状态
```

强制元数据：

| 字段 | 值 |
|---|---|
| `dataset_kind` | 固定为 `offline` |
| `source_name` | `okx_historical_file`、`okx_public_rest` 或 `okx_private_rest` |
| `schema_version` | 从 `1` 开始 |
| `downloaded_at` | UTC 下载时间 |
| `source_url` | 原始文件或 API URL；私有 API 只保存 endpoint，不保存签名参数 |
| `source_sha256` | 原始文件 SHA-256 |

这些值写在 catalog、manifest 和 Parquet 文件 metadata 中，不在逐笔/L2 每一行重复，
避免显著增加存储量。查询层读取时可投影成普通列。

## 3. 文件格式与分区规则

### 3.1 Raw 层

- 保留 OKX 原始压缩文件，不重新压缩、不改文件名。
- 第一阶段 Raw 永久保留，不设置 TTL、生命周期删除或磁盘满时自动清理。
- REST 响应按日保存为 `json.gz`。
- 下载先写 `.part`，完成大小、ZIP CRC 和 SHA-256 校验后原子改名。
- 原始文件一经验收不可覆盖；若 OKX 后续修订同名文件，按 SHA-256 保存新 revision。

示例：

```text
/data/market/v1/offline/raw/okx/public_trades/
└── source_date=2026-07-28/
    ├── BTC-USDT-SWAP-trades-2026-07-28.zip
    └── manifest.json
```

OKX 模块 1、2、3、11 的“日”按 UTC+8 解释；订单簿模块 4、5、6 的“日”按 UTC
解释。Raw 层保留上游 `source_date` 和 `date_basis`，不得假设所有文件都是 UTC 日。

### 3.2 Normalized 层

- 格式：Apache Parquet。
- 压缩：Zstandard。
- 时间：`Datetime(ms, UTC)`。
- 行情数值：`Float64`，与仓库现有 Polars schema 一致。
- 账户资金数值：`Decimal128(38, 18)`，避免账务金额的二进制浮点误差。
- ID：全部 `Utf8`，禁止转整数。
- 空字符串统一为 null；枚举保存 OKX 原始小写值。
- 上游 UTC+8 文件按事件时间拆成 UTC 日分区。

默认路径：

```text
/data/market/v1/offline/normalized/<dataset>/
└── venue=OKX/
    └── year=YYYY/
        └── date=YYYY-MM-DD/
            └── <dataset>-YYYY-MM-DD.parquet
```

这与现有持仓量数据一样，采用
`year=YYYY/date=YYYY-MM-DD/文件名-日期.parquet`。文件内部通过
`instrument_type`、`instrument_id`、`base_currency`、`index_id` 等列区分合约和
币种，不再按合约拆成多个规范化文件。

普通逐笔成交和 L2 例外：日期仍通过公共目录区分，完整合约 ID 写在文件名中，
不使用 `instrument_id=.../` 或 `base_currency=.../` 路径分区：

```text
/data/market/v1/offline/normalized/public_trades/
└── venue=OKX/year=2026/date=2026-07-28/
    ├── BTC-USDT-SWAP-public_trades-2026-07-28.parquet
    ├── ETH-USDT-SWAP-public_trades-2026-07-28.parquet
    └── BTC-USD-260925-public_trades-2026-07-28.parquet

/data/market/v1/offline/normalized/order_book_l2/
└── venue=OKX/year=2026/date=2026-07-28/
    ├── BTC-USDT-SWAP-order_book_l2-2026-07-28.parquet
    └── ETH-USDT-SWAP-order_book_l2-2026-07-28.parquet
```

文件名必须使用完整 `instrument_id`，不能只使用 `BTC`、`ETH` 等
`base_currency`，否则同一币种的 USDT/USDC/币本位永续和不同到期日交割合约会
重名。每个文件内部仍保留 `instrument_id` 和 `instrument_type` 列，不能只依赖
文件名恢复业务字段。

因此，“一个文件”的准确规则是：

- K 线、资金费率、标记价、指数价、聚合 OI、taker、多空比：一个 dataset
  一天一个全市场文件；
- 普通逐笔和 L2：一个 dataset、一个 UTC 日、一个 `instrument_id` 一个文件；
- 不把不同 dataset 的 schema 混在同一个 Parquet。

币种级和账户级数据分别使用：

```text
# 币种级聚合 OI：同一日文件包含全部 base_currency
.../aggregate_open_interest/venue=OKX/year=.../date=.../
└── aggregate_open_interest-YYYY-MM-DD.parquet

# 指数 K 线：同一日文件包含全部 index_id
.../index_price_candles/venue=OKX/year=.../date=.../
└── index_price_candles-YYYY-MM-DD.parquet

# 私有数据：同一账户、同一数据集、同一日一个文件，包含全部合约和币种
.../private/<dataset>/venue=OKX/account=<account_alias>/year=.../date=.../...
```

所有规范化文件使用分区锁、临时文件和 `os.replace` 原子提交，行为仿照现有
`OpenInterestParquetRepository`。

### 3.3 规范化文件的生成方式

对于需要合并的全市场日文件，不能随着每个合约下载完成就直接 append。正确流程是：

1. 通过正式公开接口 `GET /api/v5/public/market-data-history` 获取下载链接，各
   Raw 文件或 REST 响应先独立下载；
2. ZIP 成员通过 `ZipExtFile` 流式读取，不把解压内容整体载入内存；原始文本行先
   使用常量内存折叠连续且完全相同的重复行，再交给 `csv.DictReader`；
3. 同一主键的相邻记录若内容不同则立即失败并隔离 Raw；原始行数、输出行数、
   连续重复行数和重复率写入日志与 artifact metadata；
4. 每 `stream_batch_rows` 行规范化一次，并批量写入数据盘 `.staging` 目录的临时
   Parquet fragment；
5. DuckDB 在 `compaction_memory_mb` 内存上限内使用业务主键去重，超出内存的排序
   自动 spill 到 `.staging` 数据盘；
6. 按排序游标每批读取固定行数，通过 `PyArrow ParquetWriter` 写入日级临时文件；
7. 校验行数、主键、时间范围和 Parquet footer；
8. 使用 `os.replace` 原子替换正式日文件；
9. 删除可重建的 fragment 和 DuckDB 临时文件，Raw 文件继续永久保留。

不允许把所有合约数据一次性加载进内存。已有日文件由 DuckDB 直接扫描，不经过
Python 逐行转换；最终排序结果通过 Arrow record batch 写入。默认每批 25,000 行、
DuckDB 内存上限 512 MiB、2 个线程，2 GiB 服务器可调低到 384 MiB。

实测 OKX 正式公开接口返回的 `allfutures-candlesticks-2023-07-01.zip` 中，第一
个合约的每根分钟线连续重复 1,550 次；整个文件 141,373,440 个原始数据行折叠后
为 83,520 行。高重复率本身只记录告警，不作为拒绝条件。旧 FUTURES 文件还可能
出现与源日期明显不相容的交割日期，标准加密货币交割合约距离源日期超过 730 天
或已经到期时直接隔离，`XPERP` 不使用该规则。

逐笔和 L2 不执行全市场 compaction。解析器按
`dataset + UTC date + instrument_id` 输出临时 fragment；同一输出键的 fragment
完成后，只合并为该合约当日的一个 Parquet，再原子提交。上游文件若跨 UTC 日，
分别写入对应 UTC 日期目录。某个合约迟到、失败或被交易所修订时，只重建该合约
当日文件，不影响同一天的其他合约。

排序与 Row Group 规则：

| 数据集 | 文件内排序 |
|---|---|
| K 线、资金费率、OI、taker、多空比 | `timestamp/funding_time, instrument_type, instrument_id/base_currency` |
| 逐笔成交 | `timestamp, instrument_type, instrument_id, trade_id` |
| L2 | `timestamp, instrument_type, instrument_id, source_row_no, side, level_no` |
| 私有订单、成交、账单 | 业务时间、`instrument_id`、业务主键 |

- Row Group 目标未压缩大小为 128–256 MiB；
- 优先在时间桶边界结束 Row Group，便于一次扫描某个全市场时间窗口；
- 为 `timestamp`、`instrument_id`、`instrument_type` 写入统计信息；
- 使用 Parquet footer 和 Row Group pruning，避免读取不相关的时间段；
- 多日分析使用日期目录 glob；
- 普通逐笔和 L2 的全币种分析使用日期目录下的
  `*-public_trades-YYYY-MM-DD.parquet` 或
  `*-order_book_l2-YYYY-MM-DD.parquet` glob，一次扫描当日全部合约文件；
- 其他数据集的单日横截面只需要读取一个文件。

### 3.4 大文件策略

这种结构对 K 线、资金费率、OI、taker、多空比、标记价、指数价和私有数据没有
明显问题。普通逐笔和 L2 明确不合并为全市场单文件，以避免全日 compaction、
整日重建以及单文件损坏影响全部合约。

所有文件仍必须设置：

- `min_free_disk_ratio = 20%`；
- 生成前按 Raw 大小估算峰值空间，不足时不启动 compaction；
- 正式文件生成期间旧版本继续可读；
- 单个合约日文件超过配置的 `max_instrument_daily_file_size_gb` 时停止并告警，
  不在未评审的情况下继续拆成多个 part；
- catalog 保留 revision，失败时继续使用上一份已验证版本。

逐笔和 L2 的一个币种文件仍可能很大，尤其是 BTC 的 L2。若单合约日文件实际超过
文件系统、备份或查询可接受上限，再单独评审是否允许文件名增加 part 编号；当前
设计不自动拆分。

## 4. 数据集、字段与主键

### 4.1 普通 1 分钟 K 线 `candles`

来源：OKX 历史数据模块 `2`。

| 字段 | 类型 | 含义 |
|---|---|---|
| `venue` | Utf8 | 固定 `OKX` |
| `instrument_id` | Utf8 | 合约 ID |
| `instrument_type` | Utf8 | `SWAP` / `FUTURES` |
| `bar_type` | Utf8 | 固定 `1m` |
| `timestamp` | Datetime(ms, UTC) | K 线开始时间，来自 `open_time` |
| `open` | Float64 | 开盘价 |
| `high` | Float64 | 最高价 |
| `low` | Float64 | 最低价 |
| `close` | Float64 | 收盘价 |
| `volume` | Float64 | 合约张数，来自 `vol` |
| `volume_ccy` | Float64 | 标的币成交量，来自 `vol_ccy` |
| `volume_quote` | Float64 | 计价币成交量，来自 `vol_quote` |
| `confirm` | Int8 | `1` 已完成；离线文件中应为 `1` |

主键：`venue, instrument_id, bar_type, timestamp`。

### 4.2 普通逐笔成交 `public_trades`

来源：OKX 历史数据模块 `1`。

| 字段 | 类型 | 含义 |
|---|---|---|
| `venue` | Utf8 | 固定 `OKX` |
| `instrument_id` | Utf8 | 来自 `instrument_name` |
| `instrument_type` | Utf8 | `SWAP` / `FUTURES` |
| `trade_id` | Utf8 | 交易所成交 ID |
| `timestamp` | Datetime(ms, UTC) | 来自 `created_time` |
| `side` | Utf8 | taker 方向 `buy` / `sell` |
| `size` | Float64 | 合约张数 |
| `price` | Float64 | 成交价 |
| `source` | Utf8 | `0` 普通委托，`1` RPI/价格优化来源 |

主键：`venue, instrument_id, trade_id`。

旧文件可能同时包含中文和英文字段名；解析器以英文名为准，中文列忽略并在
manifest 记录 `legacy_headers=true`。

### 4.3 实际结算资金费率 `funding_settlement`

来源：OKX 历史数据模块 `3`，仅适用于 `SWAP`。

| 字段 | 类型 | 含义 |
|---|---|---|
| `venue` | Utf8 | 固定 `OKX` |
| `instrument_id` | Utf8 | 来自 `instrument_name` |
| `instrument_type` | Utf8 | 固定 `SWAP` |
| `funding_time` | Datetime(ms, UTC) | 实际结算时间 |
| `funding_rate` | Float64 | 已结算的实际资金费率 |

主键：`venue, instrument_id, funding_time`。

不得写入 `funding_snapshot` 或现有 `funding_history`。以后查询时，
`okx_historical_file` 可作为已成熟日期的首选来源。

### 4.4 历史 L2 订单簿 `order_book_l2`

来源：优先使用模块 `4`（400 档）。模块 `5`（5000 档）单独保存，不混入 400 档
数据；模块 `6`（50 档）已计划废弃，不作为主数据源。

Raw 文件保留 `instId, action, asks, bids, ts` 原始结构。Normalized 层把数组展开为
长表：

| 字段 | 类型 | 含义 |
|---|---|---|
| `venue` | Utf8 | 固定 `OKX` |
| `instrument_id` | Utf8 | 合约 ID |
| `instrument_type` | Utf8 | `SWAP` / `FUTURES` |
| `depth` | Int32 | `400` 或 `5000` |
| `timestamp` | Datetime(ms, UTC) | 推送时间 `ts` |
| `source_row_no` | Int64 | 原文件中的消息顺序 |
| `event_id` | Utf8 | `sha256(source_sha256 + source_row_no)` |
| `action` | Utf8 | `snapshot` / `update` |
| `side` | Utf8 | `ask` / `bid` |
| `level_no` | Int32 | 当前消息数组中的序号 |
| `price` | Float64 | 价位 |
| `size` | Float64 | 数量；update 中为 0 表示删除该价位 |
| `order_count` | Int64 | 该价位订单数 |
| `sequence_id` | Int64 nullable | 上游文件提供时保留 |
| `previous_sequence_id` | Int64 nullable | 上游文件提供时保留 |
| `checksum` | Int64 nullable | 旧文件提供时保留；不作为 2026-06-23 后连续性依据 |

主键：`venue, instrument_id, depth, event_id, side, level_no`。

同一毫秒内消息必须按 `source_row_no` 回放。`snapshot` 先重置订单簿，`update`
再逐价位修改。由于官方离线字段说明未承诺所有时期都有 sequence ID，Raw 文件是
最终审计与重放依据。

### 4.5 标记价格 1 分钟 K 线 `mark_price_candles`

来源：`GET /api/v5/market/history-mark-price-candles`。

字段：

`venue Utf8, instrument_id Utf8, instrument_type Utf8, bar_type Utf8,`
`timestamp Datetime(ms,UTC), open Float64, high Float64, low Float64,`
`close Float64, confirm Int8`

主键：`venue, instrument_id, bar_type, timestamp`。

### 4.6 指数价格 1 分钟 K 线 `index_price_candles`

来源：`GET /api/v5/market/history-index-candles`。

字段：

`venue Utf8, index_id Utf8, bar_type Utf8, timestamp Datetime(ms,UTC),`
`open Float64, high Float64, low Float64, close Float64, confirm Int8`

主键：`venue, index_id, bar_type, timestamp`。

多个合约可引用同一个 `index_id`，因此指数数据只下载一次，不按合约重复保存。

### 4.7 聚合 OI 与成交量 `aggregate_open_interest`

来源：`GET /api/v5/rubik/stat/contracts/open-interest-volume`。

该接口是币种级 `SWAP + FUTURES` 聚合，不是单合约 OI；不得伪装成某个
`instrument_id` 的数据。

字段：

`venue Utf8, metric_scope Utf8, base_currency Utf8, bar_type Utf8,`
`timestamp Datetime(ms,UTC), open_interest_usd Float64, volume_usd Float64`

其中 `metric_scope` 固定为 `currency_all_contracts`；`bar_type` 同时保存
`5m`、`1H`、`1D` 三种 OKX 原生周期，不用本地重采样结果冒充交易所原始周期。

主键：`venue, base_currency, bar_type, timestamp`。

### 4.8 合约 taker volume `taker_volume`

来源：`GET /api/v5/rubik/stat/taker-volume-contract`。

字段：

`venue Utf8, instrument_id Utf8, instrument_type Utf8, bar_type Utf8,`
`timestamp Datetime(ms,UTC), unit Utf8, sell_volume Float64,`
`buy_volume Float64, net_buy_volume Float64`

基础粒度固定 `5m`；`unit` 固定 `contracts`；
`net_buy_volume = buy_volume - sell_volume`。

主键：`venue, instrument_id, bar_type, timestamp`。

### 4.9 三类多空比 `long_short_ratio`

来源：

- `long-short-account-ratio-contract`；
- `long-short-account-ratio-contract-top-trader`；
- `long-short-position-ratio-contract-top-trader`。

字段：

`venue Utf8, instrument_id Utf8, instrument_type Utf8, bar_type Utf8,`
`timestamp Datetime(ms,UTC), ratio_type Utf8, long_short_ratio Float64`

`ratio_type` 枚举：

- `all_account`：全市场多头账户数 / 空头账户数；
- `top_trader_account`：前 5% 大户净多账户数 / 净空账户数；
- `top_trader_position`：前 5% 大户多头持仓价值 / 空头持仓价值。

基础粒度固定 `5m`。

主键：`venue, instrument_id, ratio_type, bar_type, timestamp`。

### 4.10 自己的最终订单 `private/final_orders`

来源：`GET /api/v5/trade/orders-history-archive`，并用 pending order watchlist
补偿长时间挂单。

字段：

```text
venue, account_alias,
instrument_type, instrument_id,
order_id, client_order_id, algo_id, algo_client_order_id,
attached_algo_client_order_id, tag,
trade_mode, margin_currency, order_type, side, position_side,
quantity, order_price, price_type, price_usd, implied_volatility,
target_currency, leverage, reduce_only, quick_margin_type,
self_trade_prevention_mode, category, source, trade_quote_currency,
state, accumulated_fill_size, average_fill_price,
last_fill_price, last_fill_size, last_trade_id, last_fill_time,
pnl, fee, fee_currency, rebate, rebate_currency,
cancel_source, cancel_source_reason,
created_at, updated_at, finalized_at,
is_tp_limit, attached_algo_orders_json, linked_algo_order_json, raw_json
```

类型规则：

- 所有 ID、枚举和 JSON 为 `Utf8`；
- `reduce_only`、`is_tp_limit` 为 `Boolean`；
- 时间为 `Datetime(ms, UTC)`；
- 价格、数量、PNL、费用、返佣为 `Decimal128(38,18)`；
- `finalized_at` 使用最终状态记录的 `uTime`。

主键：`venue, account_alias, order_id`。

只保存最终状态 `filled`、`canceled`、`mmp_canceled`。接口的 `begin/end` 按
`cTime` 过滤，不能直接用它表示“昨日完成”；每日任务必须用重叠窗口取数后按
`uTime` 归档。

### 4.11 自己的成交 `private/fills`

来源：`GET /api/v5/trade/fills-history`。

字段：

```text
venue, account_alias,
instrument_type, instrument_id,
trade_id, order_id, client_order_id, bill_id,
sub_type, tag, side, position_side, execution_type,
fill_price, fill_size, fill_index_price, fill_pnl,
fill_price_volatility, fill_price_usd, fill_mark_volatility,
fill_forward_price, fill_mark_price,
fee, fee_currency, fee_rate, trade_quote_currency,
fill_time, generated_at, raw_json
```

所有金额字段使用 `Decimal128(38,18)`；时间使用 `Datetime(ms, UTC)`。

主键：`venue, account_alias, instrument_id, trade_id, fill_time`。

### 4.12 自己的账单 `private/bills`

来源：`GET /api/v5/account/bills-archive`。

字段：

```text
venue, account_alias,
instrument_type, instrument_id,
bill_id, bill_type, bill_sub_type, timestamp,
currency, balance_change, position_balance_change,
balance, position_balance, size, price, pnl, fee,
auto_earn_amount, auto_earn_apr, interest,
margin_mode, order_id, client_order_id, trade_id,
execution_type, transfer_from, transfer_to, notes, tag,
fill_time, fill_index_price, fill_mark_price,
fill_price_volatility, fill_price_usd,
fill_mark_volatility, fill_forward_price, raw_json
```

所有金额字段使用 `Decimal128(38,18)`；时间使用 `Datetime(ms, UTC)`。

主键：`venue, account_alias, bill_id`。

## 5. 全合约清单

“全部合约”确定为加密货币类 `SWAP + FUTURES`：

- 包含当前 live 合约以及历史已到期、已下架合约，避免幸存者偏差；
- `instCategory=1` 时直接识别为 Crypto；
- 老历史缺少 `instCategory` 时，使用 OKX currency master、合约元数据和历史文件
  inventory 交叉识别，不能因当前 instruments 接口不再返回而漏掉已到期合约；
- 排除 `OPTION`、`EVENTS`、Stocks、Commodities、Forex、Bonds 等非加密资产；
- `ruleType=pre_market` 或状态尚未进入连续交易阶段的数据不纳入；若之后转为正常
  live 加密货币合约，只从正常连续交易阶段开始纳入。

每日在下载任务之前：

1. 刷新 OKX `public/instruments` 的 SWAP 和 FUTURES 列表及 currency master；
2. 使用现有 `InstrumentRepository` 保存当天完整快照并更新 lifecycle；
3. 对 T-N 日取当日有效合约，而不是只取今天仍然 live 的合约；
4. 已启用的历史文件模块在 daily 模式优先用 `ANY` 查询；
5. 第二阶段若启用 L2，由于 L2 不支持 `ANY`，按当日有效
   `instrument_family` 分批查询；
6. 已到期 FUTURES 在最后交易日数据完成前不得从任务清单移除。

如果后续要包含期权，应另立方案。期权数量、L2 文件量、字段和分区规模均明显不同，
不能直接打开一个 `OPTION` 开关。

第一阶段配置必须显式包含：

```toml
[datasets.public_trades]
enabled = false

[datasets.order_book_l2]
enabled = false
```

禁用项不创建下载任务、不请求文件、不占用 coverage 缺口告警，也不计入本次运行的
完成率。catalog 仍登记为 `disabled_by_config`，防止“未配置”和“故意暂缓”混淆。

## 6. 每日调度

使用 systemd timer 驱动一个幂等 orchestrator。定时任务只负责日常增量和自动补漏；
首次历史全量回补单独执行，不放进 timer，避免一次回补占满下一次日常任务的窗口。

建议主任务：每天 `02:30 UTC` 执行；失败文件在 `08:30`、`14:30`、`20:30 UTC`
重试。任务逻辑不依赖固定发布日期：

```text
刷新合约清单
  -> 计算各 dataset 的 mature_end
  -> 对比 offline/catalog.sqlite 覆盖区间
  -> 查询缺失文件/缺失 API 日
  -> 下载 raw
  -> 校验
  -> 转换 normalized Parquet
  -> 原子提交
  -> 记录 coverage 和 manifest
  -> 输出缺口与告警
```

成熟边界：

| 数据 | 正常 mature_end |
|---|---|
| 逐笔、普通 K 线、资金费率文件 | T-2 |
| L2 文件 | T-3 |
| 标记价、指数价、Rubik 指标 | T-1 |
| 私有最终订单、成交、账单 | T-1，额外回看最近 3 天 |

表中的逐笔和 L2 规则仅在第二阶段启用后生效；第一阶段不会为它们创建任务。

每天不是只下载一个日期，而是从每个已启用 dataset 的 coverage cursor 扫到
`mature_end`。因此停机若干天后会自动补齐。第一阶段的普通逐笔和 L2 不参与扫描；
以后启用时再从单独确认的起点开始补齐。

私有订单额外维护本地 `open_order_watchlist`：

1. 每天读取当前 pending orders；
2. 最近订单用 3 天重叠窗口重复抓取并按主键 upsert；
3. watchlist 中消失的订单按 `ordId` 查询最终状态；
4. 仅在确认终态后写 `final_orders`。

这用于避免只按订单创建时间抓取时遗漏“很早创建、昨天才成交或撤销”的订单。

### 6.1 计划新增的程序入口

实现阶段新增两个命令：

```text
trend-trader-offline-sync
  plan       # 只计算本次应下载的 dataset、日期和合约，不写数据
  run        # 执行下载、校验、规范化和 catalog 提交
  backfill   # 显式历史回补，不由 timer 调用

trend-trader-offline-notify
  check      # 检查 Bark 配置并发送测试消息
  finish     # 读取 run report 和 systemd 退出状态，发送最终结果
```

`run` 不接收固定的“昨天”参数，而是根据当前 UTC 时间、各数据集
`mature_end` 和 catalog coverage 自动生成任务。四个定时点调用同一个命令：

- `02:30 UTC` 是主运行；
- `08:30/14:30/20:30 UTC` 是幂等补漏；
- 已完成文件只校验 catalog 后跳过；
- 同一任务使用 catalog 全局租约防止手工命令和 timer 并发；
- 超过 5 小时 30 分钟仍未结束，由 systemd 终止，给下一补漏窗口留出时间；
- 每次运行生成唯一 `run_id`，并原子写入
  `/data/market/v1/offline/manifests/runs/<run_id>.json`。

Run report 至少保存：

```text
run_id, trigger_time, started_at, finished_at, duration_seconds,
status, exit_code, host, code_version,
datasets_planned, datasets_completed, datasets_partial, datasets_failed,
datasets_disabled,
raw_file_count, normalized_file_count, row_count, downloaded_bytes,
written_bytes, skipped_count, quarantine_count,
coverage_before, coverage_after, remaining_gaps,
disk_total_bytes, disk_free_bytes, error_summary
```

状态和进程退出码：

| 状态 | 退出码 | 含义 |
|---|---:|---|
| `success` | 0 | 本次计划全部完成 |
| `no_change` | 0 | 没有成熟缺口，正常空跑 |
| `partial` | 2 | 有成功提交，但仍有下载、校验或上游未就绪项 |
| `failed` | 1 | 初始化、catalog、磁盘或主流程失败 |
| `notify_failed` | 3 | 数据任务已结束，但 Bark 最终通知发送失败 |

`partial` 和 `failed` 不回滚已经原子提交的数据；下一次运行只继续未完成项。

### 6.2 systemd service

实现后提供 `deploy/systemd/trend-trader-offline-sync.service`：

```ini
[Unit]
Description=Trend Trader OKX offline data synchronization
Wants=network-online.target
After=network-online.target
RequiresMountsFor=/data/market/v1/offline

[Service]
Type=oneshot
User=trader
Group=trader
WorkingDirectory=/opt/trend-trader
EnvironmentFile=/etc/trend-trader/offline-sync.env
Environment=PYTHONDONTWRITEBYTECODE=1
ExecStart=/opt/trend-trader/.venv/bin/trend-trader-offline-sync run --config /etc/trend-trader/offline-sync.toml
ExecStopPost=/opt/trend-trader/.venv/bin/trend-trader-offline-notify finish --run-dir /data/market/v1/offline/manifests/runs --service-result=${SERVICE_RESULT} --exit-code=${EXIT_CODE} --exit-status=${EXIT_STATUS}
TimeoutStartSec=5h30m
UMask=0077
Nice=10
IOSchedulingClass=best-effort
IOSchedulingPriority=6
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/data/market/v1/offline
```

最终通知放在 `ExecStopPost`，而不是只写在主程序末尾。这样主程序正常结束、抛异常、
超时或被信号终止时，systemd 都会单独启动通知命令，并传入
`SERVICE_RESULT/EXIT_CODE/EXIT_STATUS`。若主程序来不及生成 run report，通知器使用
systemd 状态生成一条最小失败通知。云服务器断电时无法即时通知；下次启动会把没有
`finished_at` 的旧 run 标记为 `interrupted` 并补发通知。

### 6.3 systemd timer

实现后提供 `deploy/systemd/trend-trader-offline-sync.timer`：

```ini
[Unit]
Description=Schedule Trend Trader OKX offline data synchronization

[Timer]
OnCalendar=*-*-* 02:30:00 UTC
OnCalendar=*-*-* 08:30:00 UTC
OnCalendar=*-*-* 14:30:00 UTC
OnCalendar=*-*-* 20:30:00 UTC
Persistent=true
AccuracySec=1min
RandomizedDelaySec=5min
Unit=trend-trader-offline-sync.service

[Install]
WantedBy=timers.target
```

重复的 `OnCalendar` 会触发同一个 service。`Persistent=true` 使服务器在计划时间
关机、重启后补触发一次；5 分钟随机延迟用于避免固定整点访问上游。应用层租约仍是
最终的防重入保障。

部署时执行以下步骤，但在方案确认前不执行：

```bash
sudo install -m 0644 deploy/systemd/trend-trader-offline-sync.service /etc/systemd/system/
sudo install -m 0644 deploy/systemd/trend-trader-offline-sync.timer /etc/systemd/system/
sudo install -d -o trader -g trader -m 0700 /etc/trend-trader
sudo install -o trader -g trader -m 0600 deploy/config/offline-sync.env /etc/trend-trader/offline-sync.env
sudo install -o trader -g trader -m 0600 deploy/config/offline-sync.toml /etc/trend-trader/offline-sync.toml
sudo systemd-analyze verify /etc/systemd/system/trend-trader-offline-sync.service
sudo systemd-analyze verify /etc/systemd/system/trend-trader-offline-sync.timer
sudo systemctl daemon-reload
sudo systemctl enable --now trend-trader-offline-sync.timer
```

启用前必须先运行：

```bash
sudo -u trader /opt/trend-trader/.venv/bin/trend-trader-offline-sync plan \
  --config /etc/trend-trader/offline-sync.toml
sudo -u trader /opt/trend-trader/.venv/bin/trend-trader-offline-notify check \
  --env-file /etc/trend-trader/offline-sync.env
sudo systemctl start trend-trader-offline-sync.service
systemctl list-timers trend-trader-offline-sync.timer
journalctl -u trend-trader-offline-sync.service
```

systemd timer 的多个 `OnCalendar`、`Persistent` 和随机延迟行为参见
[systemd.timer 文档](https://www.man7.org/linux/man-pages/man5/systemd.timer.5.html)。

### 6.4 Bark 执行结果通知

仓库已有 `BarkNotifier` 用于模拟盘/实盘，它通过 daemon 线程异步发送。离线批处理
不能直接复用其异步 `send()`：主进程退出时 daemon 线程可能尚未完成。实现时保留
现有实时通知行为，另增加同步批处理接口，使用 Bark 支持的 JSON POST。

`/etc/trend-trader/offline-sync.env`：

```dotenv
BARK_URL=https://api.day.app/<device-key>
BARK_GROUP=trend-trader-offline
BARK_NOTIFY_NO_CHANGE=true
BARK_SUCCESS_LEVEL=active
BARK_FAILURE_LEVEL=timeSensitive
```

- 文件权限必须是 `0600`，属主 `trader:trader`；
- `BARK_URL` 包含设备 key，日志、run report、异常消息中必须脱敏；
- HTTP 超时 10 秒，按 1、3、9 秒间隔最多重试 3 次；
- 校验 HTTP 2xx；响应中存在 `code` 时还必须为成功值；
- Bark 失败不回滚已提交数据，但通知写入 catalog outbox，下一次任务开始前重发；
- `BARK_NOTIFY_NO_CHANGE=true` 表示每次空跑也通知，确保每次执行都有结果；若以后
  觉得四次通知过多，可改为 `false`，但失败和部分成功永远通知。

通知标题：

```text
[OKX离线][成功] 2026-07-30 02:30 UTC
[OKX离线][无变更] 2026-07-30 08:30 UTC
[OKX离线][部分失败] 2026-07-30 14:30 UTC
[OKX离线][失败] 2026-07-30 20:30 UTC
```

通知正文示例：

```text
run_id: 20260730T023000Z-a13f
耗时: 18m 42s
完成: all enabled datasets
暂缓: public_trades, order_book_l2
新增: raw 23, parquet 9, 1.8 GB
隔离: 0 files
剩余缺口: 0
磁盘: 3.1 TB free (42%)
```

成功通知使用 `level=active`；`partial/failed/interrupted/notify_failed` 使用
`level=timeSensitive`；统一使用 `group=trend-trader-offline`。正文只放汇总和前几项
错误，完整结果保存在 run report 与 journald。Bark 的 POST JSON、`group` 和
`level` 参数参见 [Bark 官方文档](https://github.com/Finb/Bark/blob/master/docs/en-us/tutorial.md)。

## 7. 校验、幂等和告警

### 7.1 下载校验

- HTTP 200；
- 文件大小与 API 返回的 `sizeMB` 合理一致；
- ZIP 成员完整读到 EOF 并通过 CRC 校验；
- SHA-256 已记录；
- CSV/JSON schema 能映射到当前版本；
- 文件覆盖日期与 manifest 一致；
- 规范化时间都落在预期源文件区间。

### 7.2 数据校验

- K 线：主键唯一、OHLC 合法、`confirm=1`；完全相同的连续原始行在 CSV 解析前
  折叠，同主键不同内容则隔离；
- 标准 FUTURES：交割日期必须与源日期相容；`XPERP` 单独识别；
- 逐笔：`trade_id` 唯一、价格和数量为正；
- 资金费率：`funding_time` 唯一，允许结算周期变化；
- L2：首个可回放事件必须是 snapshot；有 sequence ID 时检查连续性；
- OI/taker/ratio：时间对齐 5 分钟，主键唯一；
- 私有数据：订单、成交、账单三表按 `order_id/trade_id/bill_id` 交叉核对。

### 7.3 空数据语义

未上市、已到期或当天无成交不应直接视为失败。catalog 必须区分：

- `complete_with_rows`；
- `complete_empty`；
- `not_listed`；
- `upstream_not_ready`；
- `failed_validation`。

### 7.4 告警

- 成熟边界以前仍缺文件；
- 文件 checksum 发生变化；
- schema 出现新列或缺少必需列；
- 任一合约连续两天下载失败；
- L2 无可用 snapshot 或 sequence 断裂；
- 磁盘剩余空间低于 20%；
- 私有 API 鉴权失败或覆盖落后超过 24 小时。

以上告警写入 run report，并纳入本次 Bark `partial` 或 `failed` 通知。Bark 本身
不可用时写入 notification outbox 和 journald，由下一次运行重试，不能因为通知失败
重复下载或回滚已经提交的数据。

## 8. 安全

- 私有 API key 仅授予 Read 权限，不授予 Trade/Withdraw；
- key 放 systemd `EnvironmentFile`，权限 `0600`，不写仓库；
- `/data/market/v1/offline/normalized/private` 权限 `0700`；
- 云盘启用静态加密；
- 日志不得记录签名 header、完整响应或账户 UID；
- `account_alias` 使用本地别名，不把真实 UID 写入路径。

## 9. 已确认的历史回补范围

下表基于 2026-07-30 的 OKX 官方页面、API 文档和只读公共 API 探测。

已确认采用“每个已启用数据集从最早可得时间分别回补”，不设置全局统一起点。

| 数据 | 官方/实测历史范围 | 最终回补规则 |
|---|---|---|
| 普通逐笔成交 | 官方离线文件从 2021-09 起 | 第一阶段禁用；第二阶段容量评估后再定，`2021-09-01` 仅为最早候选 |
| 普通 1m K 线 | 官方离线文件从 2023-07 起 | `2023-07-01` |
| 实际结算资金费率 | 官方离线文件从 2022-03 起 | `2022-03-01` |
| L2 400 档 | 官方 L2 从 2023-03 起 | 第一阶段禁用；第二阶段容量评估后再定，`2023-03-01` 仅为最早候选 |
| L2 5000 档 | API 文档明确从 2025-11-01 起 | 第一阶段禁用；以后如需要再作为独立增强集评审 |
| 标记价格 1m K 线 | 官方只承诺“recent years”；BTC-USDT-SWAP 实测 2020-01-02 有数据，之前无数据 | 每个合约从 `max(listing_time, 2020-01-02)` 尝试 |
| 指数价格 1m K 线 | 官方只承诺“recent years”；BTC-USDT 实测 2020-01-02 有数据，之前无数据 | 每个指数从 `2020-01-02` 尝试 |
| 聚合 OI 5m | 实测只保留约 2 天；2026-07-30 查询最早约 2026-07-28 05:45 UTC | 首次运行动态探测当时最早可得记录并立即固化，不硬编码日期 |
| 聚合 OI 1H | 实测约 30 天 | 已启用；首次运行动态探测最早记录并回补 |
| 聚合 OI 1D | 实测约 180 天 | 已启用；首次运行动态探测最早记录并回补 |
| 合约 taker volume 5m | 官方说明可回到 2024-02 上旬 | 从 `2024-02-01` 探测，逐合约记录真实首条 |
| 全市场账户多空比 5m | 官方说明可回到 2024-02 上旬 | 从 `2024-02-01` 探测 |
| 大户账户多空比 5m | 官方说明可回到 2024-03-22 | `2024-03-22` |
| 大户持仓多空比 5m | 官方说明可回到 2024-03-22 | `2024-03-22` |
| 自己的最终订单 | 私有 API 仅近 3 个月；网页报表可从 2021-01-28 起导出 | API 先补近 3 个月；另手工导出 `2021-01-28` 至 API 起点 |
| 自己的成交和账单 | 私有 API 仅近 3 个月；Trading history 网页报表可从 2021-01-28 起导出 | 同上 |

重要限制：

- 标记价和指数价的 2020-01-02 是 BTC 代表性实测，不是 OKX 对所有合约的统一
  SLA；其他合约的真实起点不能早于上市时间。
- 聚合 OI 的 5m 数据无法从交易所补到 2024 年。若云服务器此前没有本地保存，
  现在能拿到的就只有最近约两天。
- 普通逐笔和 L2 的官方最早可得日期只代表上游能力，不代表本项目已经决定从该日
  回补。第一阶段程序必须保持禁用，不能因“最大历史模式”自动开始下载。
- `taker volume` 和多空比的起点会随合约上市时间变化，回补器必须允许
  `not_listed`，不能把空数据当成永久失败。
- 私有 2021 年以来的网页报表生成/下载不是现有公共自动化 API；应作为一次性
  人工导出 + 本地导入步骤，不在每日任务中模拟网页操作。

### 9.1 分析层有效区间

下载范围与分析范围分开管理：

- 单数据集研究使用该数据集自己的完整历史；
- 多数据集联合分析使用
  `analysis_start = max(所选数据集实际首条时间)`；
- 每个合约还要应用
  `instrument_start = max(analysis_start, listing_time, 各特征实际首条时间)`；
- 数据缺失保存为 null，不用 0 填充；
- 每个时间点记录有效合约数和 coverage ratio，低于分析配置阈值时不计算
  全市场横截面因子；
- 分析产物记录使用的数据集、schema version、实际起止时间和 coverage 条件，
  保证结果可以复现。

因此，增加一个历史较短的新特征只会缩短使用该特征的分析区间，不会导致存储层
删除其他数据集更早的历史。

## 10. 建议采用的实施批次

### 10.1 第一阶段：先完成中小型数据

1. 批次 A：普通 K 线，从 `2023-07-01` 至 mature_end；
2. 批次 B：资金费率，从 `2022-03-01` 至 mature_end；
3. 批次 C：标记价从 `max(listing_time, 2020-01-02)` 探测，指数价从
   `2020-01-02` 探测；
4. 批次 D：taker 和全市场账户多空比从 `2024-02-01` 探测，大户账户/持仓
   多空比从 `2024-03-22` 回补；
5. 批次 E：聚合 OI 的 `5m`、`1H`、`1D` 都从首次运行时仍可取得的最早时间
   开始回补；
6. 批次 F：主账户及实际参与交易的子账户先通过私有 API 回补近 3 个月；随后导入
   2021-01-28 以来网页 CSV。

第一阶段不包含普通逐笔和 L2；timer 上线后也继续保持这两个 dataset 禁用。

### 10.2 第二阶段：逐笔与 L2 容量评估

其他数据完成后，先读取 OKX 文件元数据估算 Raw 大小；若元数据不足，再经用户确认
后下载少量代表样本。评估至少覆盖：

- 普通日与极端行情日；
- BTC、ETH 和一个中小币种；
- SWAP 与 FUTURES；
- Raw 压缩大小、解压峰值、normalized Parquet 大小；
- 单合约生成时长、临时空间和全市场日总量；
- 按 30 天、1 年和最大历史区间估算的容量及云盘成本。

容量评估只输出报告，不自动改变 `enabled`，也不启动历史下载。

### 10.3 第三阶段：单独确认后启用

只有用户再次明确确认以下三项后，才修改配置并执行：

1. 普通逐笔的回补起点；
2. L2 的档位和回补起点；
3. 磁盘容量、逐笔/L2 Raw 保留策略和空间上限。

启用顺序建议先普通逐笔、后 L2；先跑日常增量和一个小历史窗口，验收容量增长后再
扩大回补区间。

## 11. 确认状态与后续事项

第一阶段的历史回补起点已经确认：所有已启用数据集按第 9 节规则回补各自最大
可得历史，不使用统一起点。当前第一阶段方案没有未决的历史区间问题。

第一阶段的合约范围、OI 周期、私有账户范围和 Raw 保留策略也已确认：

- 仅加密货币类 `SWAP + FUTURES`，包含历史到期/下架合约，排除 pre-market 阶段；
- 聚合 OI 保存 `5m + 1H + 1D`；
- 私有数据覆盖主账户和实际交易子账户；
- 第一阶段 Raw 永久保留，不自动删除。

普通逐笔和 L2 不纳入这次起点确认，它们保持 `disabled_by_config`；磁盘容量、
这两个大数据集的 Raw 保留周期、L2 档位以及历史起点，都推迟到第二阶段容量报告
完成后另行确认。

## 12. 实现对应关系

第一阶段实现已落到以下模块：

- `src/trend_trader/data/offline/config.py`：强类型 TOML 配置及第一阶段范围保护；
- `client.py`：历史数据中心、公共 REST、私有 REST、鉴权、限速、重试和分页；
- `schemas.py`：全部已启用数据集的规范化 schema 和解析器；
- `storage.py`：Raw revision、全市场日级 Parquet、锁和原子提交；
- `catalog.py`：运行、覆盖率、可用性、合约和通知 outbox；
- `sync.py`：daily/backfill/range 计划及任务执行；
- `cli.py`、`notify.py`：命令行和同步 Bark 结果通知；
- `deploy/systemd/`：02:30、08:30、14:30、20:30 UTC timer；
- `configs/offline_sync.example.toml`：部署配置样例。

逐笔成交和 L2 的配置仍由校验器强制保持关闭。部署步骤和运维命令见
`docs/okx_offline_sync_runbook.md`。
