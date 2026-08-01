# Binance 全永续离线数据下载

下载器从 Binance 官方公共数据归档发现 USDⓈ-M（UM）和 COIN-M（CM）的当前及
历史永续合约，下载 `1h` K 线、资金费率、`1h` 标记价格、`1h` 指数价格和
AggTrades，并转换为项目统一的 Parquet 离线结构。

## 数据目录

默认根目录是当前项目的 `data/market/v1/offline`：

```text
data/market/v1/offline/
├── raw/binance/futures/{um,cm}/{monthly,daily}/...
├── normalized/
│   ├── candles/venue=BINANCE/year=YYYY/date=YYYY-MM-DD/...
│   ├── funding_rates/venue=BINANCE/year=YYYY/date=YYYY-MM-DD/...
│   ├── mark_price_candles/venue=BINANCE/year=YYYY/date=YYYY-MM-DD/...
│   ├── index_price_candles/venue=BINANCE/year=YYYY/date=YYYY-MM-DD/...
│   └── aggregate_trades/venue=BINANCE/year=YYYY/date=YYYY-MM-DD/...
├── manifests/runs/
├── quarantine/binance/
└── .staging/binance/
```

四类低容量数据每天生成一个全市场文件。AggTrades 按
`市场类型 + 完整合约 ID + UTC 日` 分文件。Raw ZIP 和官方 CHECKSUM 永久保留。

## 先估算容量

完整 AggTrades 历史很大。正式下载前先列出所有官方对象并汇总压缩包大小：

```bash
uv run trend-trader-binance-download plan
```

只估算单个市场或数据集：

```bash
uv run trend-trader-binance-download plan \
  --market um \
  --dataset candles \
  --dataset funding_rates
```

## 最高吞吐下载

默认使用 64 个并发下载连接，转换进程数等于本机 CPU 核数：

```bash
uv run trend-trader-binance-download sync
```

高带宽机器可继续调高下载并发；转换进程不建议明显超过物理 CPU 核数：

```bash
uv run trend-trader-binance-download sync \
  --download-workers 128 \
  --convert-workers 16
```

下载写入 `.part` 文件，网络中断后再次执行相同命令会续传；每个 ZIP 完成后使用
官方 SHA-256 CHECKSUM 验证。Normalized 文件先在 `.staging` 生成，通过 DuckDB
并行去重、排序和压实后再原子替换正式文件。

## 小范围验证与补偿

```bash
uv run trend-trader-binance-download sync \
  --market um \
  --symbol BTCUSDT \
  --dataset candles \
  --start 2025-01-01 \
  --end 2025-01-31
```

`--symbol` 使用 Binance 原生标识，例如 UM 的 `BTCUSDT` 或 CM 的
`BTCUSD_PERP`。可以重复传入 `--market`、`--dataset` 和 `--symbol`。

## 数据口径

- Klines、Mark Price、Index Price 固定下载 `1h`，规范字段为 `bar_type=1H`；
- Funding Rate 保留实际结算时刻和结算周期；
- AggTrades 保留交易所聚合成交粒度，不进行小时聚合；
- 所有规范时间均为 UTC，时间类型为 `Datetime(ms, UTC)`；
- `venue=BINANCE`、`market_type=UM/CM`、`instrument_type=SWAP`；
- 历史永续集合由 Funding Rate 归档、日 K 线目录和当前 `exchangeInfo` 的并集发现；
- UM 排除日期后缀交割合约，CM 只接受 `_PERP` 合约。

Binance 月归档优先用于历史区间，未被月归档覆盖的月份再使用日归档，避免重复下载
相同日期。范围参数可能仍需下载覆盖目标日期的整个月包，但只会输出请求日期范围内
的 Parquet 分区。
