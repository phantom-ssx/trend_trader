# trend_trader

基于 `uv` 和 `nautilus-trader` 的最小 OKX 合约策略项目。

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
uv run nt-okx-download \
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
uv run nt-okx-download \
  --source okx-rest \
  --inst-id ETH-USDT-SWAP \
  --bar 1m \
  --start 2026-01-01T00:00:00Z \
  --end 2026-07-07T12:23:02Z
```

也可以使用 `ccxt` 统一接口：

```bash
uv run nt-okx-download \
  --source ccxt \
  --inst-id ETH-USDT-SWAP \
  --bar 1m \
  --start 2026-07-06T00:00:00Z \
  --end 2026-07-07T00:00:00Z
```

当前实测 OKX 专用 REST 更快；`ccxt` 更适合未来扩展其它交易所或做通用 fallback。

输出字段：

`ts, open, high, low, close, volume, volume_ccy, volume_quote, confirm, exchange, inst_id, bar`

## 回测

```bash
uv run nt-okx-backtest --config configs/backtest.example.toml
```

对比 MA 交叉过滤策略的整体或月度表现：

```bash
uv run python scripts/evaluate_filters.py \
  --config configs/backtest.eth-2026.toml \
  --mode monthly \
  --format table
```

用 NautilusTrader 执行当前效果最好的 `spread_0.35%+ATR` 小时级全仓策略：

```bash
uv run nt-okx-backtest \
  --config configs/backtest.eth-2026.toml \
  --resample 1h \
  --strategy best-filter \
  --sizing all-in
```

默认会跳过名义金额低于 `50 USDT` 的小额调仓订单；可用
`--min-order-notional 0` 关闭过滤，或传其它数值调整阈值。

如需查看 Nautilus 回测的订单明细，可导出 CSV：

```bash
uv run nt-okx-backtest \
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
uv run nt-okx-chart \
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
uv run nt-okx-chart \
  --parquet data/clean/okx/ETH-USDT-SWAP/ETH-USDT-SWAP_1m_20260101T000000Z_20260707T124753Z.parquet \
  --start 2026-01-01T00:00:00Z \
  --end 2026-07-07T12:47:53Z \
  --resample 1h \
  --title "ETH-USDT-SWAP 1h 2026 YTD" \
  --out outputs/eth_usdt_swap_1h_2026_ytd_chart.html
```

## 模拟盘 / 实盘

先复制环境变量：

```bash
cp .env.example .env
```

模拟盘：

```bash
uv run nt-okx-paper --config configs/paper.example.toml
```

默认只做 dry-run：读取配置、检查密钥、构造 Nautilus 的 `OKXDataClientConfig` 和
`OKXExecClientConfig`。确认无误后再启动真实 Nautilus node：

```bash
uv run nt-okx-paper --config configs/paper.example.toml --start
```

实盘入口默认要求显式确认：

```bash
uv run nt-okx-live --config configs/live.example.toml --i-understand-this-is-live
```

实盘同样默认 dry-run。确认配置无误后才加 `--start`：

```bash
uv run nt-okx-live \
  --config configs/live.example.toml \
  --i-understand-this-is-live \
  --start
```

当前项目已按 `nautilus-trader==1.230.0` 验证 OKX adapter 类名：
`OKXDataClientConfig`、`OKXExecClientConfig`、`OKXLiveDataClientFactory`、
`OKXLiveExecClientFactory`。
