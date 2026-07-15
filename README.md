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
uv run nt-okx-backtest \
  --config configs/backtest.eth-2026.toml \
  --resample 1h \
  --strategy best-filter \
  --sizing all-in
```

执行带独立退出和冷却规则的小时级 MA5/20 策略：

```bash
uv run nt-okx-backtest \
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
uv run nt-okx-paper --config configs/paper.example.toml
```

默认只做 dry-run：读取配置、检查密钥、构造 Nautilus 的 `OKXDataClientConfig` 和
`OKXExecClientConfig`。确认无误后再启动真实 Nautilus node：

```bash
uv run nt-okx-paper --config configs/paper.example.toml --start
```

启动后，模拟盘策略会先请求 `warmup_bars` 根历史 K 线初始化 MA/ATR。在历史数据
达到指标所需的最小数量之前，策略仍接收实时行情，但不会提交订单；预热失败时会
保持禁用下单并输出错误日志。

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
