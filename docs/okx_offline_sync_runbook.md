# OKX 离线下载部署与运行

当前代码只完成实现和本地测试；下面命令应在云服务器确认路径、用户和密钥后执行。
生产数据根目录固定为 `/data/market/v1/offline`，不会写入现有实时 snapshot 目录。
服务器服务账户固定为 `trader`；`trend-trader` 只用于项目目录、命令和 systemd
unit 名称，不是 Linux 用户名。

## 1. 准备配置

```bash
id trader
sudo install -d -m 0750 -o trader -g trader /opt/trend-trader
sudo install -d -m 0750 -o trader -g trader /etc/trend-trader
sudo install -d -m 0750 -o trader -g trader /data/market/v1/offline
sudo install -m 0600 -o trader -g trader \
  configs/offline_sync.example.toml /etc/trend-trader/offline-sync.toml
sudo install -m 0600 -o trader -g trader \
  deploy/systemd/offline-sync.env.example /etc/trend-trader/offline-sync.env
```

在 TOML 中为主账户和实际交易子账户分别增加 `[[private_accounts]]`。在
`offline-sync.env` 中设置对应的只读 OKX API key，以及：

```text
BARK_URL=https://api.day.app/<device-key>
```

OKX key 只授予读取权限，不授予交易或提币权限。

## 2. 安装并先做只读检查

项目假定部署在 `/opt/trend-trader`，虚拟环境位于 `/opt/trend-trader/.venv`：

```bash
sudo chown -R trader:trader /opt/trend-trader
sudo -u trader sh -c '
  cd /opt/trend-trader
  uv sync --frozen
  .venv/bin/trend-trader-offline-sync \
    --config /etc/trend-trader/offline-sync.toml plan --mode daily
'
```

`plan` 只列出任务，不请求 OKX、不下载数据。可用下列命令做一个小范围首跑：

```bash
sudo -u trader /opt/trend-trader/.venv/bin/trend-trader-offline-sync \
  --config /etc/trend-trader/offline-sync.toml \
  range --start 2026-07-27 --end 2026-07-28 --dataset candles
```

## 3. 启用定时器

```bash
sudo cp deploy/systemd/trend-trader-offline-sync.service /etc/systemd/system/
sudo cp deploy/systemd/trend-trader-offline-sync.timer /etc/systemd/system/
sudo cp deploy/systemd/trend-trader-offline-backfill.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now trend-trader-offline-sync.timer
systemctl list-timers trend-trader-offline-sync.timer
```

timer 每天按 UTC 的 `02:30 / 08:30 / 14:30 / 20:30` 运行，并随机延迟最多
5 分钟。每次运行会扫描最近 14 个已成熟日期中的缺口；K 线和资金费率按 T+2，
其他第一阶段数据按 T+1。私有数据固定重抓最近 3 天并按主键覆盖，以吸收迟到更新。

Bark 由 `ExecStopPost` 同步发送。通知失败不会改变已完成下载的状态，未发送消息留在
SQLite outbox，下一次运行会重试。

## 4. 历史回补

先分别小批验证，再启动完整回补：

```bash
# 指定数据集和区间
sudo -u trader /opt/trend-trader/.venv/bin/trend-trader-offline-sync \
  --config /etc/trend-trader/offline-sync.toml range \
  --start 2020-01-02 --end 2020-01-03 --dataset candles

# 验证 REST/批量文件切换边界及重叠一致性
sudo -u trader /opt/trend-trader/.venv/bin/trend-trader-offline-sync \
  --config /etc/trend-trader/offline-sync.toml range \
  --start 2023-06-30 --end 2023-07-01 --dataset candles

# 按每个数据集配置的最早时间回补所有缺口
sudo -u trader sh -c '
  set -a
  . /etc/trend-trader/offline-sync.env
  set +a
  exec /opt/trend-trader/.venv/bin/trend-trader-offline-sync \
    --config /etc/trend-trader/offline-sync.toml backfill
'
```

普通 1m K 线在 `2023-07-01` 前按合约调用 REST，每次最多返回 300 根，因此完整
回补的请求数远多于批量文件下载。完整 `backfill` 可能运行很久，仓库提供手工启动、
不会随开机自动运行的独立
systemd service：

```bash
sudo cp deploy/systemd/trend-trader-offline-backfill.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl stop trend-trader-offline-sync.timer
sudo systemctl start trend-trader-offline-backfill.service

systemctl status trend-trader-offline-backfill.service
journalctl -fu trend-trader-offline-backfill.service
```

停止回补：

```bash
sudo systemctl stop trend-trader-offline-backfill.service
sudo systemctl start trend-trader-offline-sync.timer
```

停止后再次 `start` 是安全的，catalog 会让它只处理未解决日期。service 使用
`trader:trader`、`/etc/trend-trader/offline-sync.toml` 和
`/etc/trend-trader/offline-sync.env`，结束后通过同一 Bark outbox 发送结果。
不要对该 service 执行 `systemctl enable`，避免服务器每次启动都自动开始全量回补。
启动回补前暂停 daily timer，避免长时间回补期间定时任务重复竞争 catalog 锁；
回补结束或停止后再启动 timer。
逐笔与 L2 即使被误写为 `enabled=true`，配置校验也会拒绝启动。

两个 service 使用 `ProtectHome=read-only`：服务不能修改 `/home`，但可以执行
`uv` 可能安装在 `/home/trader/.local/share/uv/` 下的 Python 运行时。不能改为
`ProtectHome=true`，否则 `.venv/bin/python` 指向该运行时时会导致 systemd
`status=203/EXEC`。

## 5. 查看结果和实际可得区间

```bash
sudo -u trader /opt/trend-trader/.venv/bin/trend-trader-offline-sync \
  --config /etc/trend-trader/offline-sync.toml status

sudo -u trader /opt/trend-trader/.venv/bin/trend-trader-offline-sync \
  --config /etc/trend-trader/offline-sync.toml availability

sudo -u trader /opt/trend-trader/.venv/bin/trend-trader-offline-sync \
  --config /etc/trend-trader/offline-sync.toml invalid-identifiers

journalctl -u trend-trader-offline-sync.service -n 200 --no-pager
```

`status` 给出已完整落盘的起止日期、分区数和行数。`availability` 同时显示方案中的
候选起点与下载过程中观测到的首条事件时间。最终研究区间应由实际使用的数据集求
交集，不应直接使用某个全局固定起点。

`invalid-identifiers` 显示 OKX 明确返回 `51001` 的标识、原始错误、实际失败次数和
缓存命中的跳过次数。未上市合约不在此列表中：程序按 `listTime` 逐日判断，上市前
跳过、上市日开始请求。合约元数据变化会使旧排除记录自动失效并触发重新探测。

## 6. 补偿早期缺失的指数币种

旧版本只从当前合约目录生成指数 universe，因此 2020 年分区会漏掉后来下架合约所
使用的 `EOS-USD`、`EOS-USDT`、`BSV-USD` 和 `BSV-USDT`。新版默认将它们加入
`historical_index_ids`。如果服务器 TOML 希望显式固定该配置，在顶层加入：

```toml
historical_index_ids = ["BSV-USD", "BSV-USDT", "EOS-USD", "EOS-USDT"]
discover_historical_indices = true
```

先用一天验证。即使该日期原 coverage 已经是 complete，带 `--identifier` 的 range
也会按 identifier coverage 执行，并把新增行合并到原 Parquet：

```bash
sudo -u trader /opt/trend-trader/.venv/bin/trend-trader-offline-sync \
  --config /etc/trend-trader/offline-sync.toml range \
  --start 2020-02-10 --end 2020-02-10 \
  --dataset index_price_candles \
  --identifier EOS-USD --identifier EOS-USDT \
  --identifier BSV-USD --identifier BSV-USDT
```

验证无误后直接重新启动原有 backfill service：

```bash
sudo systemctl stop trend-trader-offline-sync.timer
sudo systemctl start trend-trader-offline-backfill.service
sudo journalctl -fu trend-trader-offline-backfill.service
```

查看可续跑进度：

```bash
sudo -u trader /opt/trend-trader/.venv/bin/trend-trader-offline-sync \
  --config /etc/trend-trader/offline-sync.toml compensation-status
```

原有 `backfill` 会同时检查日期级 coverage 和历史补充指数的 identifier coverage：旧日期
即使已经是 complete，只要缺少某个补充指数，仍会生成仅包含该指数的补偿任务。
SQLite 对每个指数和日期记录 `complete` 或 `unavailable`，所以停止或异常退出后再次
启动同一个 backfill service 只处理缺口。补偿完成后再启动
`trend-trader-offline-sync.timer`。

第一次使用新版 backfill 时，会在 `index_price_candles.start` 当天对 OKX 当前指数列表
做一次历史存在性扫描。候选通常较多，前几分钟可能只看到
`discover-index-universe` 进度而尚未开始日级任务；扫描结果和完成标记写入 SQLite，
后续重启不会重复扫描。

私有 API 只能自动回补近 3 个月。更早的最终订单、成交和账单，需要从 OKX 网页生成
2021-01-28 以来的 Trading history 报表；这部分是一次性人工导出流程，不应把网页登录
凭证交给定时下载程序。

## 7. 小内存服务器与 OOM

`2023-07-01` 起的历史 K 线和资金费率使用有界内存流程：ZIP 原始文本行先以常量内存折叠连续完全
重复行，再进行 CSV 解析并按固定批次写临时 Parquet fragment；DuckDB 在固定内存
上限内完成主键去重和外部排序，PyArrow 再分批写入最终日级 Parquet。相关配置为：

```toml
stream_batch_rows = 25000
compaction_memory_mb = 512
compaction_threads = 2
```

对于 2 GiB 服务器，先使用默认值。如果仍然出现内核 OOM，可调整为：

```toml
stream_batch_rows = 10000
compaction_memory_mb = 384
compaction_threads = 2
```

批次越小，解析峰值内存越低；DuckDB 超出内存限制后会自动使用数据盘完成排序。
程序不会因为官方文件重复率高而失败，但会在 journald 和 artifact metadata 记录
`input_rows`、`emitted_rows`、`adjacent_duplicate_rows` 与 `duplicate_ratio`。
同主键不同内容、源日期越界或明显错误的标准 FUTURES 交割日期仍会隔离 Raw 并使
任务失败。

对 OKX 官方 `allfutures-candlesticks-2023-07-01.zip` 的本机只读基准：ZIP
39.64 MB、成员解压大小 12.11 GB、141,373,440 个原始行、去重后 83,520 行；
预解析去重耗时约 30 秒，峰值 RSS 约 182 MB。服务器实际耗时取决于单核性能。
临时文件位于：

```text
/data/market/v1/offline/.staging/
```

异常退出后再次运行是安全的：正式 Parquet 通过 `os.replace` 原子提交，fragment
可删除，catalog 会把上一次未结束的 run 标记为 `interrupted`。新任务会自动清理旧
staging；如需手工清理，必须先确认没有同步进程：

```bash
pgrep -af trend-trader-offline
sudo -u trader find /data/market/v1/offline/.staging \
  -mindepth 1 -depth -delete
```

## 8. 从 SQLite compactor 版本升级

先停止仍在运行的旧回补任务，再更新代码和依赖：

```bash
sudo systemctl stop trend-trader-offline-backfill.service 2>/dev/null || true
cd /opt/trend-trader
git pull
sudo -u trader sh -c 'cd /opt/trend-trader && uv sync --frozen'
```

在 `/etc/trend-trader/offline-sync.toml` 顶层删除旧的 `sqlite_cache_mb`，2 GiB
服务器建议使用：

```toml
stream_batch_rows = 10000
compaction_memory_mb = 384
compaction_threads = 2
```

先重跑一个 UTC 日。CLI 会即时输出任务开始、Raw 下载完成、每解析 25 万行和最终
提交状态，不再等整日转换完成后才显示输出：

```bash
sudo -u trader /usr/bin/time -v \
  /opt/trend-trader/.venv/bin/trend-trader-offline-sync \
  --config /etc/trend-trader/offline-sync.toml \
  range --start 2023-07-01 --end 2023-07-01 --dataset candles
```
