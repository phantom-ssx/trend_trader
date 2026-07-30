# OKX 离线下载部署与运行

当前代码只完成实现和本地测试；下面命令应在云服务器确认路径、用户和密钥后执行。
生产数据根目录固定为 `/data/market/v1/offline`，不会写入现有实时 snapshot 目录。

## 1. 准备配置

```bash
sudo install -d -m 0750 -o trend-trader -g trend-trader /etc/trend-trader
sudo install -d -m 0750 -o trend-trader -g trend-trader /data/market/v1/offline
sudo cp configs/offline_sync.example.toml /etc/trend-trader/offline-sync.toml
sudo cp deploy/systemd/offline-sync.env.example /etc/trend-trader/offline-sync.env
sudo chmod 0600 /etc/trend-trader/offline-sync.env
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
cd /opt/trend-trader
uv sync --frozen
set -a
source /etc/trend-trader/offline-sync.env
set +a
.venv/bin/trend-trader-offline-sync \
  --config /etc/trend-trader/offline-sync.toml plan --mode daily
```

`plan` 只列出任务，不请求 OKX、不下载数据。可用下列命令做一个小范围首跑：

```bash
.venv/bin/trend-trader-offline-sync \
  --config /etc/trend-trader/offline-sync.toml range \
  --start 2026-07-27 --end 2026-07-28 --dataset candles
```

## 3. 启用定时器

```bash
sudo cp deploy/systemd/trend-trader-offline-sync.service /etc/systemd/system/
sudo cp deploy/systemd/trend-trader-offline-sync.timer /etc/systemd/system/
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
.venv/bin/trend-trader-offline-sync \
  --config /etc/trend-trader/offline-sync.toml range \
  --start 2023-07-01 --end 2023-07-31 --dataset candles

# 按每个数据集配置的最早时间回补所有缺口
.venv/bin/trend-trader-offline-sync \
  --config /etc/trend-trader/offline-sync.toml backfill
```

完整 `backfill` 可能运行很久，建议在独立 systemd service 或 tmux 中执行。文件和
catalog 都是幂等的；重新运行只处理未解决日期。逐笔与 L2 即使被误写为
`enabled=true`，配置校验也会拒绝启动。

## 5. 查看结果和实际可得区间

```bash
.venv/bin/trend-trader-offline-sync \
  --config /etc/trend-trader/offline-sync.toml status

.venv/bin/trend-trader-offline-sync \
  --config /etc/trend-trader/offline-sync.toml availability

journalctl -u trend-trader-offline-sync.service -n 200 --no-pager
```

`status` 给出已完整落盘的起止日期、分区数和行数。`availability` 同时显示方案中的
候选起点与下载过程中观测到的首条事件时间。最终研究区间应由实际使用的数据集求
交集，不应直接使用某个全局固定起点。

私有 API 只能自动回补近 3 个月。更早的最终订单、成交和账单，需要从 OKX 网页生成
2021-01-28 以来的 Trading history 报表；这部分是一次性人工导出流程，不应把网页登录
凭证交给定时下载程序。
