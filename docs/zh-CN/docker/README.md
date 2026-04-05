# TrajectoryRL Docker 部署指南

使用 Docker 是运行 TrajectoryRL 验证者和矿工的**推荐方式**。一个 All-in-One（多合一）的镜像中包含了所需的一切组件：验证者、mock-tools 服务器以及 OpenClaw 网关。Watchtower 会负责自动从 GHCR 拉取并更新镜像。

## 快速开始 — 验证者版

```bash
# 1. 克隆代码仓库
git clone https://github.com/trajectoryRL/trajectoryRL.git
cd trajectoryRL

# 2. 配置文件
cp .env.example .env.validator
# 编辑 .env.validator，设置 WALLET_NAME（钱包名）、WALLET_HOTKEY（热密钥）和 CLAWBENCH_LLM_API_KEY 等参数

# 3. 启动
docker compose -f docker/docker-compose.validator.yml --env-file .env.validator up -d

# 4. 查看日志
docker compose -f docker/docker-compose.validator.yml logs -f validator
```

## 配置项

在仓库的根目录下创建 `.env.validator`：

```bash
# 必填项
WALLET_NAME=validator
WALLET_HOTKEY=default
NETUID=11
NETWORK=finney
CLAWBENCH_LLM_API_KEY=...
CLAWBENCH_LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
CLAWBENCH_DEFAULT_MODEL=zhipu/glm-5

# 选填项
LOG_LEVEL=INFO                    # 若处于开发环境请使用 DEBUG
EVAL_INTERVAL_BLOCKS=7200         # ~24小时 (如果是开发环境可设为100)
WEIGHT_INTERVAL_BLOCKS=360        # ~72分钟
EMA_ALPHA=0.3
INACTIVITY_BLOCKS=14400           # ~48小时
```

### 钱包设置

钱包要求位于 `~/.bittensor/wallets/`（并且是以只读挂载的方式）：

```
~/.bittensor/wallets/{WALLET_NAME}/
  ├── coldkey
  ├── coldkeypub.txt
  └── hotkeys/
      └── {WALLET_HOTKEY}
```

## 快速开始 — 矿工版

```bash
# 配置文件
cp .env.miner.example .env.miner
# 编辑文件并设置：WALLET_NAME、WALLET_HOTKEY 和 ANTHROPIC_API_KEY 等参数

# 运行终端 CLI 命令行
docker compose -f docker/docker-compose.miner.yml --env-file .env.miner run miner build
docker compose -f docker/docker-compose.miner.yml --env-file .env.miner run miner validate
docker compose -f docker/docker-compose.miner.yml --env-file .env.miner run miner submit
```

## 运维操作指令

```bash
# 启动
docker compose -f docker/docker-compose.validator.yml --env-file .env.validator up -d

# 查看日志
docker compose -f docker/docker-compose.validator.yml logs -f validator
docker compose -f docker/docker-compose.validator.yml logs --tail=100 validator

# 停止
docker compose -f docker/docker-compose.validator.yml down

# 强制更新（注：如果你用的是 Watchtower，这部分是会自动处理的）
docker compose -f docker/docker-compose.validator.yml pull validator
docker compose -f docker/docker-compose.validator.yml up -d validator
```

## 架构

All-in-One（多合一）验证器镜像由一个 bash 入口脚本（entrypoint）统领并运行着三个主要进程：

```
┌─────────────────────────────────────────────┐
│  trajectoryrl-validator 容器              │
│                                             │
│  1. mock-tools   (端口 3001)  ─ 后台常驻    │
│  2. OpenClaw     (端口 18789) ─ 后台常驻    │
│  3. validator    (进程号 1)   ─ 前台常驻    │
│                                             │
│  共享卷区: /workspace (内含 AGENTS.md, 测试数据集) │
└─────────────────────────────────────────────┘
```

- **mock-tools**：这是一个确定性工具反馈服务器（用于模拟如邮件、日历、Slack、任务列表这类环境的反馈数据）
- **OpenClaw gateway**：带有 clawbench-tools 插件的 LLM Agent 编排网关
- **validator**：验证者核心主程，用于读取链上承诺、获取并解析打包文件、使用 ClawBench 数据集跑分析测试并设定权重。

Watchtower 将以每 5 分钟轮询一次 GHCR 仓库的频率探测并自动升级整个镜像框架。

### 有关 Compose 多版本的变体形式

| 选用文件 | 适配应用场景 |
|------|----------|
| `docker-compose.validator.yml` | 生产环境首选（拉取对应：`ghcr.io/.../trajectoryrl:latest` 镜像） |
| `docker-compose.validator-staging.yml` | 灰度预上线或准生产环境（对应拉取的是 `:staging` 标签） |
| `docker-compose.validator-dev.yml` | 开发环境（支持本地构建以及将代码卷区双向挂靠等功能） |

## 故障排查

### 验证者启动失败
```bash
# 检查相关的起步加载日志
docker compose -f docker/docker-compose.validator.yml logs validator

# 你需要着重寻找检查：
#   [entrypoint] mock-tools ready
#   [entrypoint] OpenClaw gateway ready
#   [entrypoint] Starting validator...

# 常见的几类问题错误：
# 1. 确实所需 API Key 参数 → 请在 .env.validator 补充设定 CLAWBENCH_LLM_API_KEY
# 2. 识别/获取不到所需的钱包文件 → 请务要确保你的 ~/.bittensor/wallets 真实存在且文件配置正确
```

### 得分雷同 / 总是瞬间完成评估
该异常表征常指 OpenClaw 在无预警式死机或崩溃。排查并翻阅含有错误关键字眼的后台容器运行异常报错：
```bash
docker compose -f docker/docker-compose.validator.yml logs validator | grep -i "openclaw\|error"
```

### 网络连通故障
```bash
# 用于连通测验链网络交互状况
docker exec trajectoryrl_validator python -c "import bittensor as bt; print(bt.Subtensor(network='finney'))"
```

## 安全性要求建议

1. **钱包安全**：确保存放于受加密保护的文件系统之上，而且以只读权限模式映射挂靠进容器环境里。
2. **API 密钥**：切记采用独立解耦后的 `.env.validator` 进行传参（永远不要将配置给 Git Commit 或者 Push 到代码库里去），并且保持要周期性进行翻转换密配置更新。
3. **网络安全层**：鉴于核心主程验证器沿用本地的主机网群(host networking) 体系；还请适当地利用外部的网络防火墙策略约束去截留过滤没有必要的内部流入连接及外部探测等流量请求。

## 支持

- GitHub 问询: https://github.com/trajectoryRL/trajectoryRL/issues
- 官方主站点: https://trajrl.com
