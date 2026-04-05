# TrajectoryRL

> **Bittensor 子网 11** — 通过去中心化竞争优化 AI agent 策略

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Bittensor](https://img.shields.io/badge/bittensor-7.0+-green.svg)](https://github.com/opentensor/bittensor)

TrajectoryRL 是一个 Bittensor 子网，矿工通过竞争来优化面向真实任务的 AI agent 策略包。验证者使用确定性场景评估这些策略包，并奖励那些 **安全**、**高效** 且 **可靠** 的 agent。

## 概览

```
┌──────────────────────────────────────────────────────────────┐
│                   TRAJECTORYRL SUBNET (SN11)                 │
│                                                              │
│  MINERS                              VALIDATORS              │
│  ┌───────────────┐                   ┌───────────────────┐   │
│  │ upload        │   on-chain        │ Read commitments  │   │
│  │ pack.json to  │   commitment      │ from chain        │   │
│  │ public HTTP   │─────────────────> │                   │   │
│  │ endpoint      │                   │ Fetch packs via   │   │
│  └───────────────┘                   │ HTTP, verify      │   │
│        │                             │ hash + timestamp  │   │
│        │                             │                   │   │
│        │                             │ Evaluate via      │   │
│        │                             │ ClawBench         │   │
│        │                             └───────────────────┘   │
│        │                                      │              │
│        │                                      │ set_weights  │
│        ▼                                      ▼              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              BITTENSOR BLOCKCHAIN                    │    │
│  │   Commitments, weights, TAO rewards                  │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

- **无需服务器** — 矿工可以把 pack 上传到任意 HTTP 端点，并在链上提交元数据。不需要公网 IP，也不需要持续在线。
- **两阶段评估** — [ClawBench](https://github.com/trajectoryRL/clawbench) 场景使用固定 fixture；LLM-as-judge 会根据自然语言标准对轨迹打分（阶段 1：pack 完整性，阶段 2：轨迹质量）
- **内容寻址** — pack 通过 SHA256 hash 标识，并与链上承诺进行校验
- **赢家通吃** — 表现最好的矿工获得 100% 奖励；先发优势保护早期创新者
- **反抄袭** — 链上区块时间戳 + NCD 相似度检测 + 先发阈值（delta=0.10）

有关完整的评分、奖励和反作弊细节，请参见 [INCENTIVE_MECHANISM.md](INCENTIVE_MECHANISM.md)。

### ROI 示例（每天 1,000 个任务）

```
未优化的 GLM-5:                      $12,300/月

阶段 1 — 提示词优化（AGENTS.md 调优）：
	优化提示词 + 停止规则：              $3,300/月  （降低 73%）

阶段 2 — 混合路由（AGENTS.md + 注入式技能）：
	多模型动态路由：                     $900/月    （降低 93%）
		├─ Qwen 3.5（阿里巴巴）处理 40% 的子任务（工具调用、检索）
		├─ GLM-5（Z.ai）处理 25%（结构化抽取、格式化）
		├─ Gemini 3 Flash（Google）处理 20%（搜索、总结）
		├─ GPT-5.2（OpenAI）处理 10%（推理、起草）
		└─ Claude Opus 4.6（Anthropic）处理 5%（复杂判断）
```

## 快速开始

### 适用于验证者

验证者通过 Docker 运行，并通过 Watchtower 自动从 GHCR 获取更新。当有新代码推送到 `prod` 分支时，GitHub Actions 会构建新镜像，Watchtower 会在 5 分钟内自动拉取并重启。

#### 1. 前置条件（一次性）

```bash
# 安装 btcli
pip install bittensor-cli

# 创建或导入钱包
btcli wallet create --wallet-name my-validator

# 在 SN11 上注册 hotkey（约 0.2 TAO 燃烧费用）
btcli subnets register --wallet-name my-validator --hotkey default --netuid 11

# 质押 alpha，使你的权重计入统计（验证者许可要求 stake 排名前 64）
btcli stake add --wallet-name my-validator --hotkey default --netuid 11 --amount 100
```

#### 2. 配置环境

```bash
cat > .env.validator <<'EOF'
WALLET_NAME=my-validator
WALLET_HOTKEY=default
NETUID=11
NETWORK=finney
CLAWBENCH_LLM_API_KEY=your-api-key
CLAWBENCH_LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
CLAWBENCH_DEFAULT_MODEL=zhipu/glm-5
EOF
```

**支持的提供方**（任何兼容 OpenAI 的 API 都可以）：

| 提供方 | `CLAWBENCH_LLM_BASE_URL` | `CLAWBENCH_DEFAULT_MODEL` |
|----------|--------------------------|---------------------------|
| [Zhipu AI](https://bigmodel.cn)（默认） | `https://open.bigmodel.cn/api/paas/v4` | `zhipu/glm-5` |
| [Chutes](https://chutes.ai) | `https://llm.chutes.ai/v1` | `chutes/zai-org/GLM-5-TEE` |
| [OpenRouter](https://openrouter.ai) | `https://openrouter.ai/api/v1` | `openrouter/zhipu/glm-5` |

| 变量 | 必需 | 说明 |
|----------|:--------:|-------------|
| `WALLET_NAME` | 是 | Bittensor 钱包名称 |
| `WALLET_HOTKEY` | 是 | hotkey 名称（通常是 `default`） |
| `NETUID` | 是 | 子网 UID（`11`） |
| `NETWORK` | 是 | `finney`、`test` 或 `local` |
| `CLAWBENCH_LLM_API_KEY` | 是 | 用于评估的 LLM API key（例如 [Zhipu AI](https://bigmodel.cn)、[Chutes](https://chutes.ai)、[OpenRouter](https://openrouter.ai)） |
| `CLAWBENCH_LLM_BASE_URL` | 是 | OpenAI 兼容 API 的 base URL |
| `CLAWBENCH_DEFAULT_MODEL` | 是 | 用于评估的 LLM 模型（默认：`zhipu/glm-5`） |
| `JUDGE_MODEL` | 否 | 裁判模型（默认取 `CLAWBENCH_DEFAULT_MODEL`） |
| `JUDGE_API_KEY` | 否 | 裁判模型的 API key（默认取 `CLAWBENCH_LLM_API_KEY`） |
| `JUDGE_BASE_URL` | 否 | 裁判模型的 base URL（默认取 `CLAWBENCH_LLM_BASE_URL`） |

#### 3. 启动验证者

```bash
# 启动验证者 + Watchtower（自动从 GHCR 更新）
docker compose -f docker/docker-compose.validator.yml --env-file .env.validator up -d

# 查看日志
docker compose -f docker/docker-compose.validator.yml logs -f validator
```

Docker 容器会从挂载的 `~/.bittensor/wallets/` 目录读取钱包密钥文件。容器内部不需要安装 btcli。

> **提示：** Watchtower 每 5 分钟检查一次新镜像。若要立即更新：
> ```bash
> docker compose -f docker/docker-compose.validator.yml pull
> docker compose -f docker/docker-compose.validator.yml --env-file .env.validator up -d
> ```

有关成本模型、自动更新细节和运维指导，请参见 [VALIDATOR_OPERATIONS.md](VALIDATOR_OPERATIONS.md)。

### 适用于矿工

挖矿的本质是编写 **策略包** —— 系统提示词、工具使用规则和停止条件 —— 让 AI agent 以更安全、更低成本的方式完成任务。你不需要 GPU、不需要服务器，也不需要持续在线。

> **IP 提示：** 所有提交到 TrajectoryRL 的策略包都会发布到公共仓库，并依据 [MIT 许可证](LICENSE) 许可。提交 pack 即表示你同意该提交物可以被任何人——包括 TrajectoryRL、其他矿工以及第三方——自由使用、修改和再分发。请不要提交你不愿意在 MIT 许可下公开的内容。

#### 1. 前置条件（一次性）

```bash
pip install bittensor-cli

btcli wallet create --wallet-name my-miner
btcli subnets register --wallet-name my-miner --hotkey default --netuid 11
```

#### 2. 配置环境

```bash
cat > .env.miner <<'EOF'
WALLET_NAME=my-miner
WALLET_HOTKEY=default
NETUID=11
NETWORK=finney
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
LLM_MODEL=zhipu/glm-5
EOF
```

> **提示：** 任何兼容 OpenAI 的提供方都可以使用。若使用 OpenRouter，将 `LLM_BASE_URL` 设为 `https://openrouter.ai/api/v1`，并将 `LLM_MODEL` 设为 `zhipu/glm-5`。

#### 3. 开始挖矿

```bash
git clone https://github.com/trajectoryRL/trajectoryRL.git
cd trajectoryRL
pip install -e .

# 默认模式：生成 AGENTS.md → 构建 pack → 上传 → 提交
python neurons/miner.py run --mode default
```

> **注意**：只是让 LLM 随机生成 AGENTS.md，通常拿不到好分数。你需要主动优化并改进策略包——研究 ClawBench 场景，理解什么样的 agent 表现更好，并不断迭代提示词、工具规则和停止条件。

#### 4. 手动操作（可选）

```bash
# 从你自己的 AGENTS.md 构建 pack
python neurons/miner.py build --agents-md ./AGENTS.md -o pack.json

# 在本地验证 pack
python neurons/miner.py validate pack.json

# 查看链上状态
python neurons/miner.py status
```

#### 5. 使用 ClawBench 进行本地测试

```bash
cd clawbench
pip install -e .
# 在 .env 中设置 CLAWBENCH_LLM_API_KEY、CLAWBENCH_LLM_BASE_URL、CLAWBENCH_DEFAULT_MODEL
# 智谱示例：      CLAWBENCH_LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/, CLAWBENCH_DEFAULT_MODEL=zhipu/glm-5
# Chutes 示例：     CLAWBENCH_LLM_BASE_URL=https://llm.chutes.ai/v1,              CLAWBENCH_DEFAULT_MODEL=chutes/zai-org/GLM-5-TEE
# OpenRouter 示例： CLAWBENCH_LLM_BASE_URL=https://openrouter.ai/api/v1,           CLAWBENCH_DEFAULT_MODEL=openrouter/zhipu/glm-5

# 测试单个场景
python scripts/run_episode.py --scenario inbox_triage --variant optimized --json

# 测试全部场景
python scripts/run_batch.py
```

更多细节请参见 [MINER_OPERATIONS.md](MINER_OPERATIONS.md)，其中包括自动化模式、S3 上传、pack 格式和评分目标。

## trajrl CLI

一个独立 CLI，用于查询实时子网数据——验证者、矿工、分数、提交记录和评估日志。它既面向人类，也面向 AI agent（Claude Code、Cursor、Codex、OpenClaw、Manus）。

```bash
pip install trajrl

trajrl status                       # 网络健康概览
trajrl validators                   # 列出所有验证者
trajrl scores                       # 每个矿工的分数（自动选择验证者）
trajrl miner --uid <uid>            # 矿工详情 + 诊断信息
trajrl download -u <uid>            # 下载矿工的 pack + 评估结果
trajrl submissions --failed         # 最近失败的提交
trajrl logs --show                  # 下载并显示最新的周期日志
trajrl logs --type cycle            # 列出周期日志归档
```

当输出被管道连接时，会自动输出 JSON；在交互式终端中则输出 Rich 表格。完整文档请参见 [trajrl/README.md](trajrl/README.md)。

## 文档

- **[激励机制](INCENTIVE_MECHANISM.md)** — 评分、奖励、赢家通吃和反抄袭保护
- **[验证者运维](VALIDATOR_OPERATIONS.md)** — 成本模型、自动更新和运维指导
- **[矿工运维](MINER_OPERATIONS.md)** — pack 格式、运行模式、本地测试和提交流程
- **[ClawBench](https://github.com/trajectoryRL/clawbench)** — 评估框架（场景、fixture、评分）
- **[trajrl CLI](trajrl/README.md)** — 从终端查询实时子网数据

## 社区

- **GitHub**: https://github.com/trajectoryRL/trajectoryRL
- **Website**: https://trajrl.com

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

所有矿工提交的策略包都是公开的，并以相同的 MIT 许可证发布。作为矿工参与，即表示你接受你的提交会成为任何人都可使用的开源贡献。

---

**Built on [Bittensor](https://bittensor.com)** | **Powered by [ClawBench](https://github.com/trajectoryRL/clawbench)**
