# trajrl

[TrajectoryRL 子网](https://trajrl.com)（Bittensor SN11）的命令行工具（CLI）。可直接在终端中查询验证者、矿工及评估的实时数据。

专为 AI Agent（Claude Code、Cursor、Codex、OpenClaw、Manus）和人类用户双重场景设计——管道输出时返回 JSON，交互式运行时输出富文本表格。

## 安装

```bash
pip install trajrl
```

## 命令列表

```
trajrl status                          # 网络健康状态概览
trajrl validators                      # 列出所有验证者
trajrl scores                          # 各矿工得分（自动选取第一个活跃的验证者）
trajrl scores --uid <uid>              # 查询指定验证者对各矿工的评分
trajrl miner --uid <uid>               # 矿工详情及诊断信息
trajrl miner <hotkey>                  # 通过 hotkey 查询矿工
trajrl download -u <uid>               # 下载矿工当前的策略包及评估结果
trajrl download <hotkey> <pack_hash>   # 下载指定版本的策略包
trajrl submissions [--failed]          # 查看近期的策略包提交记录
trajrl logs                            # 列出评估日志存档
trajrl logs --type cycle               # 仅列出周期日志
trajrl logs --show                     # 下载并展示最新的周期日志
trajrl --version                       # 显示 CLI 版本号
```

### 全局选项

所有命令均支持以下选项：

| 选项 | 说明 |
|--------|-------------|
| `--json` / `-j` | 强制 JSON 输出（当 stdout 被管道时自动启用） |
| `--base-url URL` | 覆盖 API 地址（默认：`https://trajrl.com`，环境变量：`TRAJRL_BASE_URL`） |
| `--version` / `-v` | 显示 CLI 版本并退出 |

### v0.2.0 新特性

- **UID 全局支持**：可通过 UID 而非 hotkey 查询矿工
  ```bash
  trajrl miner --uid 65
  trajrl download -u 104
  trajrl scores --uid 221
  ```

- **`download` 命令**：替代原有的 `pack` 命令。可下载矿工的策略包及评估结果，并自动从 UID 解析 hotkey 和 pack hash。
  ```bash
  trajrl download -u 104   # 只需提供 UID，其余信息自动解析
  ```

- **统一的 `logs` 命令**：合并了原有的 `eval-history`、`cycle-log` 和旧版 `logs` 命令，一条命令覆盖所有日志操作。
  ```bash
  trajrl logs                            # 列出所有归档
  trajrl logs --type cycle               # 按类型过滤
  trajrl logs --show                     # 下载并展示最新日志
  trajrl logs --show --validator 5Cd6h...  # 指定某个验证者
  ```

- **`scores` 无参数可用**：不再因无参数而崩溃，而是自动选取第一个活跃的验证者。

## 使用示例

### 快速网络检查

```bash
trajrl status
```
```
╭──────────────────── Network Status ────────────────────╮
│   Validators: 7 total, 7 active (seen <1h)             │
│   LLM Models: zhipu/glm-5 (3), chutes/GLM-5-TEE (3)    │
│   Latest Eval: 7h ago                                  │
│   Submissions: 65 passed, 35 failed (last batch)       │
╰────────────────────────────────────────────────────────╯
```

### 检查矿工信息

```bash
trajrl miner --uid 65
```

显示排名、资格状态、成本、场景分项得分、各验证者报告、近期提交记录及封禁记录。

### 下载矿工策略包

```bash
trajrl download -u 104
```

返回策略包的缓存内容、评估状态、各验证者的场景分项数据，以及用于下载已验证策略包 JSON 的 `gcsPackUrl`。

### 查看得分

```bash
trajrl scores                  # 任意验证者
trajrl scores --uid 221        # 指定验证者
```

返回各矿工条目，包含 `qualified`（是否合格）、`costUsd`（成本）、`score`（得分）、`weight`（权重）、`scenarioScores`（场景分项得分）及拒绝原因等字段。

### 查看失败的提交记录

```bash
trajrl submissions --failed
```

显示未通过预评估完整性检查的近期策略包，并附带拒绝原因。

### 查看评估日志

```bash
trajrl logs                              # 列出所有日志归档
trajrl logs --type cycle --limit 5       # 近期周期日志
trajrl logs --show                       # 下载并展示最新周期日志
trajrl logs --miner 5HMgR6...           # 指定矿工的日志
```

### 供 AI Agent 使用的 JSON 输出

使用管道即可自动输出 JSON 格式：

```bash
trajrl validators | jq '.validators[].hotkey'
trajrl scores | jq '.entries[] | select(.qualified) | {uid, costUsd, weight}'
trajrl miner --uid 65 | jq '.scenarioSummary'
trajrl download -u 104 | jq '.gcsPackUrl'
```

在交互式终端中强制使用 JSON：

```bash
trajrl miner --uid 65 --json
```

## API 参考

所有数据均来自 [TrajectoryRL 公开 API](https://trajrl.com)——只读访问，无需鉴权。完整的接口文档请参阅 [PUBLIC_API.md](PUBLIC_API.md)。

| 接口端点 | 对应 CLI 命令 |
|----------|----------------|
| `GET /api/validators` | `trajrl validators` |
| `GET /api/scores/by-validator?validator=` | `trajrl scores [--uid <uid>]` |
| `GET /api/miners/:hotkey` | `trajrl miner [--uid <uid>]` |
| `GET /api/miners/:hotkey/packs/:hash` | `trajrl download [-u <uid>]` |
| `GET /api/submissions` | `trajrl submissions` |
| `GET /api/eval-logs` | `trajrl logs` |
| `GET /api/eval-logs` + GCS 下载 | `trajrl logs --show` |
