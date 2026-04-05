# TrajectoryRL 公开 API

基础 URL：`https://trajrl.com`

所有接口均为只读的 `GET` 请求，无需鉴权，响应均为 JSON 格式。

---

## 目录

- [按验证者获取得分](#get-apiscoresby-validator)
- [验证者列表](#get-apivalidators)
- [矿工详情](#get-apiminershotkey)
- [策略包详情](#get-apiminershotkeypacks-packhash)
- [近期提交记录](#get-apisubmissions)
- [评估日志](#get-apieval-logs)

---

## GET /api/scores/by-validator

获取指定验证者在过去 24 小时内上报的所有矿工最新评估结果。

### 查询参数

| 参数 | 类型 | 必填 | 说明 |
|-----------|------|----------|-------------|
| `validator` | string | 是 | 验证者 SS58 hotkey |

### 响应

| 字段 | 类型 | 说明 |
|-------|------|-------------|
| `validator` | string | 查询的验证者 hotkey |
| `entries` | array | 各矿工评估条目（见下文） |

#### `entries[]` 字段说明

| 字段 | 类型 | 说明 |
|-------|------|-------------|
| `minerHotkey` | string | 矿工 SS58 hotkey |
| `uid` | number \| null | 矿工链上 UID |
| `qualified` | boolean | 该矿工是否合格 |
| `costUsd` | number \| null | 评估成本（美元） |
| `score` | number \| null | 评估得分 |
| `weight` | number \| null | 分配给该矿工的链上权重 |
| `scenarioScores` | object \| null | 各场景评估结果（以场景名为键） |
| `packHash` | string \| null | 策略包哈希 |
| `blockHeight` | number | 本次评估的区块高度 |
| `createdAt` | string | 本次报告的 ISO 8601 时间戳 |
| `rejected` | boolean | 是否为预评估阶段拒绝 |
| `rejectionStage` | string \| null | 拒绝阶段（`"pack_fetch"` \| `"schema_validation"` \| `"integrity_check"`） |
| `rejectionDetail` | string \| null | 可读的拒绝原因 |
| `llmModel` | string \| null | 该验证者使用的 LLM 模型 |

### 响应示例

```json
{
  "validator": "5FFApaS75bvpgP9gQ5hTUdZHiTc6LB2VPP9gvHN6VQCNug6f",
  "entries": [
    {
      "minerHotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
      "uid": 7,
      "qualified": true,
      "costUsd": 8.42,
      "score": 0.95,
      "weight": 1.0,
      "scenarioScores": {
        "client_escalation": {
          "score": 1.0,
          "cost": 4.2,
          "qualified": true
        }
      },
      "packHash": "abc123def456...",
      "blockHeight": 4215678,
      "createdAt": "2026-03-23T10:30:00.000Z",
      "rejected": false,
      "rejectionStage": null,
      "rejectionDetail": null,
      "llmModel": "claude-sonnet-4-6"
    }
  ]
}
```

### 错误码

| 状态码 | 响应体 |
|--------|---------|
| 400 | `{ "error": "validator query parameter is required" }` |
| 500 | `{ "error": "Internal server error" }` |

---

## GET /api/validators

列出所有验证者及其心跳状态、软件版本、LLM 模型和运行时间戳。

### 响应

| 字段 | 类型 | 说明 |
|-------|------|-------------|
| `validators` | array | 验证者条目列表 |

#### `validators[]` 字段说明

| 字段 | 类型 | 说明 |
|-------|------|-------------|
| `hotkey` | string | 验证者 SS58 hotkey |
| `version` | string \| null | 运行中的软件版本（如 `"1.2.0"`） |
| `lastSeen` | string \| null | 最近一次心跳的 ISO 8601 时间戳 |
| `lastSetWeightsAt` | string \| null | 最后一次 `set_weights` 调用的 ISO 8601 时间戳 |
| `lastEvalAt` | string \| null | 最后一次评估完成的 ISO 8601 时间戳 |
| `llmModel` | string \| null | 该验证者使用的 LLM 模型（如 `"claude-sonnet-4-6"`） |
| `latestReport` | string \| null | 最新得分报告的 ISO 8601 时间戳 |

### 响应示例

```json
{
  "validators": [
    {
      "hotkey": "5FFApaS75bvpgP9gQ5hTUdZHiTc6LB2VPP9gvHN6VQCNug6f",
      "version": "1.2.0",
      "lastSeen": "2026-03-23T10:30:00.000Z",
      "lastSetWeightsAt": "2026-03-23T10:28:00.000Z",
      "lastEvalAt": "2026-03-23T10:25:00.000Z",
      "llmModel": "claude-sonnet-4-6",
      "latestReport": "2026-03-23T10:28:00.000Z"
    },
    {
      "hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
      "version": "1.1.8",
      "lastSeen": "2026-03-23T10:26:00.000Z",
      "lastSetWeightsAt": "2026-03-23T10:20:00.000Z",
      "lastEvalAt": "2026-03-23T10:22:00.000Z",
      "llmModel": "claude-sonnet-4-6",
      "latestReport": "2026-03-23T10:25:00.000Z"
    }
  ]
}
```

---

## GET /api/miners/:hotkey

获取指定矿工的详细评估数据，包括各验证者分项、场景汇总、近期提交记录及封禁状态。

### 路径参数

| 参数 | 类型 | 说明 |
|-----------|------|-------------|
| `hotkey` | string | 矿工 SS58 hotkey |

### 响应

| 字段 | 类型 | 说明 |
|-------|------|-------------|
| `hotkey` | string | 矿工 SS58 hotkey |
| `ownerkey` | string \| null | 矿工冷钱包地址（owner key） |
| `uid` | number \| null | 链上 UID |
| `rank` | number \| null | 排行榜名次 |
| `isBanned` | boolean | 该矿工的所有者是否当前处于封禁状态 |
| `bannedUntil` | string \| null | 封禁到期时间（ISO 8601 格式） |
| `banRecord` | object \| null | 封禁详情（见下文） |
| `qualified` | boolean | 综合资格状态 |
| `totalCostUsd` | number \| null | 汇总成本（美元） |
| `score` | number \| null | 汇总得分 |
| `validatorCount` | number | 上报该矿工的验证者数量 |
| `confidence` | string | 共识置信度（`"high"`、`"medium"`、`"low"`） |
| `coverage` | number \| null | 验证者覆盖率 |
| `costDeviation` | number \| null | 各验证者间的成本标准差 |
| `scoreDeviation` | number \| null | 各验证者间的得分标准差 |
| `hasDivergence` | boolean | 各验证者报告是否存在显著差异 |
| `isActive` | boolean | 该矿工是否处于活跃状态 |
| `isBootstrap` | boolean | 网络是否处于引导（bootstrap）模式 |
| `packHash` | string \| null | 当前策略包哈希 |
| `gcsPackUrl` | string \| null | 策略包 URL（GCS 托管） |
| `commitBlock` | number \| null | 提交策略包时的区块高度 |
| `scenarioSummary` | array | 各场景的汇总统计（见下文） |
| `validators` | array | 各验证者的评估详情（见下文） |
| `recentSubmissions` | array | 最近 5 次策略包提交记录（见下文） |

#### `banRecord` 字段说明（有封禁时）

| 字段 | 类型 | 说明 |
|-------|------|-------------|
| `failedPackCount` | number | 未通过完整性检查的策略包数量 |
| `failedPacks` | array | 失败策略包列表 |
| `bannedAt` | string \| null | 封禁开始时间（ISO 8601 格式） |
| `bannedUntil` | string \| null | 封禁到期时间（ISO 8601 格式） |
| `isBanned` | boolean | 封禁是否仍有效 |

#### `scenarioSummary[]` 字段说明

| 字段 | 类型 | 说明 |
|-------|------|-------------|
| `name` | string | 场景名称 |
| `avgCost` | number \| null | 各验证者的平均成本 |
| `avgScore` | number \| null | 各验证者的平均得分 |
| `qualCount` | number | 对该场景判定合格的验证者数量 |
| `validatorCount` | number | 上报该场景的验证者数量 |

#### `validators[]` 字段说明

| 字段 | 类型 | 说明 |
|-------|------|-------------|
| `hotkey` | string | 验证者 SS58 hotkey |
| `name` | string \| null | 验证者显示名称 |
| `qualified` | boolean | 该验证者是否判定矿工合格 |
| `costUsd` | number \| null | 该验证者上报的成本 |
| `score` | number \| null | 该验证者上报的得分 |
| `blockHeight` | number \| null | 报告时的区块高度 |
| `createdAt` | string | 报告的 ISO 8601 时间戳 |
| `rejected` | boolean | 是否为预评估阶段拒绝 |
| `rejectionStage` | string \| null | 拒绝阶段 |
| `rejectionDetail` | string \| null | 拒绝原因 |
| `scenarios` | array | 各场景分项结果：`{ name, cost, score, qualified }` |

#### `recentSubmissions[]` 字段说明

| 字段 | 类型 | 说明 |
|-------|------|-------------|
| `packHash` | string | 策略包哈希 |
| `gcsPackUrl` | string \| null | 策略包 URL（GCS 托管） |
| `qualified` | boolean \| null | 资格状态 |
| `totalCostUsd` | number \| null | 总成本 |
| `score` | number \| null | 得分 |
| `evalStatus` | string | 预评估结果（`"passed"` 或 `"failed"`） |
| `evalReason` | string \| null | 预评估失败原因 |
| `submittedAt` | string | ISO 8601 提交时间戳 |
| `evaluatedAt` | string \| null | ISO 8601 评估时间戳 |

### 响应示例

```json
{
  "hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
  "ownerkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
  "uid": 7,
  "rank": 1,
  "isBanned": false,
  "bannedUntil": null,
  "banRecord": null,
  "qualified": true,
  "totalCostUsd": 8.42,
  "score": 0.95,
  "validatorCount": 8,
  "confidence": "high",
  "coverage": 1.0,
  "costDeviation": 0.15,
  "scoreDeviation": 0.02,
  "hasDivergence": false,
  "isActive": true,
  "isBootstrap": false,
  "packHash": "abc123def456...",
  "gcsPackUrl": "https://storage.googleapis.com/...",
  "commitBlock": 4215000,
  "scenarioSummary": [
    {
      "name": "client_escalation",
      "avgCost": 4.2,
      "avgScore": 0.95,
      "qualCount": 8,
      "validatorCount": 8
    }
  ],
  "validators": [
    {
      "hotkey": "5FFApaS75bvpgP9gQ5hTUdZHiTc6LB2VPP9gvHN6VQCNug6f",
      "name": "validator-1",
      "qualified": true,
      "costUsd": 8.5,
      "score": 0.96,
      "blockHeight": 4215678,
      "createdAt": "2026-03-23T10:30:00.000Z",
      "rejected": false,
      "rejectionStage": null,
      "rejectionDetail": null,
      "scenarios": [
        { "name": "client_escalation", "cost": 4.2, "score": 1.0, "qualified": true }
      ]
    }
  ],
  "recentSubmissions": [
    {
      "packHash": "abc123def456...",
      "gcsPackUrl": "https://storage.googleapis.com/...",
      "qualified": true,
      "totalCostUsd": 8.42,
      "score": 0.95,
      "evalStatus": "passed",
      "evalReason": null,
      "submittedAt": "2026-03-23T09:00:00.000Z",
      "evaluatedAt": "2026-03-23T09:05:00.000Z"
    }
  ]
}
```

### 错误码

| 状态码 | 响应体 |
|--------|---------|
| 500 | `{ "error": "Internal server error" }` |

---

## GET /api/miners/:hotkey/packs/:packHash

获取指定矿工策略包的详细评估数据，包括各验证者和各场景的分项结果。

### 路径参数

| 参数 | 类型 | 说明 |
|-----------|------|-------------|
| `hotkey` | string | 矿工 SS58 hotkey |
| `packHash` | string | 策略包 SHA-256 哈希 |

### 响应

| 字段 | 类型 | 说明 |
|-------|------|-------------|
| `packHash` | string | 策略包哈希 |
| `gcsPackUrl` | string \| null | 策略包 URL（GCS 托管） |
| `evalStatus` | string | 预评估结果（`"passed"`、`"failed"` 或 `"pending"`） |
| `evalReason` | string \| null | 预评估失败原因 |
| `submittedAt` | string \| null | ISO 8601 提交时间戳 |
| `evaluatedAt` | string \| null | ISO 8601 评估时间戳 |
| `minerHotkey` | string | 矿工 hotkey |
| `minerUid` | number \| null | 矿工 UID |
| `minerColdkey` | string \| null | 矿工冷钱包（owner key） |
| `summary` | object | 汇总评分摘要（见下文） |
| `scenarios` | string[] | 验证者报告中涉及的场景名称列表 |
| `validators` | array | 各验证者评估详情（见下文） |

#### `summary` 字段说明

| 字段 | 类型 | 说明 |
|-------|------|-------------|
| `qualified` | boolean | 仅当所有验证者均判定该包合格时为 `true` |
| `qualifiedCount` | number | 判定合格的验证者数量 |
| `bestCost` | number \| null | 合格报告中的最低成本 |
| `avgCost` | number \| null | 所有报告的平均成本 |
| `validatorCount` | number | 上报该策略包的验证者总数 |

#### `validators[]` 字段说明

| 字段 | 类型 | 说明 |
|-------|------|-------------|
| `hotkey` | string | 验证者 hotkey |
| `name` | string \| null | 验证者显示名称 |
| `qualified` | boolean | 该验证者是否判定合格 |
| `costUsd` | number \| null | 上报的成本 |
| `score` | number \| null | 上报的得分 |
| `blockHeight` | number | 区块高度 |
| `createdAt` | string | ISO 8601 报告时间戳 |
| `rejected` | boolean | 是否为预评估阶段拒绝 |
| `rejectionStage` | string \| null | 拒绝阶段 |
| `rejectionDetail` | string \| null | 拒绝原因 |
| `scenarios` | array | 各场景结果：`{ name, cost, score, qualified }` |

### 响应示例

```json
{
  "packHash": "abc123def456...",
  "gcsPackUrl": "https://storage.googleapis.com/...",
  "evalStatus": "passed",
  "evalReason": null,
  "submittedAt": "2026-03-23T09:00:00.000Z",
  "evaluatedAt": "2026-03-23T09:05:00.000Z",
  "minerHotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
  "minerUid": 7,
  "minerColdkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
  "summary": {
    "qualified": true,
    "qualifiedCount": 8,
    "bestCost": 7.9,
    "avgCost": 8.42,
    "validatorCount": 8
  },
  "scenarios": ["client_escalation", "morning_brief"],
  "validators": [
    {
      "hotkey": "5FFApaS75bvpgP9gQ5hTUdZHiTc6LB2VPP9gvHN6VQCNug6f",
      "name": "validator-1",
      "qualified": true,
      "costUsd": 8.5,
      "score": 0.96,
      "blockHeight": 4215678,
      "createdAt": "2026-03-23T10:30:00.000Z",
      "rejected": false,
      "rejectionStage": null,
      "rejectionDetail": null,
      "scenarios": [
        { "name": "client_escalation", "cost": 4.2, "score": 1.0, "qualified": true },
        { "name": "morning_brief", "cost": 4.3, "score": 0.92, "qualified": true }
      ]
    }
  ]
}
```

### 错误码

| 状态码 | 响应体 |
|--------|---------|
| 500 | `{ "error": "Internal server error" }` |

---

## GET /api/submissions

查询最近 100 条已完成评估的策略包提交记录（包括通过和失败的）。

### 响应

| 字段 | 类型 | 说明 |
|-------|------|-------------|
| `submissions` | array | 提交记录列表（见下文） |

#### `submissions[]` 字段说明

| 字段 | 类型 | 说明 |
|-------|------|-------------|
| `id` | number | 提交记录 ID |
| `minerHotkey` | string | 矿工 SS58 hotkey |
| `packHash` | string | 策略包哈希 |
| `evalStatus` | string | 预评估结果（`"passed"` 或 `"failed"`） |
| `evalReason` | string \| null | 预评估失败原因 |
| `submittedAt` | string | ISO 8601 提交时间戳 |
| `evaluatedAt` | string \| null | ISO 8601 评估完成时间戳 |

### 响应示例

```json
{
  "submissions": [
    {
      "id": 1234,
      "minerHotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
      "packHash": "abc123def456...",
      "evalStatus": "passed",
      "evalReason": null,
      "submittedAt": "2026-03-23T09:00:00.000Z",
      "evaluatedAt": "2026-03-23T09:05:00.000Z"
    },
    {
      "id": 1233,
      "minerHotkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
      "packHash": "def789ghi012...",
      "evalStatus": "failed",
      "evalReason": "hard-coded responses detected",
      "submittedAt": "2026-03-23T08:50:00.000Z",
      "evaluatedAt": "2026-03-23T08:55:00.000Z"
    }
  ]
}
```

---

## GET /api/eval-logs

验证者上传的评估日志归档。每个条目包含一个指向 GCS 上 `.tar.gz` 格式详细评估日志的 URL。

日志分为两类：
- **`miner`** — 单次矿工评估日志，包含场景级别的工具调用、HTTP 请求、裁判得分及成本明细。
- **`cycle`** — 评估周期汇总日志，涵盖整个评估周期：metagraph 同步、矿工枚举、预评估拒绝记录、各矿工评估耗时，以及周期完成摘要。

使用 `type=cycle` 仅获取周期汇总日志；配合 `miner` 或 `pack_hash` 参数使用 `type=miner` 可钻取特定矿工的评估详情。

### 查询参数

| 参数 | 类型 | 必填 | 说明 |
|-----------|------|----------|-------------|
| `validator` | string | 否 | 按验证者 SS58 hotkey 过滤 |
| `miner` | string | 否 | 按矿工 SS58 hotkey 过滤 |
| `type` | string | 否 | `"miner"` 表示矿工评估日志，`"cycle"` 表示周期汇总日志 |
| `eval_id` | string | 否 | 按评估周期标识符过滤（如 `"20260323_143025"`） |
| `pack_hash` | string | 否 | 按策略包哈希过滤 |
| `from` | string | 否 | 日期范围起始（ISO 8601，如 `"2026-03-23T00:00:00Z"`） |
| `to` | string | 否 | 日期范围截止（ISO 8601，如 `"2026-03-23T23:59:59Z"`） |
| `limit` | number | 否 | 最大返回数量（默认：50，最大：200） |
| `offset` | number | 否 | 分页跳过的记录数（默认：0） |

### 查询示例

获取来自特定验证者的所有周期汇总日志：

```
GET /api/eval-logs?validator=5FFApaS7...&type=cycle
```

获取特定评估周期的所有日志（包括汇总日志和所有矿工评估日志）：

```
GET /api/eval-logs?eval_id=20260323_143025
```

获取特定矿工在某日期的矿工评估日志：

```
GET /api/eval-logs?miner=5GrwvaEF...&type=miner&from=2026-03-23T00:00:00Z&to=2026-03-23T23:59:59Z
```

### 响应

| 字段 | 类型 | 说明 |
|-------|------|-------------|
| `logs` | array | 评估日志条目（见下文） |

#### `logs[]` 字段说明

| 字段 | 类型 | 说明 |
|-------|------|-------------|
| `id` | string | 日志记录 ID |
| `validatorHotkey` | string | 上传该日志的验证者 |
| `evalId` | string | 评估周期标识符（格式：`YYYYMMDD_HHMMSS`） |
| `logType` | string | `"miner"`（矿工评估日志）或 `"cycle"`（完整周期日志） |
| `minerHotkey` | string \| null | 矿工 hotkey（周期级日志为 null） |
| `minerUid` | number \| null | 矿工 UID（周期级日志为 null） |
| `packHash` | string \| null | 本次评估的策略包哈希（周期级日志为 null） |
| `blockHeight` | number | 评估时的区块高度 |
| `gcsUrl` | string | GCS 上的 `.tar.gz` 日志归档公开 URL |
| `sizeBytes` | number | 归档大小（字节） |
| `createdAt` | string | ISO 8601 上传时间戳 |

### 响应示例

```json
{
  "logs": [
    {
      "id": "42",
      "validatorHotkey": "5FFApaS75bvpgP9gQ5hTUdZHiTc6LB2VPP9gvHN6VQCNug6f",
      "evalId": "20260323_143025",
      "logType": "miner",
      "minerHotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
      "minerUid": 7,
      "packHash": "abc123def456...",
      "blockHeight": 4215678,
      "gcsUrl": "https://storage.googleapis.com/bucket/logs/abc123/20260323_143025/miner.tar.gz",
      "sizeBytes": 245760,
      "createdAt": "2026-03-23T14:30:30.000Z"
    },
    {
      "id": "41",
      "validatorHotkey": "5FFApaS75bvpgP9gQ5hTUdZHiTc6LB2VPP9gvHN6VQCNug6f",
      "evalId": "20260323_143025",
      "logType": "cycle",
      "minerHotkey": null,
      "minerUid": null,
      "packHash": null,
      "blockHeight": 4215678,
      "gcsUrl": "https://storage.googleapis.com/bucket/logs/abc123/__cycle__/4215678.tar.gz",
      "sizeBytes": 51200,
      "createdAt": "2026-03-23T14:35:00.000Z"
    }
  ]
}
```

### 错误码

| 状态码 | 响应体 |
|--------|---------|
| 400 | `{ "error": "type must be \"miner\" or \"cycle\"" }` |
| 500 | `{ "error": "Internal server error" }` |

---

## 请求频率限制

- 服务端没有强制的限速机制，但建议消费者以合理间隔轮询（推荐：≥ 30 秒）。

## 错误处理

所有接口均返回 JSON 格式的错误响应。常见模式如下：

| 状态码 | 说明 |
|--------|-------------|
| 200 | 成功（部分接口在空状态时会返回降级数据，并附带 `message` 字段） |
| 400 | 缺少或无效的查询参数 |
| 500 | 服务器内部错误 |
| 502 | 上游数据源不可用（返回降级空数据） |
