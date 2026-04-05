# 工具集

本文档整合了用于 TrajectoryRL 子网的一系列独立诊断和分析工具。每个工具针对特定的运维需求——验证者检查、策略包去重检查等。它们独立于 `trajrl` CLI 包，并直接使用 `python3` 运行。

## analyze_consensus.py — 链上共识与赢家选举模拟器

读取 Bittensor 链上所有验证者的共识承诺，从 IPFS 下载评估数据，计算基于质押权重的共识成本和资格，然后应用赢家保护机制（Winner Protection）来决定最终胜出的矿工——其运作方式与生产环境中的验证者完全一致。

### 用法

```bash
python3 tools/analyze_consensus.py                                       # 使用默认参数运行
python3 tools/analyze_consensus.py --network finney --netuid 11          # 明确指定链参数
python3 tools/analyze_consensus.py --prev-winner 5Ew5P... --prev-winner-cost 0.015  # 模拟赢家保护
python3 tools/analyze_consensus.py --qual-threshold 0.5 --cost-delta 0.10           # 调整共识参数
```

### 功能展示

- **时间窗分布 (Window distribution)** — 链上的验证者提交数据对应哪些评估时间段。
- **下载状态 (Download status)** — 每个验证者的 IPFS 数据下载结果（包含每个数据源的 JSON 完整性验证，以及数据截断时的自动网关回退）。
- **过滤管道 (Filter pipeline)** — 有多少提交通过了 6 层数据过滤（协议、时间窗、质押、完整性、版本、零信号）。
- **共识成本 (Consensus costs)** — 所有矿工按照基于质押权重的共识成本排序，并附带资格审查结果（通过/失败）。
- **赢家选举 (Winner election)** — 胜出的赢家、其共识成本，以及针对每个验证者的明细，显示该验证者给出的单独成本和资格投票。

### 选项参数

| 标志 | 默认值 | 说明 |
|------|---------|-------------|
| `--network` | `finney` | Subtensor 网络 |
| `--netuid` | `11` | 子网 UID |
| `--prev-winner` | 无 | 用于赢家保护模拟的前任赢家 hotkey |
| `--prev-winner-cost` | 无 | 前任赢家锁定时的成本 |
| `--qual-threshold` | `0.5` | 共识合格所需的质押比例 |
| `--cost-delta` | `0.10` | 赢家保护阈值（挑战者的成本必须低于 `前任成本 × (1 - δ)`） |

### 依赖项

需要 `bittensor`、`aiohttp` 和 `trajectoryrl` 包（项目根目录）。

## analyze_validator.py — 验证者评估分析

交互式检查验证者的评估行为：分数分布、矿工资格、成本明细、权重分配以及针对每个矿工的详细分析。

### 用法

```bash
python3 tools/analyze_validator.py                     # 交互模式：列出验证者，选择其一
python3 tools/analyze_validator.py <hotkey>             # 分析特定验证者
python3 tools/analyze_validator.py <hotkey> --deep      # 包含针对每个矿工的详细追踪分析
python3 tools/analyze_validator.py --list               # 仅列出验证者
python3 tools/analyze_validator.py <hotkey> --dump      # 将原始 JSON 导出至文件
```

### 功能展示

- **分数摘要 (Score summary)** — 合格/拒绝计数，成本统计（最小/最大/平均/中位数），得分与权重分布。
- **权重分布 (Weight distribution)** — 解析自验证者最新的周期日志（WEIGHT RESULTS 区域），包含每个矿工的权重、成本、矿主 hotkey 和 set_weights 状态。
- **矿工深潜 (`--deep`)** — 每个场景的分数、策略包级别的耗时，以及每个矿工单独的评估细节。

### 依赖项

需要 `trajrl` 包（位于 `trajrl/` 中）。安装方式：

```bash
pip install -e trajrl/
```

## compare_pack_ncd.py — 策略包去重相似度检查

计算两个策略包中 `AGENTS.md` 文件之间的 NCD（归一化压缩距离）相似度，使用的算法与验证者的去重层 (`trajectoryrl.utils.ncd`) 完全相同。

### 用法

```bash
python3 tools/compare_pack_ncd.py <pack_url_a> <pack_url_b>
python3 tools/compare_pack_ncd.py <pack_url_a> <pack_url_b> --threshold 0.85
python3 tools/compare_pack_ncd.py <pack_url_a> <pack_url_b> -v    # 详细模式：显示 zlib 压缩大小和 NCD 公式详细计算过程
python3 tools/compare_pack_ncd.py <pack_url_a> <pack_url_b> -q    # 安静模式：只打印相似度数值
python3 tools/compare_pack_ncd.py <pack_url_a> <pack_url_b> --fail-on-similar  # 如果过于相似则退出返回状态码 1
```

### 功能展示

- NCD 相似度得分（0.0 = 完全不同，1.0 = 完全相同）。
- 该对数据是否会被验证者标记为抄袭（相似度 >= 阈值）。
- 详细模式 (`-v`) 打印原始 zlib 压缩后文件的大小和 NCD 算法公式的具体拆解。

### 依赖项

需要使用 `trajectoryrl` 包（项目根目录）去调用 `trajectoryrl.utils.ncd`。
