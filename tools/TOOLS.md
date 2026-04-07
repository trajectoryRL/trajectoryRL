# Tools

A collection of standalone diagnostic and analysis tools for the TrajectoryRL subnet. Each tool targets a specific operational need — validator inspection, pack deduplication check, etc. They are independent of the `trajrl` CLI package and run directly with `python3`.

## analyze_consensus.py — On-chain consensus & winner election simulator

Reads all validator consensus commitments from the Bittensor chain, downloads evaluation payloads from IPFS, computes stake-weighted consensus costs and qualification, then applies Winner Protection to determine the elected winner — exactly as a production validator would.

### Usage

```bash
python3 tools/analyze_consensus.py                                       # run with defaults
python3 tools/analyze_consensus.py --network finney --netuid 11          # explicit chain params
python3 tools/analyze_consensus.py --prev-winner 5Ew5P... --prev-winner-cost 0.015  # simulate winner protection
python3 tools/analyze_consensus.py --qual-threshold 0.5 --cost-delta 0.10           # tune consensus params
```

### What it shows

- **Window distribution** — which evaluation windows have validator submissions on-chain.
- **Download status** — per-validator IPFS payload download results (with per-source JSON integrity validation and automatic gateway fallback on truncated data).
- **Filter pipeline** — how many submissions passed the 6-layer filter (protocol, window, stake, integrity, version, zero-signal).
- **Consensus costs** — all miners ranked by stake-weighted consensus cost, with qualification gate (PASS/FAIL).
- **Winner election** — the elected winner, their consensus cost, and per-validator breakdown showing each validator's individual cost and qualification vote.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--network` | `finney` | Subtensor network |
| `--netuid` | `11` | Subnet UID |
| `--prev-winner` | none | Previous winner hotkey for Winner Protection simulation |
| `--prev-winner-cost` | none | Previous winner's locked-in cost |
| `--qual-threshold` | `0.5` | Stake fraction required for consensus qualification |
| `--cost-delta` | `0.10` | Winner Protection threshold (challenger must beat `winner_cost × (1 - δ)`) |

### Dependencies

Requires `bittensor`, `aiohttp`, and the `trajectoryrl` package (project root).

## analyze_validator.py — Validator evaluation analysis

Interactively inspect a validator's evaluation behavior: score distribution, miner qualification, cost breakdown, weight allocation, and per-miner drill-down.

### Usage

```bash
python3 tools/analyze_validator.py                     # interactive: list validators, pick one
python3 tools/analyze_validator.py <hotkey>             # analyze a specific validator
python3 tools/analyze_validator.py <hotkey> --deep      # include per-miner drill-down
python3 tools/analyze_validator.py --list               # just list validators
python3 tools/analyze_validator.py <hotkey> --dump      # dump raw JSON to file
```

### What it shows

- **Score summary** — qualified / rejected counts, cost stats (min / max / mean / median), score & weight distributions.
- **Weight distribution** — parsed from the validator's latest cycle log (WEIGHT RESULTS section), including per-miner weight, cost, owner hotkey, and set_weights status.
- **Per-miner deep dive** (`--deep`) — per-scenario scores, pack-level timing, and individual eval details for every miner.

### Dependencies

Requires the `trajrl` package (now a standalone repo: https://github.com/trajectoryRL/trajrl). Install with:

```bash
pip install trajrl
```

## compare_pack_ncd.py — Pack deduplication similarity check

Computes NCD (Normalized Compression Distance) similarity between two packs' `AGENTS.md` files, using the same algorithm as the validator's deduplication layer (`trajectoryrl.utils.ncd`).

### Usage

```bash
python3 tools/compare_pack_ncd.py <pack_url_a> <pack_url_b>
python3 tools/compare_pack_ncd.py <pack_url_a> <pack_url_b> --threshold 0.85
python3 tools/compare_pack_ncd.py <pack_url_a> <pack_url_b> -v    # verbose: show zlib sizes and NCD formula
python3 tools/compare_pack_ncd.py <pack_url_a> <pack_url_b> -q    # quiet: print similarity number only
python3 tools/compare_pack_ncd.py <pack_url_a> <pack_url_b> --fail-on-similar  # exit 1 if too similar
```

### What it shows

- NCD similarity score (0.0 = completely different, 1.0 = identical).
- Whether the pair would be flagged as a copy by the validator (similarity >= threshold).
- Verbose mode (`-v`) prints raw zlib compressed sizes and the NCD formula breakdown.

### Dependencies

Requires the `trajectoryrl` package (project root) for `trajectoryrl.utils.ncd`.
