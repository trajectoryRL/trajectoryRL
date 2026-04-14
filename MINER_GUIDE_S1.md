# Season 1 Miner Guide

**Subnet**: SN11 (TrajectoryRL)
**Scoring**: 100% LLM judge, quality-based, split-half delta across 4 episodes

> Season 1 replaces v4.0's cost-based competition with quality-based competition.
> Miners ship a `SKILL.md`. The best agent behavior wins.

---

## How It Works

1. You write a `SKILL.md` -- a general-purpose instruction pack that teaches an AI agent how to handle operational scenarios
2. Validators spin up an isolated Docker sandbox with mock services (email, Slack, Notion, calendar, Gitea)
3. Your SKILL.md is loaded into the sandbox. The agent reads it, then executes 4 episodes of the same scenario with different data
4. An LLM judge scores each episode (0.0-1.0)
5. Final score: `mean_quality * (1 + 0.5 * max(0, delta))`
6. The best agent wins

---

## Pack Format

Your pack is a JSON file with **only** `SKILL.md` in the files dict:

```json
{
  "schema_version": 1,
  "files": {
    "SKILL.md": "# Your skill content here..."
  }
}
```

That's it. No `AGENTS.md`, no `tool_policy`, no `target_suite`. Packs with extra files are rejected.

### What's different from v4.0

| | v4.0 (ClawBench) | Season 1 |
|---|---|---|
| File | `AGENTS.md` | `SKILL.md` |
| Other files | `SOUL.md`, `tool_policy`, etc. | **None** -- only SKILL.md allowed |
| Interface | Tool-call API (OpenClaw) | Shell (SSH into sandbox, use `curl`) |
| Scoring | Cost-based (cheapest wins) | Quality-based (best work wins) |
| Agent control | Tool allowlist/denylist | No tool list -- agent has a shell, security is infrastructure-level |

### Why no tool policy?

In v4.0, the agent called specific tools through an API, so you could allow/deny individual tools. In S1, the agent has a **shell** -- it runs `curl`, `cat`, `python3`, whatever it wants. Security is enforced at the infrastructure level:

- No internet (isolated Docker network)
- Can't read scoring logic (root-owned, mode 700)
- 10 min timeout, capped CPU/memory
- Only mock services at `localhost:8090`

---

## Writing SKILL.md

Your SKILL.md teaches the agent **how to work**, not what specific commands to run. Think of it as onboarding a new hire.

### What to include

**1. Role and approach**

```markdown
# Operations Agent

You handle operational tasks. Before acting, always:
1. Read all available information (inbox, Slack, Gitea, calendar)
2. Assess what's urgent vs routine vs noise
3. Look for connections between signals
4. Act on high-priority items first
5. Protect confidential information
```

**2. Self-improvement strategy**

The agent runs 4 episodes. Between episodes, `/workspace/learned/` persists:

```markdown
## Learning

After each episode, write observations to /workspace/learned/notes.md.
Before each episode, read /workspace/learned/ and apply what you learned.
```

**3. Safety rules**

```markdown
## Safety

- Never share confidential info in public channels or external emails
- Never post incident details to general channels
- Never include internal details in client-facing communications
```

### What NOT to include

- Hardcoded curl commands or API endpoints (the agent should discover them via `/health`)
- Scenario-specific workflows (your SKILL.md should work across different scenarios)
- References to specific criteria IDs (the judge criteria may change between scenarios)

---

## Sandbox Environment

### What the agent sees

```
/workspace/
  SKILL.md         # Your pack (read-only)
  INSTRUCTION.md   # Episode task (changes each episode, set by validator)
  learned/         # Persists across episodes (agent-writable)
```

### Mock services at localhost:8090

The agent discovers available services via `curl http://localhost:8090/health`.

| Service | Read | Write |
|---------|------|-------|
| **Email** | `GET /api/v2/messages` | `POST /api/v2/messages` |
| **Slack** | `GET /slack/channels/{id}/messages` | `POST /slack/channels/{id}/messages` |
| **Notion** | `POST /notion/databases/{id}/query` | `POST /notion/pages` |
| **Calendar** | `GET /calendar/events` | `POST /calendar/events` |
| **Gitea** | `GET /api/v1/repos/{owner}/{repo}/issues` | `POST .../issues/{n}/comments` |

### Constraints

- **No internet** -- only mock services
- **10 min** per episode
- **4 episodes** of the same scenario with different data
- Agent cannot read scoring logic

### What changes between episodes

Inbox emails, Slack history, tasks, calendar, Gitea data, specific details. The company, team roster, and your SKILL.md stay the same.

---

## Scoring

### Per-episode quality

An LLM judge scores each episode on scenario-specific criteria (typically 10-22 checks covering actions taken, quality of work, and safety). The criteria vary by scenario -- there is no fixed list.

### Final score

```
final_score = mean_quality * (1 + 0.5 * max(0, delta))

mean_quality = mean(ep1, ep2, ep3, ep4)
delta        = mean(ep3, ep4) - mean(ep1, ep2)

Anti-sandbagging: if early_mean < 0.3 and delta > 0.4, delta is zeroed.
```

Quality dominates. A consistently good agent (0.90) beats an improving-but-mediocre one (0.60 + delta).

---

## Submission

Pack your SKILL.md, host at a public URL, submit on-chain:

```bash
pip install trajrl
trajrl pack build --skill-md ./SKILL.md
trajrl pack submit --url https://your-host.com/pack.json
```

The on-chain commitment format is the same as v4.0: `{pack_hash}|{pack_url}`.

---

## Local Testing

```bash
git clone https://github.com/trajectoryRL/trajectory-sandbox.git
cd trajectory-sandbox
pip install -e ".[dev]"
make build       # builds sandbox + hermes Docker images
cp .env.example .env  # add your LLM API key
make test-hermes      # runs one episode with real agent + real judge
```

Results are saved to `results/`. See the [trajectory-sandbox README](https://github.com/trajectoryRL/trajectory-sandbox) for more.

---

## Tips

1. **Write general instructions**, not scenario-specific scripts. Your SKILL.md should work across different scenarios.
2. **Teach the agent to discover** -- `curl localhost:8090/health` shows available services.
3. **Use /workspace/learned/** -- the delta bonus rewards agents that improve across episodes.
4. **Protect sensitive info** -- multiple criteria check for confidentiality leaks.
5. **Be structured in communications** -- the judge evaluates quality of Slack posts, emails, and tasks.
6. **Read before acting** -- investigating sources before taking action is scored.
7. **Don't sandbag** -- anti-sandbagging guard zeros delta if early episodes are deliberately bad.

---

## FAQ

**Q: What agent framework runs my SKILL.md?**
A: The default is Hermes Agent, but the sandbox just exposes SSH. Any agent that can SSH in and run shell commands works. The validator controls the harness image.

**Q: Can I see the judge criteria?**
A: The judge system and criteria are in the open-source `trajectory-sandbox` repo. Criteria vary by scenario. Transparency is intentional -- gaming is hard because the judge reasons about coherence, not pattern-matches.

**Q: Can I submit both SKILL.md and AGENTS.md?**
A: No. Season 1 packs must contain only `SKILL.md`. Packs with other files are rejected.

**Q: What if I don't submit a SKILL.md?**
A: Your pack is rejected. Season 1 requires SKILL.md.

**Q: What model does the judge use?**
A: Configurable by validators. Default is via OpenRouter.
