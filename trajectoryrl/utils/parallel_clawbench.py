"""Parallel ClawBench harness — runs scenarios concurrently.

Each scenario gets its own mock-tools + OpenClaw instance on unique ports
with a dedicated workspace directory, enabling safe parallel evaluation.

Slot layout (for slot index i):
    mock-tools:  port  3001 + i    (http://localhost:{3001+i})
    OpenClaw:    port 18789 + i    (http://localhost:{18789+i})
    workspace:   /workspace_{i}    (or configurable base)
"""

import asyncio
import json
import logging
import os
import shutil
import signal
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .clawbench import ClawBenchHarness, EvaluationResult

logger = logging.getLogger(__name__)

MOCK_TOOLS_BASE_PORT = 3001
OPENCLAW_BASE_PORT = 18789


class ServiceSlot:
    """A mock-tools + OpenClaw pair running on dedicated ports with its own workspace."""

    def __init__(
        self,
        slot_index: int,
        clawbench_path: Path,
        workspace_base: Path,
        openclaw_bin: Path,
        openclaw_home_base: Path,
        env_overrides: Dict[str, str],
        slot_label: Optional[str] = None,
    ):
        self.slot_index = slot_index
        label = slot_label or str(slot_index)
        self.mock_port = MOCK_TOOLS_BASE_PORT + slot_index
        self.openclaw_port = OPENCLAW_BASE_PORT + slot_index
        self.mock_url = f"http://localhost:{self.mock_port}"
        self.openclaw_url = f"http://localhost:{self.openclaw_port}"
        self.workspace = workspace_base / f"workspace_{label}"
        self.openclaw_home = openclaw_home_base / f"openclaw_home_{label}"
        self.clawbench_path = clawbench_path
        self.openclaw_bin = openclaw_bin
        self.env_overrides = env_overrides

        self._mock_proc: Optional[asyncio.subprocess.Process] = None
        self._openclaw_proc: Optional[asyncio.subprocess.Process] = None
        self._started = False

    async def start(self) -> None:
        """Start mock-tools and OpenClaw for this slot."""
        if self._started:
            return

        self.workspace.mkdir(parents=True, exist_ok=True)
        self.openclaw_home.mkdir(parents=True, exist_ok=True)

        # Generate openclaw.json for this slot
        self._generate_openclaw_config()

        # Build environment for subprocesses
        env = {**os.environ, **self.env_overrides}
        env["MOCK_TOOLS_URL"] = self.mock_url
        env["OPENCLAW_URL"] = self.openclaw_url
        env["WORKSPACE_PATH"] = str(self.workspace)

        # OpenClaw reads OPENAI_* env vars
        env["OPENAI_API_KEY"] = env.get("CLAWBENCH_LLM_API_KEY", "")
        env["OPENAI_BASE_URL"] = env.get(
            "CLAWBENCH_LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"
        )
        env["OPENCLAW_GATEWAY_TOKEN"] = env.get(
            "OPENCLAW_GATEWAY_TOKEN", "sandbox-token-12345"
        )
        # Point OpenClaw at this slot's config and port
        env["OPENCLAW_HOME"] = str(self.openclaw_home)
        env["OPENCLAW_GATEWAY_PORT"] = str(self.openclaw_port)

        # Start mock-tools
        logger.info(
            "Slot %d: starting mock-tools on port %d", self.slot_index, self.mock_port
        )
        self._mock_proc = await asyncio.create_subprocess_exec(
            "python", "-m", "mock_tools.server",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**env, "MOCK_TOOLS_PORT": str(self.mock_port)},
            cwd=str(self.clawbench_path),
        )

        if not await self._wait_for_health(self.mock_url, timeout=30):
            raise RuntimeError(
                f"Slot {self.slot_index}: mock-tools failed to start on port {self.mock_port}"
            )
        logger.info("Slot %d: mock-tools ready on port %d", self.slot_index, self.mock_port)

        # Start OpenClaw
        logger.info(
            "Slot %d: starting OpenClaw on port %d", self.slot_index, self.openclaw_port
        )
        self._openclaw_proc = await asyncio.create_subprocess_exec(
            "node", "dist/index.js", "gateway",
            "--allow-unconfigured", "--bind", "loopback",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=str(self.openclaw_bin),
        )

        if not await self._wait_for_health(self.openclaw_url, timeout=60):
            raise RuntimeError(
                f"Slot {self.slot_index}: OpenClaw failed to start on port {self.openclaw_port}"
            )
        logger.info(
            "Slot %d: OpenClaw ready on port %d", self.slot_index, self.openclaw_port
        )

        self._started = True

    def _generate_openclaw_config(self) -> None:
        """Generate openclaw.json from template for this slot's ports."""
        template_path = self.clawbench_path / "config" / "openclaw.json"
        if not template_path.exists():
            template_path = self.clawbench_path / "config" / "openclaw.json.template"
        if not template_path.exists():
            logger.warning("Slot %d: no openclaw.json template found", self.slot_index)
            return

        config_text = template_path.read_text()

        model = self.env_overrides.get(
            "CLAWBENCH_DEFAULT_MODEL",
            os.environ.get("CLAWBENCH_DEFAULT_MODEL", "zhipu/glm-5"),
        )
        base_url = self.env_overrides.get(
            "CLAWBENCH_LLM_BASE_URL",
            os.environ.get("CLAWBENCH_LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
        )
        api_key = self.env_overrides.get(
            "CLAWBENCH_LLM_API_KEY",
            os.environ.get("CLAWBENCH_LLM_API_KEY", ""),
        )

        config_text = config_text.replace("${CLAWBENCH_DEFAULT_MODEL}", model)
        config_text = config_text.replace("${CLAWBENCH_LLM_BASE_URL}", base_url)
        config_text = config_text.replace("${CLAWBENCH_LLM_API_KEY}", api_key)
        config_text = config_text.replace("${MOCK_TOOLS_URL}", self.mock_url)

        # Also patch the gateway port in the config JSON
        try:
            config = json.loads(config_text)
            config["gateway"]["port"] = self.openclaw_port
            # Update workspace path
            if "agents" in config and "defaults" in config["agents"]:
                config["agents"]["defaults"]["workspace"] = str(self.workspace)
            # Update mock-tools URL in plugin config
            if "plugins" in config and "entries" in config["plugins"]:
                cb_tools = config["plugins"]["entries"].get("clawbench-tools", {})
                if "config" in cb_tools:
                    cb_tools["config"]["mockServerUrl"] = self.mock_url
            config_text = json.dumps(config, indent=2)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Slot %d: could not patch openclaw.json: %s", self.slot_index, e)

        # OpenClaw resolves config at $OPENCLAW_HOME/.openclaw/openclaw.json
        config_dir = self.openclaw_home / ".openclaw"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_dest = config_dir / "openclaw.json"
        config_dest.write_text(config_text)
        logger.debug("Slot %d: wrote openclaw.json to %s", self.slot_index, config_dest)

    async def _wait_for_health(self, url: str, timeout: int = 30) -> bool:
        """Poll a health endpoint until ready."""
        import urllib.request
        import urllib.error

        health_url = f"{url}/health"
        for _ in range(timeout):
            try:
                urllib.request.urlopen(health_url, timeout=2)
                return True
            except (urllib.error.URLError, OSError):
                await asyncio.sleep(1)
        return False

    async def stop(self) -> None:
        """Stop mock-tools and OpenClaw for this slot."""
        for name, proc in [("mock-tools", self._mock_proc), ("OpenClaw", self._openclaw_proc)]:
            if proc is not None and proc.returncode is None:
                try:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        proc.kill()
                        await proc.wait()
                    logger.info("Slot %d: %s stopped", self.slot_index, name)
                except Exception as e:
                    logger.warning("Slot %d: error stopping %s: %s", self.slot_index, name, e)
        self._mock_proc = None
        self._openclaw_proc = None
        self._started = False

    def create_harness(
        self,
        clawbench_path: Path,
        timeout: int,
        clawbench_default_model: str,
        clawbench_api_key: str,
        clawbench_base_url: str,
    ) -> ClawBenchHarness:
        """Create a ClawBenchHarness configured for this slot."""
        return ClawBenchHarness(
            clawbench_path=clawbench_path,
            timeout=timeout,
            workspace_path=self.workspace,
            clawbench_default_model=clawbench_default_model,
            clawbench_api_key=clawbench_api_key,
            clawbench_base_url=clawbench_base_url,
            extra_env={
                "MOCK_TOOLS_URL": self.mock_url,
                "OPENCLAW_URL": self.openclaw_url,
            },
        )


class ParallelClawBenchHarness:
    """Manages multiple service slots for parallel scenario evaluation.

    Spins up N independent (mock-tools + OpenClaw + workspace) triplets,
    then dispatches scenarios across them concurrently.

    Usage:
        harness = ParallelClawBenchHarness(
            num_slots=5,
            clawbench_path=config.clawbench_path,
            ...
        )
        await harness.start()
        results = await harness.evaluate_pack_parallel(pack, scenarios, ...)
        await harness.stop()
    """

    def __init__(
        self,
        num_slots: int,
        clawbench_path: Path,
        scenario_names: Optional[List[str]] = None,
        timeout: int = 300,
        workspace_base: Optional[Path] = None,
        openclaw_bin: Optional[Path] = None,
        openclaw_home_base: Optional[Path] = None,
        clawbench_default_model: str = "zhipu/glm-5",
        clawbench_api_key: str = "",
        clawbench_base_url: str = "https://open.bigmodel.cn/api/paas/v4",
        env_overrides: Optional[Dict[str, str]] = None,
    ):
        self.num_slots = num_slots
        self.clawbench_path = clawbench_path
        self.timeout = timeout
        self.clawbench_default_model = clawbench_default_model
        self.clawbench_api_key = clawbench_api_key
        self.clawbench_base_url = clawbench_base_url

        workspace_base = workspace_base or Path("/tmp/parallel_workspaces")
        openclaw_bin = openclaw_bin or Path("/app/openclaw")
        openclaw_home_base = openclaw_home_base or Path("/tmp/parallel_openclaw_homes")

        _env = env_overrides or {}

        self.slots: List[ServiceSlot] = []
        for i in range(num_slots):
            # Use scenario name for directory naming if available
            slot_label = scenario_names[i] if scenario_names and i < len(scenario_names) else str(i)
            slot = ServiceSlot(
                slot_index=i,
                slot_label=slot_label,
                clawbench_path=clawbench_path,
                workspace_base=workspace_base,
                openclaw_bin=openclaw_bin,
                openclaw_home_base=openclaw_home_base,
                env_overrides=_env,
            )
            self.slots.append(slot)

        # Create per-slot harnesses (available after start)
        self._harnesses: List[ClawBenchHarness] = []
        self._slot_semaphore = asyncio.Semaphore(num_slots)
        self._slot_lock = asyncio.Lock()
        self._available_slots: List[int] = list(range(num_slots))

    async def start(self) -> None:
        """Start all service slots in parallel."""
        logger.info("Starting %d parallel service slots...", self.num_slots)

        # Start all slots concurrently
        results = await asyncio.gather(
            *[slot.start() for slot in self.slots],
            return_exceptions=True,
        )

        # Check for failures
        failed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Slot %d failed to start: %s", i, result)
                failed.append(i)

        if failed:
            # Stop any that started successfully
            await self._stop_slots([i for i in range(self.num_slots) if i not in failed])
            raise RuntimeError(
                f"Failed to start slots: {failed}. "
                f"Check port conflicts or resource limits."
            )

        # Create harnesses for each slot
        self._harnesses = [
            slot.create_harness(
                clawbench_path=self.clawbench_path,
                timeout=self.timeout,
                clawbench_default_model=self.clawbench_default_model,
                clawbench_api_key=self.clawbench_api_key,
                clawbench_base_url=self.clawbench_base_url,
            )
            for slot in self.slots
        ]

        self._available_slots = list(range(self.num_slots))
        logger.info("All %d slots started successfully", self.num_slots)

    async def stop(self) -> None:
        """Stop all service slots."""
        logger.info("Stopping %d parallel service slots...", self.num_slots)
        await self._stop_slots(list(range(self.num_slots)))
        logger.info("All slots stopped")

    async def _stop_slots(self, indices: List[int]) -> None:
        await asyncio.gather(
            *[self.slots[i].stop() for i in indices],
            return_exceptions=True,
        )

    async def _acquire_slot(self) -> int:
        """Acquire a free slot index."""
        await self._slot_semaphore.acquire()
        async with self._slot_lock:
            return self._available_slots.pop(0)

    async def _release_slot(self, slot_index: int) -> None:
        """Release a slot back to the pool."""
        async with self._slot_lock:
            self._available_slots.append(slot_index)
        self._slot_semaphore.release()

    async def evaluate_scenario(
        self,
        pack: dict,
        scenario_name: str,
        seed: int = 0,
        context_preamble: str = "",
        user_context: Optional[Dict[str, str]] = None,
    ) -> EvaluationResult:
        """Evaluate a single scenario using an available slot."""
        slot_idx = await self._acquire_slot()
        try:
            harness = self._harnesses[slot_idx]
            logger.info(
                "Evaluating %s on slot %d (ports %d/%d)",
                scenario_name, slot_idx,
                self.slots[slot_idx].mock_port,
                self.slots[slot_idx].openclaw_port,
            )
            result = await harness.evaluate_pack(
                pack=pack,
                scenario_name=scenario_name,
                seed=seed,
                context_preamble=context_preamble,
                user_context=user_context,
            )
            return result
        finally:
            await self._release_slot(slot_idx)

    async def evaluate_pack_parallel(
        self,
        pack: dict,
        scenario_names: List[str],
        seed: int = 0,
        context_preamble: str = "",
        user_context: Optional[Dict[str, str]] = None,
        fail_fast: bool = True,
    ) -> Dict[str, EvaluationResult]:
        """Evaluate a pack across all scenarios in parallel.

        Args:
            pack: Policy pack dictionary (OPP v1 format)
            scenario_names: List of scenario names to evaluate
            seed: Epoch seed
            context_preamble: Epoch context prepended to AGENTS.md
            user_context: Template overrides
            fail_fast: If True, cancel remaining scenarios on first failure

        Returns:
            Dict mapping scenario_name -> EvaluationResult
        """
        results: Dict[str, EvaluationResult] = {}
        cancel_event = asyncio.Event() if fail_fast else None

        async def _eval_one(scenario_name: str) -> Tuple[str, EvaluationResult]:
            if cancel_event and cancel_event.is_set():
                return scenario_name, EvaluationResult(
                    scenario_name=scenario_name,
                    score=0.0,
                    success=False,
                    tool_calls=0,
                    response="",
                    rubric={},
                    error="Cancelled (fail-fast)",
                )
            result = await self.evaluate_scenario(
                pack=pack,
                scenario_name=scenario_name,
                seed=seed,
                context_preamble=context_preamble,
                user_context=user_context,
            )
            if fail_fast and cancel_event and result.error:
                cancel_event.set()
            return scenario_name, result

        tasks = [_eval_one(s) for s in scenario_names]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for item in completed:
            if isinstance(item, Exception):
                logger.error("Parallel eval error: %s", item)
                continue
            name, result = item
            results[name] = result

        return results

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *args):
        await self.stop()
