"""trajrl — CLI for the TrajectoryRL subnet."""

from __future__ import annotations

import json
import sys
from typing import Annotated, Optional

import typer

from trajrl.api import TrajRLClient
from trajrl import display as fmt

__version__ = "0.2.1"

app = typer.Typer(
    name="trajrl",
    help="CLI for the TrajectoryRL subnet — query live validator, miner, and evaluation data.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """CLI for the TrajectoryRL subnet."""
    pass

# -- shared option defaults ------------------------------------------------

_json_opt = typer.Option("--json", "-j", help="Force JSON output (auto when piped).")
_base_url_opt = typer.Option("--base-url", help="API base URL.", envvar="TRAJRL_BASE_URL")


def _client(base_url: str) -> TrajRLClient:
    return TrajRLClient(base_url=base_url)


def _want_json(flag: bool) -> bool:
    return flag or not sys.stdout.isatty()


def _print_json(data: dict) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def _version_callback(value: bool) -> None:
    if value:
        print(f"trajrl version {__version__}")
        raise typer.Exit()


def _resolve_validator(client: TrajRLClient, hotkey: str | None, uid: int | None) -> str:
    """Resolve a validator hotkey from hotkey or UID. Returns first active if both are None."""
    if hotkey:
        return hotkey
    validators_data = client.validators()
    valis = validators_data.get("validators", [])
    if uid is not None:
        for v in valis:
            if v.get("uid") == uid:
                return v["hotkey"]
        raise typer.BadParameter(f"No validator with UID {uid}")
    # Default: pick first active validator
    if valis:
        return valis[0]["hotkey"]
    raise typer.BadParameter("No active validators found")


def _resolve_miner(client: TrajRLClient, hotkey: str | None, uid: int | None) -> dict:
    """Resolve miner data from hotkey or UID. Returns full miner dict."""
    if hotkey is None and uid is None:
        raise typer.BadParameter("Provide miner hotkey or --uid")
    return client.miner(hotkey=hotkey, uid=uid)


# -- commands --------------------------------------------------------------

@app.command()
def status(
    json_output: Annotated[bool, _json_opt] = False,
    base_url: Annotated[str, _base_url_opt] = "https://trajrl.com",
) -> None:
    """Network health overview — validators, submissions, models."""
    client = _client(base_url)
    vali_data = client.validators()
    subs_data = client.submissions()
    if _want_json(json_output):
        _print_json({"validators": vali_data, "submissions": subs_data})
    else:
        fmt.display_status(vali_data, subs_data)


@app.command()
def validators(
    json_output: Annotated[bool, _json_opt] = False,
    base_url: Annotated[str, _base_url_opt] = "https://trajrl.com",
) -> None:
    """List all validators with heartbeat status and LLM model."""
    data = _client(base_url).validators()
    if _want_json(json_output):
        _print_json(data)
    else:
        fmt.display_validators(data)


@app.command()
def scores(
    validator: Annotated[str | None, typer.Argument(help="Validator SS58 hotkey.")] = None,
    uid: Annotated[int | None, typer.Option("--uid", "-u", help="Validator UID (alternative to hotkey)")] = None,
    json_output: Annotated[bool, _json_opt] = False,
    base_url: Annotated[str, _base_url_opt] = "https://trajrl.com",
) -> None:
    """Per-miner evaluation scores from a validator. Picks first active validator if none specified."""
    client = _client(base_url)
    hotkey = _resolve_validator(client, validator, uid)
    data = client.scores_by_validator(validator=hotkey)
    if _want_json(json_output):
        _print_json(data)
    else:
        fmt.display_scores(data)


@app.command()
def miner(
    hotkey: Annotated[str | None, typer.Argument(help="Miner SS58 hotkey.")] = None,
    uid: Annotated[int | None, typer.Option("--uid", "-u", help="Miner UID (alternative to hotkey)")] = None,
    json_output: Annotated[bool, _json_opt] = False,
    base_url: Annotated[str, _base_url_opt] = "https://trajrl.com",
) -> None:
    """Show detailed evaluation data for a specific miner."""
    client = _client(base_url)
    data = _resolve_miner(client, hotkey, uid)
    if _want_json(json_output):
        _print_json(data)
    else:
        fmt.display_miner(data)


@app.command()
def download(
    hotkey: Annotated[str | None, typer.Argument(help="Miner SS58 hotkey.")] = None,
    pack_hash: Annotated[str | None, typer.Argument(help="Pack SHA-256 hash (default: current pack).")] = None,
    uid: Annotated[int | None, typer.Option("--uid", "-u", help="Miner UID (alternative to hotkey)")] = None,
    json_output: Annotated[bool, _json_opt] = False,
    base_url: Annotated[str, _base_url_opt] = "https://trajrl.com",
) -> None:
    """Download a miner's pack and its evaluation results."""
    client = _client(base_url)
    # Resolve hotkey and pack_hash from UID if needed
    if hotkey is None and uid is not None:
        miner_data = _resolve_miner(client, None, uid)
        hotkey = miner_data["hotkey"]
        if pack_hash is None:
            pack_hash = miner_data.get("packHash")
    elif hotkey is not None and pack_hash is None:
        miner_data = _resolve_miner(client, hotkey, None)
        pack_hash = miner_data.get("packHash")

    if not hotkey or not pack_hash:
        raise typer.BadParameter("Provide miner hotkey or --uid")

    data = client.pack(hotkey, pack_hash)
    if _want_json(json_output):
        _print_json(data)
    else:
        fmt.display_pack(data)


@app.command()
def submissions(
    failed: Annotated[bool, typer.Option("--failed", help="Show only failed submissions.")] = False,
    json_output: Annotated[bool, _json_opt] = False,
    base_url: Annotated[str, _base_url_opt] = "https://trajrl.com",
) -> None:
    """Recent pack submissions (passed and failed)."""
    data = _client(base_url).submissions()
    if failed:
        data["submissions"] = [s for s in data.get("submissions", []) if s.get("evalStatus") == "failed"]
    if _want_json(json_output):
        _print_json(data)
    else:
        fmt.display_submissions(data, failed_only=failed)


@app.command()
def logs(
    validator: Annotated[Optional[str], typer.Option("--validator", "-v", help="Validator hotkey or UID.")] = None,
    miner_key: Annotated[Optional[str], typer.Option("--miner", "-m", help="Filter by miner hotkey.")] = None,
    log_type: Annotated[Optional[str], typer.Option("--type", "-t", help="Log type: 'miner' or 'cycle'.")] = None,
    eval_id: Annotated[Optional[str], typer.Option("--eval-id", help="Filter by eval cycle ID.")] = None,
    pack_hash: Annotated[Optional[str], typer.Option("--pack-hash", help="Filter by pack hash.")] = None,
    limit: Annotated[int, typer.Option("--limit", "-l", help="Max results to return.")] = 50,
    show: Annotated[bool, typer.Option("--show", "-s", help="Download and display the latest log content.")] = False,
    json_output: Annotated[bool, _json_opt] = False,
    base_url: Annotated[str, _base_url_opt] = "https://trajrl.com",
) -> None:
    """List or view evaluation log archives.

    Without --show: lists available log archives.
    With --show: downloads and displays the latest matching log.
    """
    client = _client(base_url)

    # If --show, download the actual log content
    if show:
        vali_hotkey = validator
        if vali_hotkey is None:
            vali_hotkey = _resolve_validator(client, None, None)
        try:
            data = client.cycle_log(vali_hotkey, eval_id=eval_id)
        except ValueError as e:
            if _want_json(json_output):
                _print_json({"error": str(e)})
            else:
                fmt.console.print(f"[yellow]{e}[/]")
            raise typer.Exit(1)
        if _want_json(json_output):
            _print_json(data)
        else:
            fmt.display_cycle_log(data)
        return

    # Otherwise list log archives
    data = client.eval_logs(
        validator=validator,
        miner=miner_key,
        log_type=log_type,
        eval_id=eval_id,
        pack_hash=pack_hash,
        limit=limit,
    )
    if _want_json(json_output):
        _print_json(data)
    else:
        fmt.display_logs(data)
