#!/usr/bin/env python3
"""Standalone CAS (Content-Addressed Storage) connectivity test.

Tests IPFS kubo API and public gateway backends for consensus payload upload/download.
Reads configuration from environment variables or .env.validator file.

Usage:
    python3 scripts/test_cas.py
    python3 scripts/test_cas.py --skip-upload   # read-only test with a known CID
    python3 scripts/test_cas.py --cid QmXxx...  # download a specific CID
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import aiohttp

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("test_cas")

DEFAULT_API_URL = "http://ipfs.metahash73.com:5001/api/v0"
DEFAULT_GATEWAYS = ["https://ipfs.io", "https://dweb.link"]


def load_env(path: str = None):
    """Load key=value pairs from a dotenv file into os.environ."""
    if path is None:
        path = os.path.join(ROOT, ".env.validator")
    if not os.path.exists(path):
        logger.warning("No env file found at %s, using existing env vars", path)
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.strip().strip("'\"")
            os.environ.setdefault(key.strip(), value)
    logger.info("Loaded env from %s", path)


def get_config():
    api_url = os.getenv("IPFS_API_URL", DEFAULT_API_URL)
    gateways_raw = os.getenv("IPFS_GATEWAYS", ",".join(DEFAULT_GATEWAYS))
    gateway_urls = [gw.strip() for gw in gateways_raw.split(",") if gw.strip()]
    consensus_api_url = os.getenv("CONSENSUS_API_URL", "https://trajrl.com")
    return {
        "ipfs_api_url": api_url,
        "ipfs_gateway_urls": gateway_urls,
        "consensus_api_url": consensus_api_url,
    }


def make_test_payload() -> bytes:
    """Build a minimal JSON payload for testing."""
    payload = {
        "scoring_version": "test",
        "costs": {"test_miner_hk": 0.042},
        "disqualified": {},
        "protocol_version": 1,
        "qualified": {"test_miner_hk": True},
        "timestamp": int(time.time()),
        "validator_hotkey": "test_validator_hk",
        "window_number": 0,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------

async def test_ipfs_upload(api_url: str, data: bytes) -> str | None:
    """Upload to IPFS via kubo API (POST /add). Returns CID or None."""
    print("\n" + "=" * 60)
    print("TEST: IPFS Upload (kubo API)")
    print(f"  API URL: {api_url}")
    print(f"  Payload: {len(data)} bytes")
    print("=" * 60)

    try:
        url = f"{api_url.rstrip('/')}/add"
        print(f"  Endpoint: {url}")
        form = aiohttp.FormData()
        form.add_field("file", data, content_type="application/octet-stream")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=form, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                body = await resp.text()
                if resp.status != 200:
                    print(f"  FAIL: HTTP {resp.status}")
                    print(f"  Response: {body[:500]}")
                    return None
                result = json.loads(body)
                cid = result.get("Hash")
                size = result.get("Size", "?")
                print(f"  OK: CID={cid}, Size={size}")
                return cid
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


async def test_ipfs_api_download(api_url: str, cid: str) -> bytes | None:
    """Download from IPFS via kubo API (POST /cat). Returns data or None."""
    print("\n" + "=" * 60)
    print("TEST: IPFS API Download (kubo /cat)")
    print(f"  API URL: {api_url}")
    print(f"  CID:     {cid}")
    print("=" * 60)

    try:
        url = f"{api_url.rstrip('/')}/cat"
        print(f"  Endpoint: {url}?arg={cid}")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, params={"arg": cid}, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    print(f"  FAIL: HTTP {resp.status}")
                    print(f"  Response: {body[:500]}")
                    return None
                data = await resp.read()
                print(f"  OK: {len(data)} bytes downloaded")
                _print_content_preview(data)
                return data
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


async def test_gateway_download(gateway_url: str, cid: str) -> bytes | None:
    """Download from a public IPFS gateway (HTTP GET). Returns data or None."""
    print("\n" + "=" * 60)
    print(f"TEST: Gateway Download ({gateway_url})")
    print(f"  CID: {cid}")
    print("=" * 60)

    url = f"{gateway_url.rstrip('/')}/ipfs/{cid}"
    print(f"  URL: {url}")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    print(f"  FAIL: HTTP {resp.status}")
                    print(f"  Response: {body[:300]}")
                    return None
                data = await resp.read()
                print(f"  OK: {len(data)} bytes downloaded")
                _print_content_preview(data)
                return data
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


async def test_api_upload(api_url: str, data: bytes) -> str | None:
    """Upload to trajrl.com API (GCS proxy). Returns URL or None."""
    print("\n" + "=" * 60)
    print("TEST: API Upload (trajrl.com -> GCS)")
    print(f"  API URL: {api_url}")
    print(f"  Payload: {len(data)} bytes")
    print("=" * 60)

    try:
        payload_dict = json.loads(data.decode("utf-8"))
        ts = int(time.time())
        body = {
            "validator_hotkey": "test_validator_hk",
            "timestamp": ts,
            "signature": "",
            "payload": payload_dict,
        }
        url = f"{api_url.rstrip('/')}/api/v2/consensus/payload"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                resp_body = await resp.text()
                if resp.status in (200, 409):
                    result = json.loads(resp_body)
                    public_url = result.get("url", result.get("content_hash", ""))
                    print(f"  OK: HTTP {resp.status}, address={public_url}")
                    return public_url
                print(f"  FAIL: HTTP {resp.status}")
                print(f"  Response: {resp_body[:500]}")
                return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


async def test_url_download(url: str) -> bytes | None:
    """Download from a direct URL (GCS or other). Returns data or None."""
    print("\n" + "=" * 60)
    print("TEST: URL Download (GCS/HTTP)")
    print(f"  URL: {url}")
    print("=" * 60)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    print(f"  FAIL: HTTP {resp.status}")
                    print(f"  Response: {body[:500]}")
                    return None
                data = await resp.read()
                print(f"  OK: {len(data)} bytes downloaded")
                _print_content_preview(data)
                return data
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


async def test_roundtrip_integrity(original: bytes, downloaded: bytes) -> bool:
    """Verify uploaded and downloaded data match."""
    print("\n" + "=" * 60)
    print("TEST: Roundtrip Integrity")
    print("=" * 60)

    if original == downloaded:
        print("  OK: data matches exactly")
        return True

    try:
        orig_parsed = json.loads(original)
        down_parsed = json.loads(downloaded)
        if orig_parsed == down_parsed:
            print("  OK: JSON content matches (whitespace may differ)")
            return True
    except json.JSONDecodeError:
        pass

    print(f"  FAIL: data mismatch")
    print(f"    Original:   {len(original)} bytes")
    print(f"    Downloaded: {len(downloaded)} bytes")
    return False


def _print_content_preview(data: bytes):
    try:
        parsed = json.loads(data)
        print(f"  Content preview: {json.dumps(parsed, indent=2)[:300]}")
    except json.JSONDecodeError:
        print(f"  Content (raw): {data[:200]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Test CAS read/write connectivity")
    parser.add_argument("--skip-upload", action="store_true", help="Skip upload tests (read-only)")
    parser.add_argument("--cid", type=str, default=None, help="Download a specific IPFS CID")
    parser.add_argument("--url", type=str, default=None, help="Download a specific URL")
    parser.add_argument("--env", type=str, default=None, help="Path to .env file")
    args = parser.parse_args()

    load_env(args.env)
    config = get_config()

    print("\n" + "#" * 60)
    print("# CAS Connectivity Test")
    print("#" * 60)
    print(f"  IPFS API:  {config['ipfs_api_url']}")
    print(f"  Gateways:  {config['ipfs_gateway_urls']}")
    print(f"  API URL:   {config['consensus_api_url']}")
    results = {}

    # --- Download-only mode ---
    if args.cid:
        data = await test_ipfs_api_download(config["ipfs_api_url"], args.cid)
        results["ipfs_api_download"] = data is not None

        for gw in config["ipfs_gateway_urls"]:
            gw_data = await test_gateway_download(gw, args.cid)
            results[f"gateway_{gw}"] = gw_data is not None

        _print_summary(results)
        return

    if args.url:
        data = await test_url_download(args.url)
        results["url_download"] = data is not None
        _print_summary(results)
        return

    # --- Full test ---
    test_data = make_test_payload()
    ipfs_cid = None
    api_address = None

    if not args.skip_upload:
        ipfs_cid = await test_ipfs_upload(config["ipfs_api_url"], test_data)
        results["ipfs_upload"] = ipfs_cid is not None

        api_address = await test_api_upload(config["consensus_api_url"], test_data)
        results["api_upload"] = api_address is not None

    # IPFS API download
    if ipfs_cid:
        downloaded = await test_ipfs_api_download(config["ipfs_api_url"], ipfs_cid)
        results["ipfs_api_download"] = downloaded is not None
        if downloaded:
            results["ipfs_roundtrip"] = await test_roundtrip_integrity(test_data, downloaded)

        # Gateway fallback downloads
        for gw in config["ipfs_gateway_urls"]:
            gw_data = await test_gateway_download(gw, ipfs_cid)
            results[f"gateway_{gw}"] = gw_data is not None

    # API/URL Download
    if api_address and (api_address.startswith("http://") or api_address.startswith("https://")):
        downloaded = await test_url_download(api_address)
        results["api_download"] = downloaded is not None
        if downloaded:
            results["api_roundtrip"] = await test_roundtrip_integrity(test_data, downloaded)

    _print_summary(results)


def _print_summary(results: dict):
    print("\n" + "#" * 60)
    print("# Summary")
    print("#" * 60)
    all_pass = True
    for name, passed in results.items():
        icon = "  [+]" if passed else "  [-]"
        status = "PASS" if passed else "FAIL"
        print(f"{icon} {name}: {status}")
        if not passed:
            all_pass = False

    if not results:
        print("  No tests were run.")
    elif all_pass:
        print("\n  All tests passed!")
    else:
        print(f"\n  {sum(not v for v in results.values())} test(s) failed.")


if __name__ == "__main__":
    asyncio.run(main())
