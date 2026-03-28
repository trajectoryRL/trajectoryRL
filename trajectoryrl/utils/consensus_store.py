"""Dual-backend CAS for consensus payloads: IPFS primary, trajrl.com/GCS fallback.

Upload: try IPFS first, then trajrl.com API (which stores to GCS) if IPFS fails.
Download: try IPFS first, then direct URL (GCS or other) if IPFS fails.
Both backends are written on upload (best-effort) for redundancy.

Pointer registry is on-chain via Bittensor ``set_commitment`` — not handled here.
"""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from typing import Optional

import aiohttp

from .consensus import (
    ConsensusPayload,
    verify_payload_integrity,
)

logger = logging.getLogger(__name__)


class CASBackend(ABC):
    """Abstract content-addressed storage backend."""

    @abstractmethod
    async def upload(self, data: bytes) -> Optional[str]:
        """Upload data, return content address or None on failure."""
        ...

    @abstractmethod
    async def download(self, address: str) -> Optional[bytes]:
        """Download data by content address, return bytes or None."""
        ...


class IPFSBackend(CASBackend):
    """IPFS via local kubo node HTTP API.

    Upload:   POST /api/v0/add (auto-pins)
    Download: POST /api/v0/cat?arg={cid}
    """

    def __init__(self, api_url: str = "http://localhost:5001", api_token: str = ""):
        self.api_url = api_url.rstrip("/")
        self._api_token = api_token

    def _auth_headers(self) -> dict:
        if self._api_token:
            return {"Authorization": f"Bearer {self._api_token}"}
        return {}

    async def upload(self, data: bytes) -> Optional[str]:
        try:
            url = f"{self.api_url}/api/v0/add"
            form = aiohttp.FormData()
            form.add_field("file", data, content_type="application/octet-stream")
            async with aiohttp.ClientSession(headers=self._auth_headers()) as session:
                async with session.post(url, data=form, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        logger.warning("IPFS upload failed: HTTP %d", resp.status)
                        return None
                    result = await resp.json()
                    cid = result.get("Hash")
                    if cid:
                        logger.info("IPFS upload OK: CID=%s", cid)
                    return cid
        except Exception as e:
            logger.warning("IPFS upload error: %s", e)
            return None

    async def download(self, address: str) -> Optional[bytes]:
        try:
            url = f"{self.api_url}/api/v0/cat"
            async with aiohttp.ClientSession(headers=self._auth_headers()) as session:
                async with session.post(
                    url,
                    params={"arg": address},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        logger.warning("IPFS download failed: HTTP %d (CID=%s)", resp.status, address)
                        return None
                    data = await resp.read()
                    logger.debug("IPFS download OK: CID=%s, %d bytes", address, len(data))
                    return data
        except Exception as e:
            logger.warning("IPFS download error (CID=%s): %s", address, e)
            return None


class TrajRLAPIBackend(CASBackend):
    """GCS proxy via trajrl.com API.

    Upload:   POST /api/v1/consensus/payload → stores to GCS, returns public URL
    Download: GET {url} — direct download from the public GCS URL
    """

    def __init__(
        self,
        base_url: str = "https://api.trajrl.com",
        sign_fn=None,
        validator_hotkey: str = "",
    ):
        self.base_url = base_url.rstrip("/")
        self._sign_fn = sign_fn
        self._validator_hotkey = validator_hotkey

    async def upload(self, data: bytes) -> Optional[str]:
        """Upload payload via API. Returns a public GCS URL."""
        try:
            import time
            ts = int(time.time())
            payload_dict = json.loads(data.decode("utf-8"))

            body = {
                "validator_hotkey": self._validator_hotkey,
                "timestamp": ts,
                "signature": "",
                "payload": payload_dict,
            }
            if self._sign_fn:
                msg = f"trajectoryrl-consensus:{self._validator_hotkey}:{ts}"
                body["signature"] = self._sign_fn(msg)

            url = f"{self.base_url}/api/v1/consensus/payload"
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=body,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status in (200, 409):
                        result = await resp.json()
                        public_url = result.get("url")
                        if public_url:
                            logger.info("API upload OK: url=%s", public_url)
                            return public_url
                        content_hash = result.get("content_hash", "")
                        logger.info("API upload OK: hash=%s", content_hash)
                        return content_hash
                    logger.warning("API upload failed: HTTP %d", resp.status)
                    return None
        except Exception as e:
            logger.warning("API upload error: %s", e)
            return None

    async def download(self, address: str) -> Optional[bytes]:
        """Download payload by direct URL (GCS or other public URL)."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    address, timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        logger.warning("URL download failed: HTTP %d (url=%s)", resp.status, address)
                        return None
                    data = await resp.read()
                    logger.debug("URL download OK: url=%s, %d bytes", address[:60], len(data))
                    return data
        except Exception as e:
            logger.warning("URL download error (url=%s): %s", address[:60], e)
            return None


class ConsensusStore:
    """Dual-backend CAS with IPFS primary, trajrl.com/GCS fallback.

    Pointer registry is handled on-chain (not here).
    """

    def __init__(self, ipfs: IPFSBackend, api: TrajRLAPIBackend):
        self.ipfs = ipfs
        self.api = api

    async def upload_payload(self, payload: ConsensusPayload) -> Optional[str]:
        """Upload payload to CAS. Returns content address (IPFS CID or GCS URL).

        Strategy: try IPFS first, then API/GCS. Best-effort mirror to both.
        """
        data = payload.serialize()

        ipfs_cid = await self.ipfs.upload(data)

        api_url = await self.api.upload(data)

        if ipfs_cid:
            return ipfs_cid
        if api_url:
            return api_url

        logger.error("Both IPFS and API upload failed for window %d", payload.window_number)
        return None

    async def download_payload(self, content_address: str) -> Optional[ConsensusPayload]:
        """Download and verify payload from CAS.

        Determines backend by address format:
        - IPFS CID (Qm... / bafy...): download via IPFS
        - HTTP(S) URL: download directly (GCS or other public URL)
        """
        data = None

        is_url = content_address.startswith("http://") or content_address.startswith("https://")
        is_ipfs_cid = not is_url

        if is_ipfs_cid:
            data = await self.ipfs.download(content_address)

        if data is None and is_url:
            data = await self.api.download(content_address)

        if data is None:
            logger.warning("Failed to download payload: %s", content_address[:60])
            return None

        try:
            payload = ConsensusPayload.deserialize(data)
        except Exception as e:
            logger.warning("Failed to deserialize payload %s: %s", content_address[:60], e)
            return None

        expected_hash = payload.content_hash()
        if not verify_payload_integrity(data, expected_hash):
            logger.warning(
                "Payload integrity check failed: address=%s, computed=%s",
                content_address[:60], expected_hash[:24],
            )
            return None

        return payload
