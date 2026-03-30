"""Dual-backend CAS for consensus payloads: IPFS primary, trajrl.com/GCS fallback.

Upload: try IPFS first, then trajrl.com API (which stores to GCS) if IPFS fails.
Download: try IPFS kubo API first, then public gateways, then GCS URL fallback.
Both backends are written on upload (best-effort) for redundancy.

Pointer registry is on-chain via Bittensor ``set_commitment`` — not handled here.
"""

import asyncio
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import aiohttp

from .commitments import decode_dual_address, encode_dual_address
from .consensus import (
    ConsensusPayload,
    verify_payload_integrity,
)

logger = logging.getLogger(__name__)

# Maximum payload size (10 MB).  Consensus payloads are small JSON documents;
# anything larger is either corrupt or a denial-of-service attempt.
MAX_PAYLOAD_BYTES = 10 * 1024 * 1024


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


DOWNLOAD_TIMEOUT = 60


async def _fetch_raw(session: aiohttp.ClientSession, url: str,
                     method: str = "GET", **kwargs) -> Optional[bytes]:
    """Fetch raw bytes from a URL, return None on any failure."""
    try:
        req = session.get if method == "GET" else session.post
        async with req(url, timeout=aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT),
                       **kwargs) as resp:
            if resp.status != 200:
                return None
            return await resp.read()
    except Exception:
        return None


class IPFSBackend(CASBackend):
    """IPFS via kubo-compatible HTTP API with public gateway fallback.

    Upload:   POST {api_url}/add
    Download: tries kubo API (POST /cat) then each public gateway (GET),
              validating JSON completeness per source so truncated
              streaming responses are discarded and the next source tried.

    api_url should include the /api/v0 prefix,
    e.g. ``http://ipfs.metahash73.com:5001/api/v0``.
    """

    def __init__(
        self,
        api_url: str = "http://ipfs.metahash73.com:5001/api/v0",
        gateway_urls: Optional[List[str]] = None,
    ):
        self.api_url = api_url.rstrip("/")
        self._gateway_urls = [
            gw.rstrip("/") for gw in (gateway_urls or [])
        ]

    async def upload(self, data: bytes) -> Optional[str]:
        try:
            url = f"{self.api_url}/add"
            form = aiohttp.FormData()
            form.add_field("file", data, content_type="application/octet-stream")
            async with aiohttp.ClientSession() as session:
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
        """Download CID: try kubo API first, then public gateways sequentially.

        Unlike a plain HTTP download, IPFS kubo streams the response before
        fully resolving the content graph.  This can produce truncated bytes
        under the wall-clock timeout.  Each source is therefore validated
        (JSON parse) and discarded if corrupt, falling through to the next.
        """
        sources = [
            ("kubo API", f"{self.api_url}/cat",
             {"method": "POST", "params": {"arg": address}}),
        ]
        for gw in self._gateway_urls:
            sources.append((gw, f"{gw}/ipfs/{address}", {"method": "GET"}))

        async with aiohttp.ClientSession() as session:
            for name, url, opts in sources:
                method = opts.pop("method", "GET")
                data = await _fetch_raw(session, url, method=method, **opts)
                if data is None:
                    continue
                try:
                    json.loads(data)
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.warning(
                        "%s: truncated/corrupt payload (%d bytes, CID=%s): %s, trying next",
                        name, len(data), address[:24], e,
                    )
                    continue
                logger.debug("%s download OK: CID=%s, %d bytes", name, address[:24], len(data))
                return data

        return None


class TrajRLAPIBackend(CASBackend):
    """GCS proxy via trajrl.com API.

    Upload:   POST /api/v1/consensus/payload → stores to GCS, returns public URL
    Download: GET {url} — direct download from the public GCS URL
    """

    def __init__(
        self,
        base_url: str = "https://trajrl.com",
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
        async with aiohttp.ClientSession() as session:
            data = await _fetch_raw(session, address)
        if data is None:
            logger.warning("URL download failed: %s", address[:60])
        return data


class ConsensusStore:
    """Dual-backend CAS with IPFS primary, trajrl.com/GCS fallback.

    Pointer registry is handled on-chain (not here).
    """

    def __init__(self, ipfs: IPFSBackend, api: TrajRLAPIBackend):
        self.ipfs = ipfs
        self.api = api

    async def upload_payload(self, payload: ConsensusPayload) -> Optional[str]:
        """Upload payload to CAS. Returns a content-address string.

        Strategy: upload to both IPFS and GCS concurrently for redundancy.
        The returned string encodes whichever backends succeeded:

        * Both OK:      ``{ipfs_cid};{gcs_url}``  (dual-address)
        * IPFS only:    ``{ipfs_cid}``
        * GCS only:     ``{gcs_url}``
        * Both failed:  ``None``

        Callers (and ``download_payload``) use ``decode_dual_address`` to
        recover the individual components.
        """
        data = payload.serialize()

        ipfs_result, api_result = await asyncio.gather(
            self.ipfs.upload(data),
            self.api.upload(data),
            return_exceptions=True,
        )

        if isinstance(ipfs_result, Exception):
            logger.warning("IPFS upload raised: %s", ipfs_result)
            ipfs_result = None
        if isinstance(api_result, Exception):
            logger.warning("API upload raised: %s", api_result)
            api_result = None

        address = encode_dual_address(
            ipfs_cid=ipfs_result if ipfs_result else None,
            gcs_url=api_result if api_result else None,
        )
        if address is None:
            logger.error("Both IPFS and API upload failed for window %d", payload.window_number)
        else:
            logger.info(
                "Window %d upload: ipfs=%s, gcs=%s",
                payload.window_number,
                ipfs_result or "(none)",
                (api_result or "(none)")[:60],
            )
        return address

    async def download_payload(self, content_address: str) -> Optional[ConsensusPayload]:
        """Download, verify, and deserialize a consensus payload.

        Supports three content-address formats:

        * Dual-address ``{ipfs_cid};{gcs_url}``: try IPFS first, GCS fallback.
        * Single IPFS CID (``Qm…`` / ``bafy…``): download via IPFS only.
        * Single HTTP(S) URL: download directly from GCS.

        Legacy single-address strings (no ``;``) are auto-detected by
        ``decode_dual_address`` and handled transparently.

        Default priority: **IPFS first**, GCS fallback.
        """
        ipfs_cid, gcs_url = decode_dual_address(content_address)

        data = None

        if ipfs_cid:
            data = await self.ipfs.download(ipfs_cid)
            if data is not None:
                logger.debug("Downloaded payload via IPFS: CID=%s", ipfs_cid)

        if data is None and gcs_url:
            data = await self.api.download(gcs_url)
            if data is not None:
                logger.info(
                    "IPFS download failed/unavailable, fell back to GCS: url=%s",
                    gcs_url[:60],
                )

        if data is None:
            logger.warning(
                "Failed to download payload from all backends: ipfs=%s, gcs=%s",
                ipfs_cid or "(none)",
                (gcs_url or "(none)")[:60],
            )
            return None

        if ipfs_cid and ipfs_cid.startswith("sha256:"):
            if not verify_payload_integrity(data, ipfs_cid):
                logger.warning(
                    "Payload integrity check failed: pointer=%s, computed=%s",
                    ipfs_cid[:60],
                    "sha256:" + hashlib.sha256(data).hexdigest()[:16],
                )
                return None

        try:
            return ConsensusPayload.deserialize(data)
        except Exception as e:
            logger.warning(
                "Failed to deserialize payload (ipfs=%s, gcs=%s): %s",
                ipfs_cid or "(none)",
                (gcs_url or "(none)")[:60],
                e,
            )
            return None
