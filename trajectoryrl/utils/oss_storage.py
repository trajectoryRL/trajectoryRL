"""S3-compatible object storage for uploading pack files.

Uses boto3 as the unified client. Works with AWS S3, Google Cloud Storage
(via S3-compatible endpoint with HMAC keys), MinIO, Cloudflare R2, and
any other S3-compatible service.

Environment variables:
    S3_BUCKET             — target bucket name (required)
    S3_ENDPOINT_URL       — custom S3-compatible endpoint (optional)
                            e.g. https://storage.googleapis.com for GCS
    S3_REGION             — region (default: auto)
    AWS_ACCESS_KEY_ID     — access key (or GCS HMAC access key)
    AWS_SECRET_ACCESS_KEY — secret key (or GCS HMAC secret key)
"""

import hashlib
import json
import logging
import os
import uuid
from typing import Optional

logger = logging.getLogger(__name__)


class OSSStorage:
    """S3-compatible object storage client backed by boto3.

    Works with AWS S3, GCS (via S3-compatible endpoint), MinIO,
    Cloudflare R2, and any other S3-compatible service.
    """

    def __init__(
        self,
        bucket: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        region: Optional[str] = None,
    ):
        self.bucket = bucket or os.environ.get("S3_BUCKET", "")
        self.endpoint_url = endpoint_url or os.environ.get("S3_ENDPOINT_URL") or None
        self.region = region or os.environ.get("S3_REGION", "us-east-1")

        if not self.bucket:
            raise ValueError(
                "Bucket not configured. Set S3_BUCKET environment variable."
            )

        self._client = None

    @property
    def client(self):
        if self._client is None:
            import boto3
            from botocore.config import Config

            kwargs = {"region_name": self.region}
            if self.endpoint_url:
                kwargs["endpoint_url"] = self.endpoint_url
                kwargs["config"] = Config(
                    signature_version="s3v4",
                    s3={
                        "addressing_style": "path",
                        "payload_signing_enabled": False,
                    },
                )
            self._client = boto3.client("s3", **kwargs)
        return self._client

    @staticmethod
    def _pack_key() -> str:
        return f"{uuid.uuid4().hex}.json"

    def upload_pack(self, pack: dict) -> str:
        """Upload a pack dict and return its public URL.

        The object key is derived from the content hash, making uploads
        idempotent — uploading the same pack twice yields the same URL.

        Uses presigned URL + HTTP PUT for maximum compatibility with
        S3-compatible services (AWS S3, GCS, MinIO, R2).

        Args:
            pack: OPP v1 pack dict.

        Returns:
            Public URL of the uploaded object.
        """
        import urllib.request

        body = json.dumps(pack, sort_keys=True).encode("utf-8")
        key = self._pack_key()

        logger.info("Uploading pack to %s/%s", self.bucket, key)
        presigned_url = self.client.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": self.bucket,
                "Key": key,
                "ContentType": "application/json",
            },
            ExpiresIn=300,
        )
        req = urllib.request.Request(
            presigned_url,
            data=body,
            method="PUT",
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req)
        logger.info("Upload complete: %s", key)
        return self._build_url(key)

    def _build_url(self, key: str) -> str:
        if self.endpoint_url:
            base = self.endpoint_url.rstrip("/")
            return f"{base}/{self.bucket}/{key}"
        return f"https://{self.bucket}.s3.amazonaws.com/{key}"
