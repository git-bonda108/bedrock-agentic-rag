"""API key authentication. The key lives in AWS Secrets Manager and is cached in memory
for a few minutes so rotations take effect without a redeploy."""

from __future__ import annotations

import hmac
import time
from typing import Any

_CACHE_SECONDS = 300


class ApiKeyVerifier:
    def __init__(
        self,
        static_key: str | None,
        secret_arn: str | None,
        secrets_client: Any | None = None,
    ) -> None:
        self._static_key = static_key
        self._secret_arn = secret_arn
        self._client = secrets_client
        self._cached: str | None = None
        self._fetched_at = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self._static_key or self._secret_arn)

    def _expected(self) -> str | None:
        if self._static_key:
            return self._static_key
        if not self._secret_arn:
            return None
        if self._cached is None or time.monotonic() - self._fetched_at > _CACHE_SECONDS:
            if self._client is None:
                import boto3

                self._client = boto3.client("secretsmanager")
            self._cached = self._client.get_secret_value(SecretId=self._secret_arn)["SecretString"]
            self._fetched_at = time.monotonic()
        return self._cached

    def verify(self, provided: str | None) -> bool:
        if not self.enabled:
            return True
        expected = self._expected()
        if not provided or not expected:
            return False
        return hmac.compare_digest(provided.encode(), expected.encode())
