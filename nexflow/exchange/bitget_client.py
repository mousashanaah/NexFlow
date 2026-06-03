"""Authenticated Bitget REST API client for USDT-Perpetual Futures.

Handles:
  - HMAC-SHA256 request signing
  - paptrading: 1 header for Bitget Demo Trading
  - Automatic retry on transient network errors (5xx, timeout)
  - JSON response validation (code == "00000")

Credentials are loaded from environment variables:
  BITGET_API_KEY
  BITGET_API_SECRET
  BITGET_PASSPHRASE
  BITGET_PAPER          set to "1" (or any truthy value) for demo trading

Never hard-code credentials. Load via .env or shell environment only.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

_BASE_URL = "https://api.bitget.com"
_TIMEOUT  = 20
_MAX_RETRY = 4


class BitgetAPIError(Exception):
    """Raised when the Bitget API returns a non-success code."""
    def __init__(self, code: str, msg: str, endpoint: str) -> None:
        super().__init__(f"Bitget API error {code}: {msg}  (endpoint: {endpoint})")
        self.code = code
        self.msg  = msg


class BitgetClient:
    """Thin authenticated REST client for Bitget USDT-Perp Futures.

    Usage:
        client = BitgetClient.from_env()   # reads BITGET_* env vars
        data   = client.get("/api/v2/mix/account/account", {"symbol": "BTCUSDT", ...})
        result = client.post("/api/v2/mix/order/place-order", body_dict)
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
        paper: bool = False,
    ) -> None:
        if not all([api_key, api_secret, passphrase]):
            raise ValueError("api_key, api_secret, and passphrase are all required")
        self._key        = api_key
        self._secret     = api_secret.encode()
        self._passphrase = passphrase
        self._paper      = paper

    @classmethod
    def from_env(cls) -> "BitgetClient":
        """Create a client from BITGET_* environment variables.

        Required env vars:
          BITGET_API_KEY, BITGET_API_SECRET, BITGET_PASSPHRASE
        Optional:
          BITGET_PAPER=1   — enable demo trading mode
        """
        key        = os.environ.get("BITGET_API_KEY", "")
        secret     = os.environ.get("BITGET_API_SECRET", "")
        passphrase = os.environ.get("BITGET_PASSPHRASE", "")
        paper_raw  = os.environ.get("BITGET_PAPER", "0")
        paper      = paper_raw.strip() not in ("0", "", "false", "False")
        return cls(key, secret, passphrase, paper=paper)

    # ── Signing ──────────────────────────────────────────────────────────

    def _sign(self, timestamp: str, method: str, path_with_query: str, body: str) -> str:
        """Return base64 HMAC-SHA256 signature per Bitget v2 spec."""
        prehash = timestamp + method.upper() + path_with_query + body
        digest  = hmac.new(self._secret, prehash.encode(), hashlib.sha256).digest()
        return base64.b64encode(digest).decode()

    def _headers(self, method: str, path_with_query: str, body: str) -> dict:
        ts  = str(int(time.time() * 1000))
        sig = self._sign(ts, method, path_with_query, body)
        h = {
            "Content-Type":    "application/json",
            "ACCESS-KEY":       self._key,
            "ACCESS-SIGN":      sig,
            "ACCESS-TIMESTAMP": ts,
            "ACCESS-PASSPHRASE": self._passphrase,
            "User-Agent":       "NexFlow/1.0",
        }
        if self._paper:
            h["paptrading"] = "1"
        return h

    # ── HTTP helpers ─────────────────────────────────────────────────────

    def _request(self, method: str, endpoint: str, params: dict | None, body: dict | None) -> Any:
        path = endpoint
        qs   = ""
        if params:
            qs   = "?" + urllib.parse.urlencode(sorted(params.items()))
            path = endpoint + qs

        body_str = json.dumps(body, separators=(",", ":")) if body else ""
        headers  = self._headers(method, path, body_str)
        url      = _BASE_URL + path

        req = urllib.request.Request(
            url,
            data    = body_str.encode() if body_str else None,
            headers = headers,
            method  = method,
        )

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRY):
            try:
                with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                    raw = json.loads(resp.read())
                code = raw.get("code", "")
                if code != "00000":
                    raise BitgetAPIError(code, raw.get("msg", ""), endpoint)
                return raw.get("data")
            except BitgetAPIError:
                raise
            except urllib.error.HTTPError as exc:
                last_exc = exc
                if exc.code < 500:
                    try:
                        body = exc.read().decode("utf-8", errors="replace")
                        raise urllib.error.HTTPError(
                            exc.url, exc.code,
                            f"{exc.reason} — {body}",
                            exc.headers, None,
                        )
                    except (AttributeError, OSError):
                        pass
                    raise
            except (urllib.error.URLError, OSError) as exc:
                last_exc = exc
            backoff = 2 ** attempt
            time.sleep(backoff)

        raise RuntimeError(f"Request failed after {_MAX_RETRY} retries: {last_exc}")

    def get(self, endpoint: str, params: dict | None = None) -> Any:
        return self._request("GET", endpoint, params, None)

    def post(self, endpoint: str, body: dict) -> Any:
        return self._request("POST", endpoint, None, body)
