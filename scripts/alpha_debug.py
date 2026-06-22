#!/usr/bin/env python3
"""
NexFlow Alpha — API debug script.
Run this to see exactly what DexScreener returns and diagnose issues.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

_TIMEOUT = 10
_HEADERS = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}

ENDPOINTS = [
    ("Token Profiles (new tokens)",  "https://api.dexscreener.com/token-profiles/latest/v1"),
    ("Token Boosts (promoted)",      "https://api.dexscreener.com/token-boosts/latest/v1"),
    ("Search: ETH",                  "https://api.dexscreener.com/latest/dex/search?q=ETH"),
    ("Pairs: ETH/USDC (known pair)", "https://api.dexscreener.com/latest/dex/pairs/ethereum/0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"),
]

def test(label: str, url: str) -> None:
    print(f"\n{'─'*60}")
    print(f"TEST: {label}")
    print(f"URL:  {url}")
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            status = resp.status
            raw    = resp.read()
            print(f"HTTP: {status}  |  bytes: {len(raw)}")
            data = json.loads(raw)
            if isinstance(data, list):
                print(f"Response: list of {len(data)} items")
                if data:
                    print(f"First item keys: {list(data[0].keys())}")
                    print(f"First item sample: {json.dumps(data[0], indent=2)[:400]}")
            elif isinstance(data, dict):
                print(f"Response: dict with keys: {list(data.keys())}")
                for k, v in data.items():
                    if isinstance(v, list):
                        print(f"  {k}: list of {len(v)} items")
                        if v:
                            print(f"    first item keys: {list(v[0].keys()) if isinstance(v[0], dict) else type(v[0])}")
                    else:
                        print(f"  {k}: {repr(v)[:100]}")
    except urllib.error.HTTPError as e:
        print(f"HTTP ERROR: {e.code} {e.reason}")
    except urllib.error.URLError as e:
        print(f"URL ERROR: {e.reason}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")

if __name__ == "__main__":
    print("NexFlow Alpha — DexScreener API Diagnostics")
    print("=" * 60)
    for label, url in ENDPOINTS:
        test(label, url)
    print(f"\n{'='*60}")
    print("Done.")
