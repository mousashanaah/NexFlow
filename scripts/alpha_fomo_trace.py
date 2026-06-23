#!/usr/bin/env python3
"""
NexFlow Alpha — FOMO/Jupiter end-to-end TRACE

This is a diagnostic, not an integration. It traces a SINGLE token from
mint address → Jupiter request → raw response → final fomo_available,
printing every intermediate value and the exact exception if one occurs.

It tries EVERY known Jupiter endpoint so we can see which (if any) is live.

Usage:
  python scripts/alpha_fomo_trace.py                      # traces RDR2, MYLOO, CATWIF from DB
  python scripts/alpha_fomo_trace.py --mint <address>     # trace a specific mint
  python scripts/alpha_fomo_trace.py --symbol RDR2        # trace by symbol from DB

Reports:
  1. The token mint address used
  2. The raw Jupiter request URL (per endpoint)
  3. The raw HTTP status + response body
  4. The exact exception/traceback if thrown
  5. The conversion from response → (tradable, reason)
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import traceback
import urllib.error
import urllib.request
from pathlib import Path

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

_DB_PATH = Path(os.environ.get("NEXFLOW_ALPHA_DB", "/var/nexflow/alpha.db"))

_SOL_MINT = "So11111111111111111111111111111111111111112"
_AMOUNT   = "1000000"   # 0.001 SOL in lamports
_TIMEOUT  = 15
_HEADERS  = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}

# Every known Jupiter quote endpoint, newest first.
_ENDPOINTS = [
    ("lite-api v1 (current free)", "https://lite-api.jup.ag/swap/v1/quote"),
    ("api v1 (paid/keyed)",        "https://api.jup.ag/swap/v1/quote"),
    ("quote-api v6 (legacy)",      "https://quote-api.jup.ag/v6/quote"),
]

# A known-good reference token (BONK) to prove the endpoint works at all,
# independent of whether the user's discovered tokens are tradable.
_REFERENCE = ("BONK", "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263")


def _build_url(base: str, out_mint: str) -> str:
    return (
        f"{base}"
        f"?inputMint={_SOL_MINT}"
        f"&outputMint={out_mint}"
        f"&amount={_AMOUNT}"
        f"&slippageBps=5000"
    )


def trace_one(symbol: str, mint: str) -> None:
    print(f"\n{'═'*72}")
    print(f"TRACE: {symbol}")
    print(f"  mint address: {mint}")
    print(f"  mint length:  {len(mint)} chars (Solana mints are typically 43-44)")
    print(f"  input mint:   {_SOL_MINT} (SOL)")
    print(f"  amount:       {_AMOUNT} lamports (0.001 SOL)")
    print(f"{'═'*72}")

    for label, base in _ENDPOINTS:
        url = _build_url(base, mint)
        print(f"\n  ── Endpoint: {label}")
        print(f"  REQUEST URL:\n    {url}")

        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                status   = resp.status
                raw      = resp.read().decode()
                print(f"  HTTP STATUS: {status}")
                print(f"  RAW BODY (first 600 chars):")
                print(f"    {raw[:600]}")

                # Conversion path → fomo_available
                try:
                    data = json.loads(raw)
                except Exception as je:
                    print(f"  PARSE ERROR: {je}")
                    continue

                # v6 uses 'outAmount' at top level; v1 may nest differently
                out = data.get("outAmount")
                if out is None and isinstance(data.get("data"), dict):
                    out = data["data"].get("outAmount")
                print(f"  PARSED outAmount: {out!r}")

                if out and int(out) > 0:
                    print(f"  ==> RESULT: TRADABLE (outAmount={out})")
                else:
                    print(f"  ==> RESULT: no_route (no positive outAmount)")

        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode()[:400]
            except Exception:
                pass
            print(f"  HTTPError: code={e.code} reason={e.reason}")
            print(f"  ERROR BODY: {body}")
            print(f"  ==> RESULT: {'no_route' if e.code in (400,404) else f'http_{e.code}'}")

        except urllib.error.URLError as e:
            print(f"  URLError: reason={e.reason}")
            print(f"  ==> RESULT: error (network/DNS — endpoint may be unreachable)")
            print(f"  TRACEBACK:")
            traceback.print_exc()

        except Exception as e:
            print(f"  UNEXPECTED {type(e).__name__}: {e}")
            print(f"  ==> RESULT: error")
            print(f"  TRACEBACK:")
            traceback.print_exc()


def _lookup_from_db(symbols: list[str]) -> list[tuple[str, str]]:
    if not _DB_PATH.exists():
        print(f"DB not found at {_DB_PATH} — cannot look up symbols.")
        return []
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    out = []
    for sym in symbols:
        row = conn.execute(
            "SELECT token_symbol, token_address FROM pools "
            "WHERE UPPER(token_symbol)=? AND chain_id='solana' LIMIT 1",
            (sym.upper(),)
        ).fetchone()
        if row:
            out.append((row["token_symbol"], row["token_address"]))
        else:
            print(f"  (symbol {sym} not found in DB)")
    conn.close()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="FOMO/Jupiter end-to-end trace")
    parser.add_argument("--mint",   default=None, help="Trace a specific mint address")
    parser.add_argument("--symbol", default=None, help="Trace a symbol from the DB")
    args = parser.parse_args()

    print("NexFlow Alpha — FOMO/Jupiter TRACE")
    print("Testing reference token first to prove endpoint liveness...")

    # Always trace BONK reference first — if this fails, the endpoint is the problem
    trace_one(*_REFERENCE)

    if args.mint:
        trace_one("CUSTOM", args.mint)
    elif args.symbol:
        pairs = _lookup_from_db([args.symbol])
        for sym, mint in pairs:
            trace_one(sym, mint)
    else:
        # Default: the user's three named tokens
        pairs = _lookup_from_db(["RDR2", "MYLOO", "CATWIF"])
        if not pairs:
            print("\nNo target tokens found in DB. Showing reference trace only.")
        for sym, mint in pairs:
            trace_one(sym, mint)

    print(f"\n{'═'*72}")
    print("INTERPRETATION:")
    print("  - If BONK reference is TRADABLE on an endpoint, that endpoint is live")
    print("    and the integration should use it.")
    print("  - If BONK fails on ALL endpoints with URLError, it's network/geo-blocking.")
    print("  - If BONK works but your tokens show no_route, they genuinely lack")
    print("    Jupiter liquidity (not tradable on FOMO yet).")
    print(f"{'═'*72}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
