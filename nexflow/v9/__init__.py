"""
V9 Confidence — production implementation.

Single source of truth for all signals, scoring, and allocation logic.
The research scripts in scripts/ validate against this package.
Any discrepancy between this package and research outputs is a bug.

Build order:
  Module 1 — state.py       (bear regime persistence, startup gate)
  Module 2 — replay.py      (nightly research/production parity check)
  Module 3 — data.py        (candle ingestion, to be built)
  Module 4 — core.py        (signals, scoring, allocation — this is the law)
  Module 5 — runner.py      (daily orchestration, to be built)
"""
