# LIFE-OS — Operating System for Mousa Shanaah

This is not a journal. It is a system. It has inputs, feedback loops, scoreboards,
and an edge that nobody else has the discipline to maintain. You built one of these
to trade markets. This one trades you toward the Asian Tour card and the life behind it.

## Prime Directive
Win Q-School Stage 1 and Final Stage outright, by a margin, and earn the best card
available. Everything in this system is downstream of that single objective and the
man it requires you to become.

## Operating Principles (read before every season block)
1. **Agency over hypnosis.** No one programs you. You program you. The coach makes
   himself less necessary over time, not more.
2. **Quality of input > hours of input.** High mileage at low intent is comfortable
   suffering. We measure intentional, uncomfortable, targeted work — not time on site.
3. **One thing per session.** You know too much about the swing. The constraint is the skill.
4. **Cycles, not machines.** Savage work, then ruthless recovery — on purpose. The
   adaptation happens in the rest. A machine that never stops is just injured early.
5. **Cold fuel beats hot fuel.** Rage at being doubted is rocket fuel — it burns out.
   We convert it into a colder, permanent standard that doesn't need an audience.
6. **Log or it didn't happen.** No narrative. Numbers. The ledger is the truth.

## Files
- `00-DIAGNOSIS.md` — the honest profile. Your wound, your lie, your leverage.
- `01-QSCHOOL-MODEL.md` — reverse-engineered target tree from Q-School backward.
- `02-PRACTICE-ARCHITECTURE.md` — structured practice + the two-way-miss isolation protocol.
- `03-DAILY-PROTOCOL.md` — the dream-day template and the discipline ledger spec.
- `logs/` — daily ledgers. One file per day. This is where the system lives or dies.

## Files (cont.)
- `WEEK.md` — this week's contract: concrete targets, non-negotiables, stakes. Rewritten Sundays.
- `04-SWING-ANALYSIS.md` — confirmed swing diagnosis and fix order.
- `scoreboard.py` — the dashboard. Parses your logs into streak / discipline trend / miss trend.

## How to run it (the daily loop)
- **Morning:** `python3 life-os/scoreboard.py new` → set today's ONE objective.
- **Night:** check the habits, fill the numbers, write ONE line. `git add` + `git commit`.
- **Anytime:** `python3 life-os/scoreboard.py` → see your streak and trends.
- **Sunday:** run the dashboard, do the review in `WEEK.md`, write next week's contract.

The streak is the number you protect. You don't break the chain. The git history is the
permanent, un-fudgeable record — when the coach reviews you, he reads your commits.
