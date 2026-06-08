#!/usr/bin/env python3
"""
life-os scoreboard — turns your daily logs into a tracked system with a dashboard.
Same architecture you built NexFlow on: logged inputs, automated scoring, a streak you can't fake.

Usage:
    python3 life-os/scoreboard.py            # print the dashboard
    python3 life-os/scoreboard.py new        # scaffold today's log from the template
    python3 life-os/scoreboard.py new 2026-06-10   # scaffold a specific date

No dependencies. Stdlib only.
"""
import os
import re
import sys
from datetime import date, datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "logs")

GREEN_THRESHOLD = 0.80  # >=80% habits AND all non-negotiables = GREEN day

TEMPLATE = """# {d}

## MORNING
- Objective: <the ONE practice objective for today>
- Round today: no

## HABITS  (mark [x] when done as designed; (NN) = non-negotiable)
- [ ] (NN) Trained golf today
- [ ] (NN) Logged this day honestly
- [ ] (NN) ONE objective set AND executed with external feedback (tape/video)
- [ ] Short game + putting
- [ ] Range on-objective (no chasing)
- [ ] Course / playing reps
- [ ] Lifting session
- [ ] NexFlow work
- [ ] Quran / reading
- [ ] Nutrition held (no unplanned snack blowup)
- [ ] Sleep protected / in bed on time
- [ ] State reset used before performance

## NUMBERS
- Catastrophic misses:
- Stock-shot command /10:
- Sleep hours:
- Weight:

## ONE LINE
>
"""

HABIT_RE = re.compile(r"^- \[([ xX])\]\s*(.*)$")
NUM_RE = lambda label: re.compile(r"^- " + re.escape(label) + r":\s*(.*)$")


def parse_log(path):
    checked = total = 0
    nn_total = nn_met = 0
    nums = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            m = HABIT_RE.match(line)
            if m:
                is_checked = m.group(1).lower() == "x"
                label = m.group(2)
                total += 1
                if is_checked:
                    checked += 1
                if "(NN)" in label:
                    nn_total += 1
                    if is_checked:
                        nn_met += 1
                continue
            for key in ("Catastrophic misses", "Stock-shot command /10", "Sleep hours", "Weight"):
                mm = NUM_RE(key).match(line)
                if mm:
                    val = mm.group(1).strip()
                    if val:
                        nums[key] = val
    score = (checked / total) if total else 0.0
    nn_ok = (nn_met == nn_total) and nn_total > 0
    green = score >= GREEN_THRESHOLD and nn_ok
    return {
        "checked": checked, "total": total, "score": score,
        "nn_met": nn_met, "nn_total": nn_total, "green": green, "nums": nums,
    }


def load_all():
    out = {}
    if not os.path.isdir(LOGS):
        return out
    for fn in os.listdir(LOGS):
        m = re.match(r"(\d{4}-\d{2}-\d{2})\.md$", fn)
        if not m:
            continue
        try:
            d = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except ValueError:
            continue
        out[d] = parse_log(os.path.join(LOGS, fn))
    return out


def current_streak(data):
    """Consecutive GREEN days ending at the most recent logged day (no gaps)."""
    if not data:
        return 0
    days = sorted(data.keys(), reverse=True)
    streak = 0
    cursor = days[0]
    for d in days:
        if d != cursor:
            break  # a gap in logging breaks the chain
        if data[d]["green"]:
            streak += 1
            cursor = d - timedelta(days=1)
        else:
            break
    return streak


def longest_streak(data):
    best = run = 0
    prev = None
    for d in sorted(data.keys()):
        if prev is not None and d == prev + timedelta(days=1) and data[d]["green"]:
            run = run + 1 if data.get(prev, {}).get("green") else 1
        else:
            run = 1 if data[d]["green"] else 0
        best = max(best, run)
        prev = d
    return best


def bar(frac, width=20):
    filled = int(round(frac * width))
    return "█" * filled + "·" * (width - filled)


def cmd_new(argv):
    d = argv[0] if argv else date.today().isoformat()
    os.makedirs(LOGS, exist_ok=True)
    path = os.path.join(LOGS, f"{d}.md")
    if os.path.exists(path):
        print(f"Log already exists: {path}")
        return
    with open(path, "w") as f:
        f.write(TEMPLATE.format(d=d))
    print(f"Created {path}\nSet your ONE objective now. Fill the rest tonight. Then commit it.")


def cmd_dashboard():
    data = load_all()
    print("=" * 56)
    print("  LIFE-OS SCOREBOARD — the chain you don't break")
    print("=" * 56)
    if not data:
        print("\nNo logs yet. Start the chain:  python3 life-os/scoreboard.py new\n")
        return
    cs, ls = current_streak(data), longest_streak(data)
    print(f"\n  CURRENT STREAK : {cs} green day(s)")
    print(f"  LONGEST STREAK : {ls} green day(s)")

    last7 = [data[d] for d in sorted(data)[-7:]]
    if last7:
        avg = sum(x["score"] for x in last7) / len(last7)
        print(f"\n  Last {len(last7)} days discipline avg: {avg*100:4.0f}%  {bar(avg)}")

    misses = [(d, data[d]["nums"].get("Catastrophic misses"))
              for d in sorted(data) if data[d]["nums"].get("Catastrophic misses")]
    if misses:
        print("\n  CATASTROPHIC MISSES (north star → drive to ZERO):")
        for d, v in misses[-7:]:
            print(f"    {d}: {v}")

    print("\n  RECENT DAYS:")
    for d in sorted(data)[-7:]:
        x = data[d]
        tag = "GREEN" if x["green"] else "RED  "
        nn = f"NN {x['nn_met']}/{x['nn_total']}"
        print(f"    {d}  [{tag}] {x['checked']}/{x['total']} habits  {nn}  {bar(x['score'], 12)}")
    print()


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "new":
        cmd_new(argv[1:])
    else:
        cmd_dashboard()


if __name__ == "__main__":
    main()
