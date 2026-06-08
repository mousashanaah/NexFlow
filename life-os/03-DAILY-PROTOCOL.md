# 03 — DAILY PROTOCOL

Your dream day is the right day. This file makes it the DEFAULT day, protects it from the
two things that blow it up (household cortisol + snacking willpower-drain), and — most
importantly — makes every day a logged, scored data point. No log = the day didn't happen.

## The Default Day (sequenced for performance)
1. **Wake.** Before anything: `python3 life-os/scoreboard.py new` → set today's ONE objective.
2. **Short game + putting** (skill when fresh, nerves calm).
3. **Gym — dynamic warmup / activation** (prep, not fatigue).
4. **Range** — the day's ONE objective, with tape + video. Exit on the condition, not the clock.
5. **Course — 9 to 18 holes**, carrying only the day's objective as a swing thought.
6. **Gym — real lifting** (AFTER golf, so it never steals from skill quality).
7. **NexFlow work** + **Quran / reading** (the cold, non-golf identity anchor).
8. **Call gf, wind down, in bed early.** Sleep is a training input, not leftover time.

### Two non-negotiable guards
- **State reset before performance.** Before course/range: 5 slow nasal breaths, longer
  exhale than inhale, drop the shoulders. Brings the nervous system down so your real swing shows up.
- **The house is not your nervous system's problem.** Parents fighting = noise, not signal.
  Headphones, leave early, protect the morning. Performance management, not coldness.

## The Snacking Lock (environment, not willpower)
Blowups happen on high-depletion days (poor sleep, stress, long sessions) — exactly when
willpower is lowest. So we remove the decision: don't keep trigger foods in the house/bag;
pre-pack tomorrow's food the night before; fuel enough around training to avoid the crash;
one *scheduled* treat slot kills the binge cycle.

---

## THE DAILY LOG (this is the system's heartbeat)
Every day: `python3 life-os/scoreboard.py new` creates `logs/YYYY-MM-DD.md` from the
template below. Fill the morning block at wake, the rest at night. Then commit it.
Format is fixed because `scoreboard.py` parses it. Do not rename the labels.

```
# YYYY-MM-DD

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
```

## Scoring (automated — see scoreboard.py)
- **Discipline score** = checked habits / total habits, per day.
- **GREEN day** = discipline >= 80% AND all (NN) non-negotiables met. Anything else = RED.
- **Streak** = consecutive GREEN days. The streak is the number you protect. You don't break the chain.

## Weekly review (with the coach)
Run `python3 life-os/scoreboard.py` to print the dashboard, then we read the week together:
discipline trend, catastrophic-miss trend, streak, and whether each day's objective advanced
the Q-School tree. Trend beats any single day. We coach the line, not the dot.
