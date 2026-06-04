# AFTERIMAGE — Production Pipeline
## Free tool stack. Full workflow. Solo-operable.

---

## THE TOOL STACK (All Free)

| Tool | Role | Cost | Where |
|------|------|------|-------|
| iPhone (any) or any camera | Capture | $0 | You have it |
| DaVinci Resolve | Edit, color, grade | FREE | blackmagicdesign.com |
| Runway ML Gen-3 | AI video generation / enhancement | Free tier (limited) | runwayml.com |
| Pika Labs | AI video generation | Free tier | pika.art |
| CapCut | Mobile edit / quick overlay | Free | capcut.com |
| Suno | AI music generation | Free tier | suno.ai |
| ElevenLabs | Ambient voice / texture (optional) | Free tier | elevenlabs.io |
| Freesound.org | Field recordings, ambient audio | FREE | freesound.org |
| Canva | Static cover images / thumbnail | FREE | canva.com |
| Notion | Lore tracking, shot lists | FREE | notion.so |

**Total monthly cost: $0 (free tiers only)**

---

## PRODUCTION MODES

AFTERIMAGE content falls into 3 production modes. Each has different requirements.

---

### MODE 1: FILMED (Primary)

**What:** Real footage of real locations. iPhone or camera. Night. The Anchors.
**Effort:** 2-4 hours per video including edit
**Quality ceiling:** Highest — nothing replicates real wet streets and real light

**Workflow:**
```
1. SCOUT — identify the location (convenience store, overpass, etc.)
2. SHOOT — 15-30 minutes of footage, multiple angles, no direction needed
3. INGEST — import to DaVinci Resolve
4. SELECT — find Part 5 (compression moment) first, work backward
5. ASSEMBLE — build the 8-part structure, rough cut
6. GRADE — color work (see below)
7. AUDIO — layer ambient + music, sync Part 4
8. EXPORT — H.264, 9:16 for vertical, 1080p minimum
```

**DaVinci Resolve Color Grade — AFTERIMAGE Look:**
- Lift shadows to dark teal-gray (lift RGB: -0.03R, 0.01G, 0.02B)
- Crush highlights to warm amber (gain RGB: 1.0R, 0.95G, 0.85B)
- Reduce saturation to 75-80%
- Re-boost cyan, blue, and magenta selectively in HSL qualifier
- Add light grain (noise overlay, 15-20% opacity)
- Slight glow on practical light sources (neon, convenience store)
- This creates the Entergalactic "painterly" effect through color rather than illustration

Save this as a DaVinci Power Grade — apply to every video. This is your visual consistency lock.

---

### MODE 2: AI-GENERATED (Supplement)

**What:** Runway ML or Pika generates footage from text prompt or image
**Effort:** 30-60 minutes per video
**Quality ceiling:** Lower, but useful for establishing shots, impossible angles, pure atmosphere

**Prompt structure for Runway/Pika:**
```
Base: "rain-soaked city street at night, neon reflections, cinematic, analog film grain, lonely atmosphere"
Add: [the specific Anchor description]
Add: "no people visible" OR "figure from behind, small in frame"
Style: "like a short film, not a music video, naturalistic"
Avoid: "dramatic, fantasy, supernatural, bright colors"
```

**How to use AI video:**
- For establishing shots of Anchors the audience hasn't seen yet
- For the city itself as character — aerial, movement through streets
- Never for protagonist footage — that needs to be filmed, it needs weight
- Layer AI footage as background behind filmed footage using DaVinci composite

---

### MODE 3: STATIC + MOTION (Fastest)

**What:** A single still image or very slow dolly + audio layer + text overlay (minimal)
**Effort:** 20-30 minutes
**Quality ceiling:** Limited but useful for specific content types (Fragment videos, Signal appearances)

**When to use:**
- Fragment videos (5-8 seconds, Parts 5-8 only)
- Symbol appearances — the cassette isolated against a surface, no protagonist
- Pure atmosphere seeding — the 23rd floor window, exterior, no movement, ambient audio
- Lore drop moments — the handwriting, alone in frame, slowly coming into focus

---

## THE CONSISTENCY SYSTEM (100+ Videos Without Drift)

### The Visual Lock

Three things must be consistent across every video:
1. **The Color Grade** — the DaVinci Power Grade applied to all footage
2. **The Sound Texture** — the ambient city layer (low, distant, wet) under all audio
3. **The Protagonist Silhouette Rule** — from behind, small in frame, same palette

These three create recognizability even when content varies wildly.

### The Anchor Map

Maintain a Notion database with one row per Anchor:
- Anchor name
- Real-world location used (for continuity in filming)
- Colors used (so the warm amber of the 23rd floor stays consistent)
- Last appeared in which video
- Next planned appearance

Every 4th video must feature an Anchor (not a new location). This is the homecoming mechanism.

### The Symbol Log

Notion database, one row per symbol:
- Symbol name
- Every video it appears in
- State it appears in (found / held / left behind / referenced)
- Whether its meaning has shifted

Before every new video: check what hasn't appeared recently and plant one thing.

### The Canon Doc

The lore bible is the source of truth. Before posting any video:
- Does this contradict anything established?
- Does this seed anything the canon doc supports?
- Does this advance or hold the lore?

A video that contradicts established canon (even subtly) breaks the obsessed viewer's trust. That audience does not forgive continuity errors.

---

## BATCH PRODUCTION MODEL

**Monthly cadence:**
- 1 filming session per month (3-4 hours, at one or two Anchors)
- Produces: 8-15 clips of raw footage
- From those clips: 4-6 final videos (rest held in reserve)
- Supplement with 2-4 AI-generated / Mode 3 videos
- Total output: 6-10 videos per month (enough for 2x/week posting)

**The reserve rule:** Always have 4 finished videos in reserve before posting the newest. This prevents posting under pressure, which breaks consistency.

---

## THE ENTERGALACTIC TECHNIQUE (Applied)

Entergalactic made the city feel inhabited through:
- Strict color scripts per location
- Environmental animation tied to emotion (the city's lights respond to the character's state)
- 3D city repainted to feel handmade, not rendered

AFTERIMAGE achieves the same without animation:
- **Color scripts per Anchor** (23rd floor = warm amber. Overpass = cold blue-gray. Convenience store = fluorescent green-white. Consistent always.)
- **Weather as emotional indicator** (rain = Phase 2. Dry streets = Phase 1. Fog = Phase 3.)
- **Practical light source as character** (the neon sign that flickers in certain videos. The window that has light sometimes and doesn't in others.)

The city must feel like it has moods. The colors create the moods. Consistency makes the moods legible over time.
