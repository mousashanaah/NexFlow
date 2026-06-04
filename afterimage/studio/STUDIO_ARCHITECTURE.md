# AFTERIMAGE STUDIO — Architecture Document
## Not a channel. A studio. Automated, internally consistent, owner-operated.

---

## CRITICAL EVALUATION FIRST

Before the architecture: an honest challenge to the current design.

---

### THE LORE PROBLEM

**Kershaw's hierarchy:** Emotion → Character → Mystery → Lore

**Current AFTERIMAGE design:** Heavy lore infrastructure. Residuals, Phase 1/2/3, Event Zero, 6 Anchors, 5 symbols, the handwriting decoding mechanic, a canonical internal timeline.

**The honest assessment: the current design drifted.**

Here is the failure mode: lore-heavy short-form content becomes ARGs. ARGs attract a specific and small audience — puzzle-solvers. Kershaw's audience is not puzzle-solvers. It is people who want to feel something at 1am.

The difference between **feeling** and **figuring out** is the difference between 10M followers and 50K.

**What actually drives Kershaw's numbers:**
- The same emotional moment, recurring, refined
- Visual language so consistent it feels like a single long film
- A figure you project onto — not a character you track
- The cut before resolution — creating the ache, not the mystery

He doesn't have a lore bible. He has a visual bible and an emotional bible. The audience doesn't theorize about *what happened* — they feel the thing that happened, from their own life, through the frame he provides.

**The ARG problem for automation:** Lore that must be internally consistent is a manual process. Every generated video must be checked against the canon. One contradiction by an LLM breaks the obsessed viewer's trust. The more complex the lore, the more manual oversight required — which defeats the studio objective.

---

### THE REDESIGN: EMOTION-FIRST ARCHITECTURE

**Keep from current design:**
- The unnamed city (visual, not narrative)
- The protagonist silhouette (projection surface)
- The cassette (one symbol, not five — the single tangible mystery object)
- The Other One (implied presence — absence, not mythology)
- The 6 Anchors as visual locations (not as narrative engines)
- The color grade system

**Strip:**
- The Residuals mythology
- The Phase 1/2/3 system
- Event Zero as a narrative thing to decode
- The handwriting decoding mechanic
- The 23rd Floor as a story engine
- The canonical timeline

**Why:** You cannot automate lore consistency without a human in the loop. You CAN automate emotional consistency, visual consistency, and symbol placement without a human in the loop.

**The new design principle:**
> Every video produces a feeling. The feeling is always the same feeling, expressed differently. The world coheres visually. The mystery is: who is she? The answer is: whoever you need her to be.

That's automatable. The ARG was not.

---

## THE STUDIO PIPELINE

```
Universe Bible (Claude context)
        │
        ▼
  Episode Generator ─────────────────────────┐
  [Claude: generate 1 episode concept]        │
        │                                     │
        ▼                                     │
  Shot Generator                              │
  [Claude: break into 8-part moment]          │
        │                                     │
        ▼                                     │
  Visual Prompt Generator                     │
  [Claude: Runway/Pika prompts per shot]       │
        │                                     │
        ▼                                     ▼
  Video Prompt Generator              Lore DB Update
  [Claude: full video brief]          [Claude: log what
        │                              this seeds/shows]
        ▼
  Caption Generator
  [Claude: fragment caption + 3 alts]
        │
        ▼
  Hashtag Generator
  [Claude: 3 contextual hashtags]
        │
        ▼
  Upload Package
  [folder: prompts/ captions/ hashtags/ brief/]
```

Each stage is a single Claude API call. Structured input → structured JSON output. No human decisions required until final review.

---

## CLAUDE AS SOURCE OF TRUTH

Claude does not browse the lore bible each time. Claude **is** the lore bible via a structured system prompt.

### The Master Context Document

A single file: `studio/UNIVERSE_CONTEXT.md`

This file is the entire universe compressed into ~2,000 tokens. It loads as the system prompt for every pipeline call. It contains:

1. **Visual identity** (color, lighting, palette per Anchor — exact, precise)
2. **Protagonist rules** (face never shown, body from behind, hands = only detailed element)
3. **Emotional register** (the target feeling — not the narrative)
4. **The symbol** (the cassette only — its appearance rules, its states: held / found / left / missing)
5. **The Other One** (rules for implying presence through absence)
6. **Anchor descriptions** (6 locations — visual description only, no narrative function)
7. **What is always withheld** (the 3 things the universe never answers)
8. **Output format spec** (the exact JSON schema each stage must produce)

When this file is the system prompt, every Claude output is automatically universe-consistent because the universe is defined in the context window.

### The Constraint

Every pipeline prompt ends with:
```
You must not introduce: names, explicit backstory, new symbols beyond the cassette, 
any resolution of the central mystery, narrative explanation. 
Output must be emotionally consistent with the target feeling: [FEELING_TAG].
```

This is the automation lock. Claude cannot drift because it cannot output what is prohibited.

---

## THE 7-STAGE PIPELINE — EXACT PROMPTS

### STAGE 1: EPISODE GENERATOR

**Input:**
```json
{
  "feeling_tag": "the weight of a place you can't return to",
  "anchor": "Overpass",
  "symbol_state": "held",
  "last_5_episodes": ["...", "...", "...", "...", "..."]
}
```

**Prompt to Claude:**
```
[UNIVERSE_CONTEXT loaded as system prompt]

Generate one AFTERIMAGE episode concept.

Constraints:
- Anchor: {anchor}
- Symbol state for the cassette: {symbol_state}  
- Target feeling: {feeling_tag}
- Must not repeat the emotional beat or visual setup of any of these recent episodes: {last_5_episodes}
- No narrative explanation. No names. No resolution.

Output as JSON:
{
  "episode_id": "EP-[number]",
  "anchor": "...",
  "feeling_tag": "...",
  "core_moment": "one sentence — the single thing that happens",
  "what_is_withheld": "one sentence — what the video refuses to show",
  "symbol_state": "held|found|left|missing|absent",
  "other_one_presence": "none|implied|trace",
  "emotional_arc": "what the viewer feels at second 0 vs second 15"
}
```

**Output:** A complete episode concept in ~150 tokens. No human needed.

---

### STAGE 2: SHOT GENERATOR

**Input:** Episode JSON from Stage 1

**Prompt to Claude:**
```
[UNIVERSE_CONTEXT loaded as system prompt]

Break this episode concept into the 8-part AFTERIMAGE shot structure.

Episode: {episode_json}

The 8 parts:
1. Establishing Texture (0:00-0:02) — surface, materiality, no action
2. Figure in Negative Space (0:02-0:04) — protagonist revealed, small in frame
3. Movement Initiation (0:04-0:05) — one small action
4. Tempo-Visual Coupling (0:05-0:07) — the sync moment
5. Compression Moment (0:07-0:08) — close-up, world disappears
6. Emotional Peak Image (0:08-0:09) — the screenshot frame
7. Release (0:09-0:10) — the cut before resolution
8. Audio Decay (0:09-0:12) — sound continues past visual

Output as JSON array of 8 shots:
[{
  "part": 1,
  "duration_seconds": 2,
  "shot_description": "...",
  "camera_position": "...",
  "what_is_visible": "...",
  "what_is_NOT_visible": "...",
  "audio_note": "..."
}]
```

---

### STAGE 3: VISUAL PROMPT GENERATOR

**Input:** Shot array from Stage 2

**Prompt to Claude:**
```
[UNIVERSE_CONTEXT loaded as system prompt]

Convert each shot into a Runway ML Gen-3 / Pika video generation prompt.

Shots: {shot_array}

For each shot, generate:
- A Runway/Pika text prompt (max 200 characters)
- A negative prompt (what to exclude)
- Style reference tags

AFTERIMAGE visual rules to enforce in every prompt:
- Color: "dark teal shadows, warm amber highlights, muted saturation, analog grain"
- Lighting: "practical light sources only, neon reflections, wet surfaces"
- Camera: "cinematic, shallow depth of field, naturalistic movement"
- Avoid in every prompt: "fantasy, bright colors, dramatic, supernatural, faces"

Output as JSON:
[{
  "part": 1,
  "runway_prompt": "...",
  "negative_prompt": "...",
  "style_tags": ["...", "..."],
  "human_filming_note": "if this must be filmed rather than AI-generated, describe the setup"
}]
```

---

### STAGE 4: VIDEO PROMPT GENERATOR

**Input:** Episode JSON + Shot Array + Visual Prompts

**Prompt to Claude:**
```
[UNIVERSE_CONTEXT loaded as system prompt]

Generate a complete video production brief for this AFTERIMAGE episode.

Episode: {episode_json}
Shots: {shot_array}

Output as JSON:
{
  "episode_id": "...",
  "total_duration_seconds": 15,
  "production_mode": "filmed|ai-generated|hybrid",
  "music_direction": "...",
  "pacing_notes": "...",
  "color_grade_notes": "any deviation from standard grade for this anchor",
  "edit_sequence": "shot order with timing",
  "audio_decay_instruction": "exact second where visual cuts vs audio fades",
  "thumbnail_frame": "which part number produces the peak image"
}
```

---

### STAGE 5: CAPTION GENERATOR

**Input:** Episode JSON

**Prompt to Claude:**
```
[UNIVERSE_CONTEXT loaded as system prompt]

Generate a caption for this AFTERIMAGE episode.

Episode: {episode_json}

Caption rules:
- 3-8 words maximum
- Lowercase, no capitalization
- Fragment syntax — not a sentence
- Evokes the feeling without naming it
- Never explains the video
- Never references the creator

Generate 3 options ranked by strength.

Output as JSON:
{
  "primary": "...",
  "alt_1": "...",
  "alt_2": "..."
}
```

---

### STAGE 6: HASHTAG GENERATOR

**Input:** Episode JSON + last 10 episodes' hashtag sets

**Prompt to Claude:**
```
[UNIVERSE_CONTEXT loaded as system prompt]

Generate exactly 3 hashtags for this AFTERIMAGE episode.

Episode: {episode_json}

Hashtag rules:
- 3 hashtags only — never more
- One should be #afterimage (always)
- The other two rotate — never repeat the same set within 5 posts
- Atmospheric, not algorithmic (no #fyp, no #trending)
- Recent hashtag sets to avoid repeating: {recent_hashtags}

Output:
{"hashtags": ["#afterimage", "#...", "#..."]}
```

---

### STAGE 7: LORE DB UPDATE

**Input:** Episode JSON + shot array + caption

**Prompt to Claude:**
```
[UNIVERSE_CONTEXT loaded as system prompt]

Update the lore database for this completed episode.

Episode: {episode_json}

Extract and output:
{
  "episode_id": "...",
  "anchor_visited": "...",
  "symbol_state": "...",
  "other_one_presence": "...",
  "emotional_register": "...",
  "seeds_planted": ["list of anything introduced that could be referenced later"],
  "previous_seeds_activated": ["list of anything this echoes from prior episodes"],
  "continuity_flags": ["anything that must be maintained in future episodes"]
}
```

This JSON appends to `lore/episode_log.json`. The next Episode Generator call reads the last 5 entries to ensure non-repetition.

---

## THE TOOL STACK

### Orchestration Layer

| Tool | Role | Cost |
|------|------|------|
| **n8n** (self-hosted) | Workflow automation — chains all 7 Claude calls | FREE (self-hosted) |
| **n8n Cloud** | Alternative if no server | Free tier (limited) |
| **Make.com** | Alternative to n8n, slightly easier UI | Free tier (1,000 ops/month) |

n8n runs the entire pipeline as a single workflow. Trigger: manual button or schedule. Output: a complete upload package folder.

### Brain Layer

| Tool | Role | Cost |
|------|------|------|
| **Claude API** (claude-haiku-4-5) | All 7 pipeline stages | ~$0.001 per full pipeline run |
| **Claude.ai (Pro)** | Manual review, universe bible editing, prompt engineering | $20/month (you have this) |

**Cost per video generated:** ~$0.001 (1/10th of a cent) via Haiku API. 100 videos = ~$0.10. Not a cost consideration.

### Video Generation Layer

| Tool | Role | Cost |
|------|------|------|
| **Runway ML Gen-3 Alpha** | Primary AI video generation | 125 credits/month free (~25 5-second clips) |
| **Pika 2.1** | Backup / different style | Free tier (limited) |
| **CapCut** | Assembly, mobile editing, direct TikTok export | Free |
| **DaVinci Resolve** | Color grade, final polish | Free |

**On free tier limits:** Free tiers produce ~20-30 clips/month. For a 2-3 video/week cadence, this requires supplementing with filmed footage (Mode 1 from the pipeline) or purchasing Runway credits ($15/month for 625 credits = ~125 5-second clips = enough for 20+ videos). This is the only likely paid cost.

### Audio Layer

| Tool | Role | Cost |
|------|------|------|
| **Suno** | Generate custom ambient tracks | Free (10 songs/day) |
| **Freesound.org** | Field recordings, ambient layers | Free (CC license) |
| **ElevenLabs** | Ambient breath/texture if needed | Free tier |

### Database Layer

| Tool | Role | Cost |
|------|------|------|
| **Notion** | Episode log, symbol tracker, anchor map | Free |
| **Notion API** | n8n reads/writes Notion automatically | Free |
| **GitHub** | Version control for the universe bible and all prompts | Free |

### Distribution Layer

| Tool | Role | Cost |
|------|------|------|
| **Buffer** | Schedule TikTok + Instagram posts | Free (3 channels) |
| **CapCut** | Direct TikTok export | Free |

---

## THE FULL AUTOMATED WORKFLOW (n8n)

```
TRIGGER: Manual button (or Monday 9am schedule)
         │
         ▼
[Node 1] Read episode_log.json (last 5 entries) from Notion
         │
         ▼
[Node 2] Claude API → Stage 1: Episode Generator
         Input: feeling_tag (rotating list), anchor (rotating), last_5 from Node 1
         Output: episode_json
         │
         ▼
[Node 3] Claude API → Stage 2: Shot Generator
         Input: episode_json
         Output: shot_array
         │
         ▼
[Node 4] Claude API → Stage 3: Visual Prompt Generator
         Input: shot_array
         Output: visual_prompts_json
         │
         ▼
[Node 5] Claude API → Stage 4: Video Prompt Generator
         Input: episode_json + shot_array
         Output: production_brief_json
         │
         ▼
[Node 6] Claude API → Stage 5: Caption Generator
         Input: episode_json
         Output: captions_json
         │
         ▼
[Node 7] Claude API → Stage 6: Hashtag Generator
         Input: episode_json + recent hashtags from Notion
         Output: hashtags_json
         │
         ▼
[Node 8] Assemble Upload Package
         Creates folder: /packages/EP-{id}/
         Files:
           - production_brief.md (human-readable)
           - runway_prompts.txt (paste into Runway)
           - caption.txt (primary + alts)
           - hashtags.txt
           - shot_list.md
         │
         ▼
[Node 9] Claude API → Stage 7: Lore DB Update
         Input: episode_json + shot_array + caption
         Output: lore_update_json
         │
         ▼
[Node 10] Append lore_update_json to Notion episode log
          │
          ▼
[Node 11] Notify (email / Slack message / webhook)
          "Upload package ready: EP-{id} — {anchor} — {feeling_tag}"
```

**Your role as owner:** Receive the notification. Open the package. Paste the Runway prompts into Runway ML. Assemble the clips in CapCut or DaVinci. Schedule in Buffer.

**Time per video after automation:** 20-45 minutes (video assembly and schedule). The creative decisions are made by the machine. You are the final filter and the uploader.

---

## CHARACTER AND VISUAL CONSISTENCY — AUTOMATED

### The Problem

AI video generators don't maintain character consistency across generations. Each Runway call produces a different "anonymous woman."

### The Solution: The Protagonist Reference System

**Step 1:** Film or generate a single reference sequence — 30 seconds of the protagonist from behind, in the standard AFTERIMAGE color grade, at the Overpass.

**Step 2:** Export 4 reference frames as still images. Save as `studio/reference/protagonist_ref_1-4.png`

**Step 3:** Every Runway call that includes the protagonist: upload one of these reference frames as the Image-to-Video input. Runway Gen-3 uses it as a style/character anchor. The anonymous figure remains visually consistent.

**Step 4:** For Anchor locations — same process. One reference frame per Anchor. `anchor_overpass_ref.png`, `anchor_23rd_ref.png`, etc. Load the relevant one as style reference for each generation.

**This is manual once.** After the reference library is built, every future generation is automated.

### Visual Consistency Lock (Non-Negotiable)

The DaVinci Resolve Power Grade exports as a `.drx` file. Every generated clip, before final assembly, runs through DaVinci with this grade applied. This takes 2 minutes per video. It is the single most important consistency action.

Without this: the universe looks different every video, regardless of what prompts produce.
With this: every video is unmistakably AFTERIMAGE.

---

## THE FEELING TAG SYSTEM

The universe doesn't need complex lore to stay coherent. It needs **emotional coherence**.

The Feeling Tag is the primary input to every pipeline run. It constrains Claude to a specific emotional register for that episode.

**The Master Feeling Tag Library** (rotate through these — never repeat within 8 episodes):

```
"the weight of a place you can't return to"
"wanting to call someone and not doing it"
"3am in a city that doesn't know you"
"something ended without a ceremony"
"recognizing the last time after it's passed"
"the specific silence of an empty space that used to have someone in it"
"a decision that became permanent without feeling like one"
"the city as witness to things it can't help you with"
"what you carry instead of saying"
"a feeling that has no arrival time and no departure"
"the ordinary moment you didn't know to remember"
"the hour when you stop pretending it's fine"
```

Each of these is a universal human experience. The audience doesn't need to decode the lore to feel them. They already live there.

**This is the Kershaw principle correctly applied:** the content is about the feeling, not the mystery. The mystery (the cassette, The Other One) is texture that deepens the feeling — not a puzzle to solve.

---

## THE SIMPLIFIED LORE CONTRACT

Everything in the current lore bible that cannot be maintained automatically is cut.

**What remains — the Automated Lore Layer:**

| Element | How it's tracked | Automation-safe? |
|---------|-----------------|-----------------|
| Cassette state (held/found/left/missing/absent) | JSON field in episode log | Yes — Stage 7 tracks it |
| Other One presence level (none/implied/trace) | JSON field | Yes |
| Anchor visit log (when each was last used) | Notion DB | Yes — Stage 1 reads it |
| Feeling tag history (prevent repetition) | JSON list | Yes — Stage 1 reads it |
| Color grade | DaVinci Power Grade file | Yes — applied manually in 2 min |

**Everything else:** Not tracked. Not automated. Not part of the machine.

The Residuals, the Phase system, the 23rd Floor narrative, the handwriting sequence — these were writer's tools. They are not studio tools. Strip them.

The studio produces: **the same feeling, differently expressed, visually consistent, the cassette present or absent, The Other One implied or absent.**

That is enough. That is what Kershaw does.

---

## WHAT REQUIRES HUMAN JUDGMENT (The Minimal List)

1. **The feeling tag for each batch** — owner selects from the library or writes new ones. 5-minute decision.
2. **Final review of the upload package** — sanity check before Runway. 5 minutes.
3. **Video assembly** — paste Runway outputs into CapCut, apply grade, export. 20-30 minutes.
4. **Scheduling** — Buffer, 2 minutes.

**Total owner time per video: 30-40 minutes.**
**Total studio time per video (automated): ~60 seconds of compute.**

---

## STARTING THE MACHINE

**Week 1: Build the foundation**
1. Write `UNIVERSE_CONTEXT.md` — the 2,000-token compressed universe bible (1-2 hours, once)
2. Build the n8n workflow (3-4 hours, once — use the node diagram above)
3. Film or generate the protagonist reference frames (1-2 hours, once)
4. Build the DaVinci Power Grade (30 minutes, once)
5. Set up Notion databases with the schema from consistency_system.md

**Week 2: First production run**
1. Run the pipeline 3 times → 3 upload packages
2. Review the outputs — refine the prompts where needed
3. Produce the videos, build the 4-video reserve
4. Launch

After that: run the pipeline twice a week. Review. Generate. Post. Own.
