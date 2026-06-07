# PRODUCTION STACK — Files from Other Timelines
## Free first. Easy second. Scalable third.

---

## THE PRODUCTION UNIT

One video = one file. Target specs:
- Duration: 40–55 seconds
- Format: 9:16 vertical (TikTok/Reels/Shorts)
- Audio: AI voice + ambient + subtle music
- Visual: Dark UI / document aesthetic + atmospheric overlay
- Production time target: 15 minutes human, 8 minutes machine

---

## FULL STACK — FREE TIER FIRST

### 1. SCRIPT GENERATION

**Tool:** Claude API (claude-haiku-4-5) or Claude.ai free
**Cost:** Free (claude.ai) or ~$0.001 per script (haiku API)
**Output:** 120–150 word script per file

**The system prompt (paste into Claude):**
```
You are an archivist generating recovered documents from parallel timelines.

STRUCTURE — every file must contain all three, in order:
1. HOOK — the first sentence stops the reader cold. A specific number, a death count, a reaction, a date. Something that happened. Not a description of a situation — an event.
2. REVEAL — the context that makes the hook make sense, and worse. The impossible fact, stated plainly as bureaucratic record.
3. TWIST — the final 1-2 sentences. Something small. Something that creates doubt about our own timeline. Not dramatic. Quiet. Wrong.

FORMAT RULES:
- Bureaucratic tone throughout. Dry. Specific. Medical report crossed with geographic survey.
- Every detail is concrete: dates, percentages, measurements, locations, names (if any), durations.
- Never use: "imagine," "what if," "in a world where," "fascinating," "incredible," "eerie," "strange," "mysterious"
- The impossible premise is never called impossible. It is stated as fact and documented consequence.
- Total word count: 115–135 words
- End with: "— Archived." Hard stop. No reflection, no summary, no outro.

THE DIFFERENCE:
DEAD: "In this timeline, humans cannot see in darkness. This created problems."
ALIVE: "The first recorded panic event occurred on Day 3. 4.2 million people had never experienced total darkness before. Candle production increased 3,000% within the first week. The grid was restored on Day 11. Fourteen people did not leave their homes again."

Format:
FILE #[NUMBER]
CLASSIFICATION: [category]
DATE: [date or REDACTED]
---
[HOOK sentence]
[REVEAL — 3-5 sentences]
[TWIST — 1-2 sentences]
---
— Archived.
```

**Batch generation:** One Claude session generates 20 scripts in 3 minutes. Copy-paste to voice tool.

---

### 2. VOICE GENERATION

**Tool:** ElevenLabs
**Cost:** Free tier (10,000 characters/month ≈ 12–15 videos/month)
**Upgrade:** $5/month (30,000 chars ≈ 35–40 videos/month), $11/month (100,000 chars)

**Voice selection criteria for Files format:**
- Neutral, slightly formal (not warm, not cold)
- Male or female — the voice should sound like a system, not a person
- No emotion variance — same tone for mundane and impossible content equally
- Recommended ElevenLabs voices: "Antoni" (calm, slight formality), "Rachel" (neutral, clear), or clone a custom voice

**Settings:**
- Stability: 0.75 (higher = more consistent, fewer variations)
- Similarity: 0.80
- Style: 0
- Speaker Boost: ON

**The pacing rule:** Script lines should be separated by a single line break in ElevenLabs to create a natural pause between the header, content, and closing stamp.

**Alternative free voice tools:**
- **Murf.ai** — free tier (10 min/month), good quality
- **TTSMaker** — free, no signup, lower quality but functional
- **Play.ht** — free tier (12,500 chars/month)

---

### 3. VISUAL GENERATION

**The key insight about Format A visuals:**

Format A does NOT need heavy image generation. The visual identity IS the document aesthetic — dark screen, monospace text, file header appearing before the voice speaks. This is cheaper and more distinctive than AI-generated scene imagery.

**Visual approach — two layers:**

**Layer 1 (text/UI — always present):**
- Dark background (#0d0d0d or #111116)
- File header text appears line by line (typewriter effect)
- Classification badge, file number prominent
- Subtle scan-line or noise overlay
- This is built in CapCut, Canva, or a simple video template — no AI image gen required

**Layer 2 (atmospheric footage — behind or beside text):**
- Slow-moving nature/space/urban footage as background
- Low opacity (15–30%) so text remains primary
- Source: Pexels.com free stock (search "night city slow," "fog forest," "space slow," "rain abstract")
- No AI generation required — free stock covers this entirely

**Optional AI image layer** (when you want a specific visual):
- **Bing Image Creator** (free, DALL-E 3) — generate one atmospheric background image per video
- **Leonardo.ai free tier** (150 tokens/day) — better quality, painterly outputs

**Template structure (CapCut or DaVinci):**
```
Background: slow atmospheric stock footage (opacity 20%)
Overlay: dark gradient (bottom-to-top, opacity 85%)
Text layer 1: FILE #NUMBER — top-left, monospace, size 28, appears at 0:00
Text layer 2: CLASSIFICATION + DATE — appears at 0:01
Divider line — appears at 0:02
Text layer 3: Content text — appears at 0:03, line by line (typewriter)
Text layer 4: "— Archived." — appears at end, red or amber, size 32
```

---

### 4. AUDIO — MUSIC + AMBIENT

**Tool:** Suno.ai (free tier) or royalty-free sources
**Cost:** Free

**Music requirement for Files format:**
- No rhythm, no melody that draws attention
- Droning, ambient, slightly unsettling
- Under the voice, not competing with it
- Duration matches video (trim in CapCut)

**Suno prompt:**
```
ambient drone, no melody, low frequency hum, slight dissonance, 
electronic, cold, institutional, 60 seconds, no percussion, no vocals,
background texture only
```

**Free alternative sources:**
- Pixabay.com/music — search "ambient dark" or "drone"
- freesound.org — "institutional ambient," "quiet room tone"

**Audio mix:**
- Voice: 0db (primary)
- Ambient/music: -18db (present but not heard consciously)
- Optional: very faint static or tape hiss at -28db

---

### 5. ASSEMBLY

**Tool:** CapCut (free, no watermark if account created)
**Alternative:** DaVinci Resolve (free, more control)

**CapCut assembly — 8 steps:**

1. Import atmospheric background footage, trim to 50 seconds
2. Drop opacity to 20–25% (Adjustment → Opacity)
3. Add dark overlay shape (full frame, #0d0d0d, opacity 75%)
4. Import voice audio track
5. Add text elements in sequence (file header → content → stamp)
   - Font: Space Mono or Courier New (monospace required)
   - Color: white or #e8e8e8 (slightly off-white = less sterile)
   - "— Archived." line: amber #e8a856 or red #c44a4a
6. Add typewriter animation to each text element (CapCut: Animation → In → Typewriter)
7. Add scan-line effect: Effects → Video Effects → Film → "Scan Lines" at 15% opacity
8. Import ambient audio at -18db
9. Export: 1080x1920, H.264, 30Mbps

**Total assembly time:** 8–12 minutes once template is saved

**CapCut template strategy:** Build the template once (step 1–8 without content). Save as template. Each new video: swap the voice file, update the text layers. Assembly drops to 5 minutes.

---

### 6. SCHEDULING

**Tool:** TikTok native scheduler (free) or Buffer free tier (3 channels, 10 posts queue)
**Cost:** Free

TikTok's native scheduler allows scheduling up to 10 days out. Buffer allows up to 3 social profiles and 10 queued posts free.

**Recommended posting frequency:** 1x/day or 2x/day during growth phase. Files format has infinite content — frequency is a lever, not a constraint.

---

## COMPLETE FREE STACK SUMMARY

| Function | Tool | Cost | Time/video |
|---|---|---|---|
| Script | Claude.ai free | $0 | 2 min |
| Voice | ElevenLabs free | $0 (15 vids/month) | 2 min |
| Visuals | Pexels stock + CapCut | $0 | 5 min |
| Music | Suno free / Pixabay | $0 | 1 min |
| Assembly | CapCut | $0 | 8 min |
| Scheduling | TikTok native | $0 | 1 min |
| **TOTAL** | | **$0/month** | **19 min/video** |

**At 2 videos/day:** $0/month, ~40 minutes/day human time.

---

## PAID TIER UPGRADE PATH (when revenue justifies)

| Current bottleneck | Upgrade | Cost | Time saved |
|---|---|---|---|
| 15 videos/month voice limit | ElevenLabs $11/month | $11 | Unlimited |
| Manual CapCut assembly | Pictory.ai or Invideo | $19/month | 6 min/video |
| Manual scheduling | Buffer Essentials | $6/month | 2 min/video |
| Manual script batching | Claude API (haiku) | ~$2/month at 60 videos | Automatable |

**Full paid stack at 60 videos/month:** ~$38/month. Revenue breakeven at ~40K monthly views on TikTok Creator Fund, or 1 sponsorship at 100K followers.

---

## THE AUTOMATION CEILING — WHAT CAN RUN WITHOUT YOU

**Fully automatable today:**
- Script generation (Claude API → batch 20 scripts, store in queue)
- Voice generation (ElevenLabs API → process queue automatically)
- Scheduling (Buffer API)

**Partially automatable today:**
- Assembly (CapCut template reduces to 5 min manual; Pictory.ai can automate further)
- Visual generation (Pexels API for stock footage selection)

**Requires human today:**
- Quality review (is the script good? does the voice pacing work?)
- Final export trigger

**Target state (Phase 2 automation):**
Claude generates 7 scripts Sunday → voice generated automatically → assembled by template → scheduled for the week → human spends 20 minutes Sunday reviewing before approval.

Daily human time at full automation: **under 5 minutes** (approval only).
