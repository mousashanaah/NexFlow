# AUTOMATION DESIGN — Files from Other Timelines
## Target: owner spends under 30 minutes/day. Final state: under 5 minutes/day.

---

## THE THREE AUTOMATION STATES

```
State 1 — TODAY:   Human creates everything with AI assistance. ~60–90 min/video.
State 2 — WEEK 2:  Human reviews, AI produces. ~15–20 min/video. Templates built.
State 3 — MONTH 2: Human approves batches. AI produces and queues. ~30 min/week total.
```

The jump from State 1 to State 2 is templates.
The jump from State 2 to State 3 is API connections + batch workflow.

---

## STATE 2 — THE TEMPLATE SYSTEM (Build This Week)

No API required. No code. Just templates.

### Script Template

Save this as a Claude Project with the system prompt pre-loaded.
Each session: type the file number and one premise. Claude generates the full script.

```
Claude system prompt (save as Project):

You are the archivist for the Temporal Archive. Your job is to format recovered documents from parallel timelines.

Format every file as follows:
FILE #[NUMBER]
CLASSIFICATION: [CATEGORY in caps]
DATE: [specific date, or REDACTED]
---
[Content — 110-130 words, bureaucratic, specific, dry. One impossible premise. One mundane consequence. One closing detail that creates doubt about our own timeline. Never use: "imagine," "what if," "in a world where," or any narrative language.]
---
— Archived.

Rules:
- No drama. No wonder. No exclamation.
- Specific details over general claims. ("March 4th, 1987" over "decades ago")
- The twist arrives in the final sentence.
- Tone: medical report crossed with geographic survey.
```

**Usage:** Open Claude, type `File #017 — humans never developed sleep`, receive full script in 15 seconds. Copy to ElevenLabs.

### Voice Template

ElevenLabs settings saved as preset:
- Voice: [chosen voice]
- Stability: 0.75
- Similarity: 0.80
- Style: 0
- Speaker Boost: ON

Paste script → generate → download. 90 seconds per file.

### CapCut Template

Build once, reuse forever:

1. Create a 50-second project with all visual layers set up (background footage, dark overlay, scan-line effect, vignette)
2. Leave text layers empty
3. Save as CapCut template
4. Each new video: open template → update text layers with new file content → swap audio track → export

Assembly time with template: **5 minutes per video.**

### Total State 2 time per video

| Step | Time |
|---|---|
| Open Claude, type premise | 1 min |
| Review/approve script | 2 min |
| Paste to ElevenLabs, generate | 2 min |
| Open CapCut template, update text + audio | 5 min |
| Export + schedule | 3 min |
| **Total** | **13 minutes** |

**At 2 videos/day: 26 minutes/day.**

---

## STATE 3 — BATCH AUTOMATION (Month 2)

Requires: Claude API key ($5 loaded = ~1,000 scripts), ElevenLabs API key (paid plan), Buffer account.

### The Weekly Batch Workflow

**Sunday — 30 minutes total:**

```
Step 1 (5 min): Open Claude.ai or your automation tool
  → Request: "Generate 14 file scripts for the following premises: [paste 14 premises]"
  → Review all 14. Approve or edit.

Step 2 (5 min): Paste all 14 scripts to ElevenLabs in sequence
  → Generate all 14 audio files
  → Download as zip

Step 3 (15 min): Batch CapCut assembly
  → Open 14 template copies
  → Update text + audio for each
  → Export all 14
  OR: Use Pictory.ai / Invideo (paid, $19/month) which can batch-assemble with text + audio + template

Step 4 (5 min): Upload all 14 to Buffer
  → Schedule: 2/day for 7 days
  → Buffer auto-posts to TikTok, Instagram Reels, YouTube Shorts simultaneously
```

**Daily time: 0 minutes. You approved the batch on Sunday.**

---

## FULL AUTOMATION ARCHITECTURE (Month 3+)

This is the n8n pipeline. Requires: n8n (free self-hosted), Claude API, ElevenLabs API, Pexels API, Buffer API.

```
TRIGGER: Manual or scheduled (Sunday 9am)
  ↓
NODE 1 — PREMISE QUEUE
  Source: Google Sheet with file number + premise columns
  Read next 14 unused premises
  ↓
NODE 2 — SCRIPT GENERATION (Claude API)
  Input: premise + file number
  System prompt: [archivist system prompt]
  Output: full script text
  Store in: Google Sheet column C
  ↓
NODE 3 — VOICE GENERATION (ElevenLabs API)
  Input: script text
  Voice ID: [saved voice ID]
  Settings: stability 0.75, similarity 0.80
  Output: MP3 file URL
  Store URL in: Google Sheet column D
  ↓
NODE 4 — BACKGROUND FOOTAGE (Pexels API)
  Input: category tag from file classification
  Query: [category] + "night atmospheric slow"
  Output: video URL
  Store URL in: Google Sheet column E
  ↓
NODE 5 — ASSEMBLY
  Option A: Shotstack API (programmatic video assembly, $0.02/video)
    Input: voice URL + background URL + text content
    Template: pre-built in Shotstack
    Output: rendered video URL
  Option B: Manual notification
    n8n sends you a message: "14 scripts and audio ready — assemble in CapCut"
    This is the human-in-the-loop checkpoint
  ↓
NODE 6 — SCHEDULING (Buffer API)
  Input: rendered video URL
  Schedule: 2/day for 7 days
  Platforms: TikTok + Instagram Reels + YouTube Shorts
  ↓
NODE 7 — LOGGING
  Update Google Sheet: posted date, platform, scheduled time
  Create weekly KPI row for tracking
```

**Cost at full automation (60 videos/month):**
- Claude API (haiku): ~$0.05/script × 60 = $3
- ElevenLabs: $11/month (100K chars)
- Pexels API: free
- Shotstack: ~$0.02/video × 60 = $1.20
- Buffer: $6/month
- n8n: free (self-hosted) or $20/month (cloud)
- **Total: ~$21–$41/month**

**Revenue breakeven:** TikTok Creator Fund pays ~$0.02–0.06 per 1,000 views. At $41/month cost, breakeven = 700K–2M views/month. That's achievable at 100K+ followers.

**Better revenue path:** One sponsorship post at 100K followers = $500–$2,000. One sponsorship covers months of production cost.

---

## WHAT CANNOT BE AUTOMATED (human-required steps)

| Task | Why human | Time |
|---|---|---|
| Script quality review | AI can generate off-brand scripts — needs approval | 1 min/script |
| Category performance analysis | Interpret what's working and why | 15 min/week |
| Comment reading | Identify breakout patterns, fan theories, follow-up ideas | 10 min/day |
| Premise ideation | New premises need domain knowledge + novelty judgment | 20 min/week |
| Platform strategy adjustment | Algorithm changes, trending formats, opportunity recognition | 20 min/week |

**Irreducible human time at full automation:** ~30 minutes/day, with most of that being comment reading (which is also market research).

---

## THE AUTOMATION BUILD ORDER

Build in this order only. Each stage funds and informs the next.

```
Week 1:  State 2 — templates built. 13 min/video. 60 videos/month possible.
Week 2:  Validate format. Identify which categories and hooks perform.
Week 3:  Claude API integration for batch scripting. Reduce script time to 0.
Week 4:  ElevenLabs API + batch download. Reduce voice time to 0.
Month 2: Buffer API scheduling. Reduce scheduling time to 0.
Month 3: Shotstack or equivalent for video assembly. Reduce assembly to 0.
Month 3: Full pipeline: Sunday batch approval → week runs automatically.
```

Do not build Month 3 automation until the format is validated. Automating a format that doesn't work wastes development time. Validate first, automate second.

---

## THE 30-MINUTE DAY — WHAT IT LOOKS LIKE AT STATE 3

```
Morning (10 min):
  Read comments on yesterday's videos (7 min)
  Note which comment patterns appeared (3 min)

Weekly (once, Sunday, 20 min):
  Review 14 queued scripts (10 min)
  Approve or edit
  Trigger batch: voice + footage + scheduling runs automatically (0 min — you just press a button)
  
Done.
```

Everything else runs.
