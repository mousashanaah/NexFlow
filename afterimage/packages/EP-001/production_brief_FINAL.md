# EP-001 — REVISED (Kershaw Challenge Applied)
**Original:** 8 shots, 15 seconds, 8-part structure
**Revised:** 5 shots, 10 seconds, emotion-first

---

## THE CHALLENGE

Shot by shot, original EP-001:

| Shot | Original Purpose | Does it create emotion? | Keep? |
|------|-----------------|------------------------|-------|
| 1. Railing texture | Establish materiality | YES — grounds the viewer in a real, cold place | KEEP |
| 2. She's at the railing, small | Figure in space | YES — she is small against something large | KEEP |
| 3. Hands tighten on railing | Movement initiation | WEAK — the tightening is too readable. Too much direction. | REVISE |
| 4. Looking down at city | Tempo coupling | NO — this is a "pretty city shot." It exists because the structure demanded something here. | **CUT** |
| 5. Hands close-up, knuckles | Compression | YES — the most physical shot, the most felt | KEEP |
| 6. Silhouette, city behind | Peak image | YES — this is the screenshot frame | KEEP |
| 7. One hand lifts, cut | Release | YES — the ambiguity is the emotion | KEEP |
| 8. 5 seconds audio decay | Structure | NO — 5 seconds of black is a writer's idea, not a viewer's experience | **REVISE → 2 seconds** |

**Shot 4 is the test.** Ask: if you removed it, would the video lose emotion? No. It would gain momentum. It only exists because the 8-part structure said "tempo coupling goes here." That is a lore reason. Cut it.

**Shot 3 (tighten) vs Shot 5 (close-up):** You don't need both grip moments. The close-up of knuckles (Shot 5) does more than the tightening action (Shot 3). Keep the image. Remove the action annotation.

---

## EP-001 — FINAL VERSION

**Duration:** 10 seconds
**Shots:** 5
**Rule:** every shot exists because it creates a feeling, not because a framework needed it filled

---

### SHOT 1 (0:00–0:02) — The Place
Wet steel railing, extreme close-up. Rain residue pooled in the channel. Cold. The texture of a surface that has been gripped thousands of times.

*What this creates:* weight, coldness, duration. You are somewhere real.

---

### SHOT 2 (0:02–0:05) — She Is There
She is already at the railing. We pull back to find her. She is small. The city is enormous behind her. She has been here before. We don't know how long.

*What this creates:* scale. The city doesn't know she's there. She is alone inside something very large.

---

### SHOT 3 (0:05–0:07) — The Hands
Her hands on the railing. Extreme close-up. The knuckles. The grip. Nothing else — no action, no tightening, no movement. Just: the grip.

*What this creates:* the feeling of holding on to something. Not dramatically. The way you hold on when you don't know what else to do.

---

### SHOT 4 (0:07–0:09) — The Peak
Her silhouette from behind, city behind her, both hands visible at the railing. The cold city glow. Her and the city and nothing between them except her grip.

*This is the frame. The screenshot. The one image that contains everything.*

---

### SHOT 5 (0:09–0:10) — The Cut
One hand lifts from the railing. Not dramatically. Just: it lifts. **Cut at the moment of lift.** Before we see what happens to it.

Black. City ambient continues 2 seconds. Gone.

*What this creates:* the ache. The brain needs to know where the hand went. It doesn't get to know. It replays to find what it missed. There is nothing to find.*

---

## PRODUCTION NOTES (Revised)

**Total runtime:** 10 seconds + 2 seconds black/audio = 12 seconds
**Music:** Single sustained note — arrives at 0:02, holds through 0:09, cuts with video. Not a fade. Cut.
**Grade:** AFTERIMAGE_MASTER + Overpass (cold blue-gray, max shadow ratio)
**The only thing that matters in the grade:** her hands must be the warmest element in the frame. Everything else is cold.

---

## RUNWAY PROMPTS — FINAL (Ready to Paste)

**REFERENCE:** `visual/refs/protagonist_A.png` for Shots 2, 3, 4, 5
**REFERENCE:** `visual/refs/anchor_overpass.png` for Shot 2 background

---

**SHOT 1:**
```
wet steel railing extreme macro close-up, rain pooled in metal channel groove, cold blue-gray steel surface, no people, no movement, night, static camera
cinematic, analog film grain, deep teal-black shadows (#0a1a1f), cold blue-gray ambient, Kodak 5219 emulation, muted saturation, slightly underexposed, practical lighting only
Negative: faces, warm light, bright colors, digital sharpness, smooth, daylight, people
```

**SHOT 2:**
```
[USE protagonist_A.png as Image-to-Video starting frame]
woman from behind standing at elevated walkway railing, city lights spread far below, dark long jacket, medium dark hair, figure lower-center-frame very small against large city, cold blue-gray ambient, wet steel railing, she is already there when we find her, static camera barely drifting
cinematic, analog film grain, cold blue-gray dominant, single ambient light source from city below, Kodak 5219, muted saturation, underexposed, figure 20% of frame height
Negative: face visible, dramatic lighting, warm ambient, figure large in frame, drone, smooth stabilized
```

**SHOT 3:**
```
[USE protagonist_A.png — crop to hands on railing only]
extreme close-up both hands gripping steel railing, knuckles visible, cold steel surface, single dim light catching skin from below, deep teal shadow one side warm skin the other, shallow depth of field, static, nothing else in frame
cinematic, heavy analog film grain, split toning warm amber on skin cold teal on steel, Kodak 5219, shallow DoF, practical single light
Negative: face, body above wrists, bright, multiple lights, movement
```

**SHOT 4:**
```
[USE protagonist_A.png as Image-to-Video starting frame]
woman silhouette from behind at railing, BOTH HANDS VISIBLE gripping rail, city spread behind her, cold blue-gray city glow, she is the only warm element in a cold frame, completely still, dark jacket, hair catching faint light from city below
cinematic, analog film grain, cold blue-gray dominant, barely perceptible warm amber on protagonist only, Kodak 5219, underexposed, muted saturation, strong silhouette
Negative: face, face reflection, bright, warm city, movement, dramatic
```

**SHOT 5:**
```
[USE protagonist_A.png — motion: one hand beginning to lift from railing]
woman from behind at railing, right hand beginning to lift from steel rail — generate ONLY the first 1 second of this motion, do not complete it, static camera, cold ambient, city behind
cinematic, analog film grain, cold blue-gray, Kodak 5219, the motion just begins
Negative: completing the motion, face, dramatic, fast
```

---

## DAVINCI ASSEMBLY (Exact)

```
Timeline: 24fps, 9:16 vertical, 1080x1920

00:00 – 02:00  Shot 1 (railing texture)
02:00 – 05:00  Shot 2 (she is there)
05:00 – 07:00  Shot 3 (the hands)
07:00 – 09:00  Shot 4 (the peak / screenshot frame)
09:00 – 10:00  Shot 5 (the lift — cuts on the motion)

10:00 – 12:00  Black frame (no video)

AUDIO:
- Layer 1 (ambient): city night ambient from Freesound.org — runs 00:00 to 12:00 at -18db
- Layer 2 (music): single sustained note (Suno prompt below) — enters at 02:00, holds, cuts hard at 10:00 (NOT a fade)
- At 10:00: video AND music cut simultaneously. Only ambient layer continues.
- At 12:00: ambient fades to silence over 0.5 seconds

COLOR:
- Apply AFTERIMAGE_MASTER Power Grade to all clips on import
- Overpass script: push shadows to #0a1a1f (cold teal-black)
- Add Power Window on Shot 3 and 4 to isolate her hands/silhouette: boost amber +10% on those areas only
- The hands must be the warmest thing in the frame

EXPORT:
- H.264, 1080x1920, 30Mbps
- No watermark, no title card, nothing
```

---

## SUNO MUSIC PROMPT

```
ambient, single sustained cello note, slightly detuned, analog warmth, 
very quiet, no rhythm, no melody, just one held tone with slight bow noise,
10 seconds, fades naturally, no percussion, melancholic, cinematic
```

Download the output. Use only the 10 seconds from 0:00–0:10. Cut hard at 10 seconds.

---

## THE VIEWER TEST — HONEST PRE-EVALUATION

If this is executed correctly:

**Would you stop scrolling?**
Yes — Shot 2. The scale of the figure against the city, and the feeling that she was already there before you arrived. The algorithm rewards the first 2 seconds. Shot 1 (the railing texture) is the hook: unfamiliar, tactile, slightly wrong. That 2-second texture shot is the scroll-stop.

**Would you rewatch?**
Yes — because of Shot 5. The hand lifts. The cut happens. Where did it go? The brain replays to find the answer. There is no answer. The loop is mechanical.

**Would you feel something?**
Yes — if the grade is correct. The cold/warm split on Shot 3 (cold steel, warm hands) is the emotional moment. It's not stated. It's physical. The feeling comes from the contrast, not from the narrative.

**Would you send it to a friend?**
Uncertain at this stage. EP-001 alone is not sendable — it's too quiet. By EP-007 (the Echo with the cassette), yes. The first video establishes the world. The sendable moment comes when the world has accumulated enough weight to transfer.

**Would you comment?**
The caption "she still comes here" implies: she has been here before. There is no "before" in EP-001. That implication is the comment trigger. Someone will ask what she comes back to. Someone else will say they know exactly how this feels. That's the two comment types that matter.

**The risk:** If Shot 2's grade is too dark — if you can't read her silhouette against the city — the video loses everything. The cold/warm contrast only works if she is legible as a figure. Test the export at thumbnail size before posting.

---

## REVISION CRITERIA

If the assembled video doesn't work, the failure will come from one of three places:

1. **Grade too flat** — if everything is the same temperature, the emotional contrast is gone. Push the split harder.
2. **Shot 2 too short** — 3 seconds isn't long enough for the viewer to feel the scale. If it feels rushed, extend to 4 seconds, cut Shot 1 to 1 second.
3. **The cut on Shot 5 too late** — if the hand is 30% lifted before the cut, the viewer sees too much. The cut should happen at 5% of the motion. The brain should not be sure it even moved.
