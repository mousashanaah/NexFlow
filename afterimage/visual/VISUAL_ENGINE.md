# AFTERIMAGE — Visual Engine
## The bottleneck is here. Everything else waits.

---

## THE ACTUAL PROBLEM

The Visual DNA exists. The feeling is defined. The locations are named.

None of that matters until one question is answered:

**Can we generate a frame — with no text, no caption, no context — that a stranger would recognize as belonging to a specific visual universe?**

Not "is it moody." Not "does it look cinematic." Those are adjectives that describe half the internet.

The test: show someone the frame cold. If they say "I know where this came from" — not what it IS, but that it CAME from somewhere — the visual engine exists.

Until that test passes, nothing else is built.

---

## THE STYLE CATEGORY CORRECTION

The VISUAL_DNA.md was built from photographic/cinematic references (Kershaw, cazsmir — real footage). The moodboard analysis pointed somewhere else:

- deviantArt / Tumblr / flash game era (2003–2012)
- Visual novel dialogue boxes
- Dreamcore / weirdcore
- Animated background plates (not photographs)
- Memory as texture, not memory as film

**The target is not: photograph with grain.**

**The target is: stylized painting that moves slightly.**

This changes which tools work and which fail.

| Approach | What it produces | AFTERIMAGE fit |
|---|---|---|
| Photorealistic AI (Midjourney photo mode) | Looks like stock photo | Wrong |
| Film emulation (grain + LUT on photo) | Looks like A24 trailer | Wrong |
| Anime/illustration AI (standard) | Looks like webtoon | Wrong |
| **Painted atmospheric illustration** | Looks like it was made by hand, for a purpose | **Target** |
| **Painted + subtle world motion** | Looks like a memory playing back | **Target** |

The reference that describes the target visually: **the background plates in a visual novel by a solo developer who studies film.** Not screenshots from a studio. From one person who cares too much about lighting.

---

## THE VISUAL ENGINE — TOOL OPTIONS

Ranked by fit for AFTERIMAGE's specific aesthetic. Not by general quality.

---

### OPTION A: Midjourney with `--style raw` + painting parameters
**Cost:** $10/month
**Fit:** Highest ceiling, most controllable

Midjourney in `--style raw` mode disables its default "pretty" aesthetic filter. Combined with painting/illustration parameters, it produces atmospheric frames rather than polished renders.

**The parameters that matter:**

```
--style raw          (disables Midjourney's beautification)
--stylize 0          (removes aesthetic bias — raw interpretation of prompt)
--ar 9:16            (vertical format)
--v 6                (current version)

For painting quality:
"oil painting", "gouache", "painterly", "impasto edges"

For the AFTERIMAGE atmosphere:
"visual novel background art", "atmospheric night scene", 
"single light source", "deep teal shadows", "amber warmth"
```

`--sref` with 3–5 reference images from your moodboard will do more than any prompt.

**The workflow:** Generate 20 frames. Use the 3 that pass as `--sref` for the next 20. The style compounds.

---

### OPTION B: Leonardo.ai (free tier — 150 tokens/day)
**Cost:** Free
**Fit:** High for painted/illustrated style if correct model selected

Leonardo has models specifically trained on atmospheric illustration. The relevant ones:

- **DreamShaper v7** — painterly, atmospheric, handles moody night scenes well
- **AlbedoBase XL** — consistent painted quality, less default anime bias than other SDXL models
- **Leonardo Diffusion XL** — their own model, good atmospheric rendering

**Free tier gives ~8–12 quality generations per day.** Enough to run the 20-frame experiment over 3–4 days.

**The model that gets closest to AFTERIMAGE:** DreamShaper v7 with negative embeddings to remove anime sheen.

---

### OPTION C: Bing Image Creator (DALL-E 3) — mentioned in VISUAL_DNA.md
**Cost:** Free (25 boosts/day, then slower)
**Fit:** Medium — DALL-E 3 can do painterly but defaults to illustration-bright

DALL-E 3 handles atmosphere reasonably at no cost. Its weakness: it defaults toward legible, well-lit scenes. Fighting that default is possible but requires precise negative description.

**Use case:** Fast iteration and testing prompts before spending Midjourney credits.

---

### OPTION D: NovelAI Diffusion (Anime V3)
**Cost:** ~$10/month
**Fit:** High for the illustrated/painted direction

NovelAI is trained specifically on illustrated art rather than photographs. Its outputs have the quality of something drawn by a skilled illustrator — which is closer to the "stylized living painting" target than any photorealistic model.

**Strength:** The atmospheric night scenes it produces have genuine painterly depth, not filter-depth.
**Weakness:** Default skews anime. Requires specific negative prompting to neutralize this toward "atmospheric illustration."

---

### OPTION E: SDXL + specific LoRA (local or Replicate)
**Cost:** Free local (needs GPU) or ~$0.01/generation on Replicate
**Fit:** Highest control, most effort

Running SDXL with a LoRA specifically trained on atmospheric illustration (not AFTERIMAGE LoRA yet — that comes later) gives full control over style without fighting a model's defaults.

Relevant public LoRAs on CivitAI:
- **Painterly illustration LoRA** — trained on atmospheric illustration art
- **Night scene atmospheric LoRA** — moody exterior scenes with correct light behavior
- **Visual novel background LoRA** — background plate quality, not character focus

**This is the path to permanent style lock without Midjourney dependency.**

---

## THE 20-FRAME EXPERIMENT

This is the next milestone. Nothing else starts until this is done.

### The 6 Locations (generate 3–4 frames each)

| Location | Key visual element | The withheld detail |
|---|---|---|
| Overpass | City below, single amber light | A figure at the railing (not yet visible) |
| Convenience store | Fluorescent interior, dark exterior | Someone inside, back to glass |
| Apartment window | Lit interior, night city behind | What's in the room — barely suggested |
| Motel sign | Blinking light, wet parking lot | A car, a door half-open or closed |
| Empty room | One light source, bare surfaces | Evidence of someone who was here |
| Snowy street | Street ahead, no destination visible | Footprints that might belong to the viewer |

### The Prompt Structure

Every frame prompt follows this structure:

```
[LOCATION DESCRIPTION] — be specific, not cinematic
[LIGHT — single source, describe it exactly]
[SHADOW — teal-black, deep, chromatic]
[WHAT IS ABSENT — specify what is NOT in the frame]
[STYLE — painted, not photographed]
```

Example (overpass):
```
elevated walkway at night, city lights far below, wet concrete railing in foreground,
single amber streetlight below casting upward light on railing surface,
deep teal-black shadows in background buildings, all windows dark except one,
no people visible, rain-wet surfaces, far city glow,
atmospheric night illustration, painterly, visual novel background art style,
impasto texture, analog warmth in highlights, deep shadow detail, 
NOT: sharp digital render, NOT: anime character style, NOT: stock photo composition
```

---

### The Pass/Fail Test

Show the generated frame to someone who hasn't seen your references.

**Ask only:** "Where do you think this came from?"

**Pass:** Any answer that implies it was made — not found. "A game?" "An illustration?" "Someone's art project?" "I don't know but it feels like something specific."

**Fail:** "AI," "Midjourney," "aesthetic TikTok," "filter," "VSCO," or no reaction at all.

A secondary test: hide the frame in a folder with 10 random atmospheric AI images. Come back in 3 days. Open the folder. Which frames do you recognize as yours immediately? Those are the frames that have identity.

---

### Rejection Criteria — What Kills a Frame

These are the things that make a frame generic. If any of these are present, reject immediately:

1. **Even lighting** — if the whole frame is roughly the same brightness, the emotional split is gone
2. **Blue-purple shadows** (not teal) — this is the most common AI default for "night scene." AFTERIMAGE shadow is #0a1a1f — teal-black, toward blue-green, not blue-purple
3. **Subject centered** — AFTERIMAGE subjects are off-center or absent. Centered composition reads as intentional (generic)
4. **Warm overall** — if the frame feels warm, the warmth has lost meaning. Only the light source should be warm
5. **Too resolved** — if every detail is sharp, the frame is asserting itself too loudly. Something must be soft, withheld
6. **Photorealistic faces** — if a face appears fully rendered, it anchors the viewer to a specific person. The figure must be anonymous, turned away, or partially obscured
7. **Obviously AI** — if the first reaction is "that's Midjourney," the tool's fingerprint is overriding the identity

---

## AFTER 20 FRAMES — WHAT TO DO WITH THE SURVIVORS

If 3–5 frames pass the test:

1. **Extract the common properties.** Not in words — visually. Line them up. What do they share that the rejected frames don't?

2. **Use survivors as `--sref` (Midjourney) or reference images (all other tools).** The style compounds from its own examples, not from descriptions of its examples.

3. **Generate 20 more.** Compare the second 20 to the first survivors. The pass rate should increase. If it doesn't — the survivors weren't specific enough.

4. **The LoRA lives here.** Once you have 20–30 frames that all pass the test, that set is the training data for a style LoRA. The LoRA makes every future generation automatically AFTERIMAGE without reference images.

5. **Animation starts here.** Only after stable stills exist. Animate the survivors using Kling I2V with a 2–3 word motion prompt ("snow falls," "light flickers," "curtain moves"). The motion should feel inevitable, not imposed.

---

## THE CORRECT SEQUENCE

```
1. Choose tool (Midjourney OR Leonardo free tier to start)
2. Generate 20 candidate frames across 6 locations
3. Apply pass/fail test
4. Identify survivors (expect 3–5 out of 20)
5. Extract what the survivors share
6. Generate 20 more using survivors as reference
7. When pass rate reaches 50%+: train LoRA
8. With stable style: first animation test
9. With stable animation: Post 1
```

Post 1 is step 9.

Not step 1.

---

## THE HONEST ASSESSMENT

Every post, every caption, every schedule, every anomaly — none of it protects you if the visual doesn't have a recognizable signature.

The accounts that have the cult quality you're studying — Kershaw, cazsmir, Entergalactic — are immediately recognizable not because of what they post about, but because of how every frame looks. You could see one frame with no caption, no username, no context, and know where it came from.

AFTERIMAGE needs that before anything else.

Find the frame. Build everything from the frame.
