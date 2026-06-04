# AFTERIMAGE — AI Generation Stack
## Character consistency, style locking, visual memory. Built for a faceless protagonist.

---

## THE CORE INSIGHT

AFTERIMAGE has a structural advantage over every other AI-generated universe:

**The protagonist has no face.**

Character consistency in AI generation is hard because faces are hard. Runway Gen-3 cannot reliably produce the same face twice. LoRA training, IP-Adapter, reference images — all of these are solving the face problem.

We don't have a face problem. We have a **silhouette problem**. And silhouette consistency is trivially achievable with reference images.

This is why the no-face rule is not a creative choice. It is an engineering choice.

---

## THE PROTAGONIST REFERENCE SYSTEM

### What Defines the Protagonist Visually (No Face Required)

1. **Hair silhouette** — dark, medium length, slight movement, falling past shoulders. The shape of the hair against a lit background is the character's visual identity.
2. **Body proportion** — medium height, slightly slight build. The body as it appears from behind against a city background.
3. **Clothing palette** — dark, muted, layered. A dark jacket (longer cut), always. Underneath: darker. Nothing with logos, color, or definition.
4. **The way she stands** — weight slightly shifted. Not posed. Not balanced. A real person's standing posture.
5. **Scale relationship to environment** — she occupies 15-25% of the frame height. Always.

### Building the Reference Library (One-Time, ~2 Hours)

**Step 1: Generate the Character Sheet**

Use Midjourney (free tier) or Stable Diffusion (SDXL, free local):

```
Prompt:
character reference sheet, anonymous woman from behind, dark long jacket, 
dark clothing, medium-length dark hair, standing in urban night environment, 
multiple poses: [standing at railing / standing at window / walking away / 
sitting with back to viewer / hands holding object], 
no face visible, cinematic, analog grain, teal shadows amber highlights, 
9:16 vertical format, white background between poses

Negative: face visible, cartoon, anime, illustration, bright colors, 
logo, pattern, white clothing
```

Save 4 poses as: `visual/refs/protagonist_A.png` through `protagonist_D.png`

**Step 2: Generate Anchor Reference Frames**

One establishing shot per Anchor:
```
Prompt template:
[ANCHOR NAME] at night, urban, wet streets, neon reflections, 
nobody present, cinematic, analog grain, teal shadows amber highlights, 
dark teal-black shadows (#0a1a1f), amber practical lights, 
naturalistic lighting, single light source, 
[ANCHOR-SPECIFIC DETAIL]

Anchor-specific details:
- Overpass: elevated walkway, city below, cold blue-gray light, steel railing
- Convenience Store: fluorescent lit interior visible through glass, green-white light spilling onto wet pavement
- The Car: interior, dashboard warm amber light, city exterior cold through windows
- Apartment Window: seen from interior, looking out at city night, warm room light behind
- Stairwell: concrete stairs, deep blue ambient, one overhead light, bottom of frame dark
- 23rd Floor: exterior building, single window warm amber against cold city, high floor
```

Save as: `visual/refs/anchor_overpass.png`, `anchor_store.png`, etc.

**Step 3: Generate Object Reference**

The cassette:
```
Prompt:
close-up photography, vintage cassette tape, black shell, no label text, 
held in hands (hands only visible, no face), 
dark background, single warm amber light source from left, 
analog grain, teal shadows, shallow depth of field, 
cassette in sharp focus, background dissolved
```

Save as: `visual/refs/cassette_held.png`, `cassette_surface.png`

---

## RUNWAY GEN-3 — STYLE LOCKING PROTOCOL

### The Image-to-Video Workflow (Primary Method)

Runway Gen-3 Alpha accepts a reference image as the first frame. This is the character lock.

**Workflow:**
1. Select the appropriate protagonist reference (A, B, C, or D) based on the required pose
2. Upload as Image-to-Video starting frame
3. Apply the motion prompt
4. Apply the style prompt
5. Apply the negative prompt

**This produces:** A video where the first frame matches your reference exactly, and Runway animates from that foundation. The protagonist's silhouette, clothing, and scale remain consistent because the starting frame locks them.

### The Master Style Prompt (Append to Every Generation)

This goes at the end of every Runway prompt, always:

```
cinematic short film, analog film grain, split toning (teal shadows amber highlights), 
practical lighting only, single light source, shallow depth of field, 
naturalistic movement, slight camera drift, no stabilization, 
dark teal-black shadows, warm amber highlights, 
Kodak 5219 emulation, slightly underexposed, muted saturation
```

### The Master Negative Prompt (Append to Every Generation)

```
face visible, digital sharpness, bright colors, saturated colors, 
clean modern aesthetic, drone shot, smooth stabilized movement, 
text overlay, cartoon, animation, CGI render, studio lighting, 
multiple light sources, daylight, sunshine, cheerful, bright
```

### Per-Shot Prompts (Variable Portion)

The variable part of each prompt describes what's happening. Keep it under 80 characters.

Example full prompts:

**Overpass establishing shot:**
```
woman from behind at elevated walkway, city below, leaning on steel railing, 
dark jacket, medium dark hair, cold blue-gray ambient light, wet metal surface, 
[MASTER STYLE PROMPT]
```

**Convenience store approach:**
```
woman from behind walking toward fluorescent-lit convenience store at night, 
wet pavement reflecting green-white light, figure small in frame, 
dark jacket, city night behind her, [MASTER STYLE PROMPT]
```

**Cassette close-up:**
```
close-up of hands holding black cassette tape, no label, 
warm amber light from left, shadow teal-dark, no face visible, 
shallow depth of field, background dissolved, [MASTER STYLE PROMPT]
```

---

## PIKA 2.1 — BACKUP STACK

Use Pika when Runway free credits are exhausted.

Pika's strength: shorter clips (2-4 seconds), faster generation, better motion on static scenes.

Pika is better for: Mode 3 content (static with motion), object close-ups, the Signal effect.

**Pika prompt structure:** Identical to Runway. Same style lock, same negative prompt, same reference image upload.

**The Signal Effect in Pika:**
```
analog video static, brief 2-second interference pattern, slight color shift to magenta, 
grain intensifies, returns to clean, [MASTER STYLE PROMPT]
Negative: [MASTER NEGATIVE]
```

---

## COMFYUI — ADVANCED STACK (Optional, Significant Setup)

If you want full local control over character consistency with zero API costs:

**Setup:** ComfyUI + SDXL + IP-Adapter + AnimateDiff
- IP-Adapter: feeds reference image as style/character anchor for image generation
- AnimateDiff: converts still images to short video clips
- Cost: $0 (runs locally on GPU)
- Setup time: 4-6 hours
- Quality ceiling: Lower than Runway Gen-3, but fully controllable and free

**Use case:** Batch-generate 50 establishing shots across all 6 Anchors in one session. Then pull from this library for the environmental/wide shots in every video, rather than using Runway credits.

This is the cost-zero path for Mode 2 content at scale.

---

## THE DAVINCI POWER GRADE — THE FINAL LOCK

Every piece of footage — filmed, AI-generated, or static — runs through DaVinci Resolve before posting.

This is non-negotiable. This is what makes everything look like AFTERIMAGE.

### The Power Grade Settings (Exact)

```
1. Color Wheels — Log mode:
   Lift:      R: -0.04  G: -0.01  B: +0.02   (teal shadow)
   Gamma:     R: -0.01  G: -0.01  B: +0.01   (slight cool midtones)
   Gain:      R: +0.04  G: +0.00  B: -0.04   (amber highlights)

2. Curves — Custom:
   Luma: gentle S-curve (lift blacks slightly off floor, roll off highlights before clip)
   Saturation: reduce to 72%
   Hue vs Sat: boost cyan/teal +15%, boost amber/orange +20%, reduce red -10%

3. Color Space Transform:
   Input: Rec.709 (or camera native)
   Output: Rec.709
   Look: apply Kodak 5219 LUT at 65% opacity

4. Grain (Resolve FX):
   Grain type: Film grain
   Strength: 0.38
   Size: 1.2
   Softness: 0.3
   Monochrome: No (slight color variation in grain)

5. Vignette:
   Strength: -0.25
   Softness: 0.85
   (Subtle — felt, not seen)

6. Chromatic Aberration:
   Lateral: 0.4%
   Applied at edges only
```

**Save this as a Power Grade in Resolve's Gallery. Name it: AFTERIMAGE_MASTER**

Apply to every clip on import. Every clip. Every time.

---

## VISUAL MEMORY — THE REFERENCE LIBRARY STRUCTURE

```
visual/refs/
├── protagonist/
│   ├── protagonist_A.png    (standing at railing, from behind)
│   ├── protagonist_B.png    (walking away)
│   ├── protagonist_C.png    (standing at window, back to viewer)
│   └── protagonist_D.png    (seated, back to viewer, hands visible)
├── anchors/
│   ├── anchor_overpass.png
│   ├── anchor_store.png
│   ├── anchor_car.png
│   ├── anchor_window.png
│   ├── anchor_stairwell.png
│   └── anchor_23rd.png
├── objects/
│   ├── cassette_held.png
│   ├── cassette_surface.png
│   └── cassette_alone.png
├── motifs/
│   ├── reflection_window.png
│   ├── rain_in_light.png
│   └── two_light_sources.png
└── grade/
    └── AFTERIMAGE_MASTER.drx   (the DaVinci Power Grade export)
```

This library is built once. Every generation, every film session, every edit references it.

**The discipline:** Never generate a protagonist shot without uploading one of the 4 protagonist references as the starting frame. Never edit a video without applying AFTERIMAGE_MASTER. These two actions, enforced, produce a visually coherent universe across 100+ videos.

---

## STYLE CONSISTENCY CHECK — THE 5-SECOND TEST

Before posting any video, play the first 5 seconds.

Ask: if you encountered this on your feed with no account name visible, would you know it was AFTERIMAGE?

If the answer is uncertain: do not post. Fix the grade first.

The audience builds recognition in the first 200 milliseconds of a video. That recognition is entirely visual — color, grain, composition, light ratio. If those are right, the viewer is already inside the universe before the first movement.

That is the goal.
