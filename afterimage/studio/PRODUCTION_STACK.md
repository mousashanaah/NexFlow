# AFTERIMAGE — Production Stack Architecture
## Designed around identity, not around AI video generation.

---

## THE DIAGNOSIS

The Runway output looked like Runway because of a structural problem, not a prompt problem.

Text-to-video models generate from their training distribution. When you prompt Runway, you receive Runway's interpretation of your description. The style is determined by the model's priors — by the average of everything it was trained on that matches your words. No prompt engineering fully overcomes this. You are negotiating with defaults, not owning them.

The accounts you're studying do not have this problem because they are not using text-to-video. They are using:
- Real footage (Kershaw, cazsmir)
- Illustrated/animated pipelines (Entergalactic)
- Static image with selective motion (anukitsu)

The lesson: **style is established at the image stage, not the video stage.** Motion is added after. The visual identity lives in the frame, not in the movement.

This means the pipeline question is not "which video model?" It is "which image generation approach locks our style?"

---

## THE CORE ARCHITECTURAL DECISION

**Text-to-Video:**
```
Prompt → Model interprets → Runway's version of your style
```

**Image-to-Video:**
```
Style-locked frame → Motion added → Your style, animated
```

Image-to-Video is definitively correct for AFTERIMAGE. Here is why:

The image generation layer has tools the video generation layer does not:
- LoRA training (bake your exact style permanently into a model)
- Style reference (feed images, not words, as the style anchor)
- Character reference (hold the silhouette across every generation)
- img2img (start from an existing image and iterate)
- ControlNet (lock composition while style is applied)

Once you have a frame that could only belong to AFTERIMAGE, I2V adds motion. The motion is secondary. The frame is the product.

**The first 100 videos should be: Image generation → Image-to-Video.**

---

## THE TOOL EVALUATION

### IMAGE GENERATION — Style Consistency

**Option A: Midjourney (Recommended for immediate start)**
Cost: $10/month (Basic plan)

Two parameters make Midjourney the right starting tool:

`--sref` (Style Reference): You upload images — reference frames that define the AFTERIMAGE look — and Midjourney generates matching that style. Not matching the description of that style. Matching the actual visual quality of the images you feed. At `--sref 800` the style lock is strong enough that every generation reads as belonging to the same universe.

`--cref` (Character Reference): Maintains the protagonist's silhouette, hair shape, and jacket across every generation. Feed it one clean reference image and the figure stays consistent without retraining anything.

**This is the fastest path to style consistency.** No training, no local GPU, no setup. Generate a set of AFTERIMAGE reference frames manually (the best outputs from any session), then use those as `--sref` for all future generations. The style compounds — each generation can feed the next.

---

**Option B: SDXL + LoRA Training (Recommended for permanent lock)**
Cost: ~$3–5 one-time training run on Replicate.com, then $0 (free local) or cheap per generation

A LoRA is a fine-tuning layer trained on your specific images. Train it on 20–50 reference frames that define the AFTERIMAGE aesthetic — the teal-amber split, the grain, the shadow ratio, the light character — and it bakes that style into the model permanently. Every generation after training automatically produces frames that look like AFTERIMAGE without any style prompting.

The difference between `--sref` and a LoRA:
- `--sref`: You reference the style each generation. Consistent but requires the reference images every time.
- LoRA: The style is in the model. Generate anything and it comes out AFTERIMAGE. No reference needed.

Training platforms:
- **Replicate.com** — cloud training, **~$0.60–$0.90 per run** ($0.000975/second on L40S GPU, 10–15 min run). No GPU required. Download weights after. Needs only 5–6 images to start.
- **Kohya_ss** — free, local, requires ~8GB VRAM GPU
- **Ostris AI Toolkit** — easier than Kohya, local, same GPU requirement

The LoRA is a one-time investment. After it exists, every generation in the pipeline is automatically style-consistent.

---

**Option C: Flux (Black Forest Labs)**
Available via Replicate API or locally.

Flux produces higher quality images than SDXL and takes LoRAs. If you train a Flux LoRA on AFTERIMAGE reference frames, you get both the quality ceiling of Flux and permanent style locking. This is the highest-quality option but requires the same LoRA training step.

---

### IMAGE-TO-VIDEO — Motion Layer

After style-locked frames exist, these tools animate them:

| Tool | Quality | I2V Support | API | Cost |
|------|---------|-------------|-----|------|
| **Kling AI 3.0** | Superior physics/motion | Yes | Yes, no waitlist | Free tier + paid |
| **Runway Gen-3 Alpha** | High, fast turbo mode | Yes | Yes | ~$12/month (625 credits) |
| **Minimax Video-01** | Good | Yes | Via Replicate | ~$0.01/second |
| **Wan 2.1 (1.3B)** | Good | Yes | Local | Free (needs 8GB VRAM) |
| **Wan 2.1 (14B)** | Best open-source | Yes | Local | Free (needs 24GB VRAM) |

**For AFTERIMAGE specifically: Kling AI 3.0.**

Confirmed by research: Kling 3.0 produces superior real-world physics and motion quality over Runway Gen-3. It outputs at 30fps. Its API has no waitlist. For near-static content — which is 90% of AFTERIMAGE shots — Kling's I2V handles the "almost-still" frame better.

**Runway advantage:** Faster turbo mode (5-second clips in under 30 seconds), structured camera presets. Use for rapid iteration when testing new shots.

**Wan 2.1 advantage:** Completely free if you have a GPU. The 1.3B model runs on 8GB VRAM (RTX 3080/4070 class). A 5-second 480p clip takes ~4 minutes on RTX 4090. Zero ongoing cost for batch production.

---

### POST-PRODUCTION — Style Completion

After generation, two post tools complete the visual identity:

**DaVinci Resolve (free)**
The AFTERIMAGE Power Grade — the teal-shadow/amber-highlight split, the grain, the chromatic aberration — is applied here to every clip before assembly. This is the final style lock. Even if the I2V output has slight color drift, the grade corrects it into the AFTERIMAGE palette.

The Entergalactic step-animation technique (18fps feel) is applied here: in the Speed Change menu, interpret clips at 18fps then conform to 24fps. No additional tool needed.

The anukitsu bloom effect (light sources that bleed into surrounding air) is applied here: Effects → Glow → applied to highlights only at 15–20% opacity.

**CapCut (free)**
Assembly and export. Direct TikTok and Instagram export built in.

---

## THE RECOMMENDED PIPELINE

### Tier 1 — Start This Week (No Setup)

```
Midjourney ($10/month)
  ↓ generate AFTERIMAGE reference frames (10-15 images)
  ↓ use best frames as --sref for all future generations
  ↓ use protagonist silhouette as --cref
  ↓ output: style-consistent still frames

Runway Gen-3 Alpha or Kling AI (free tiers)
  ↓ upload frame as Image-to-Video starting frame
  ↓ add minimal motion prompt
  ↓ output: animated clip, 3-5 seconds

DaVinci Resolve (free)
  ↓ apply AFTERIMAGE_MASTER Power Grade
  ↓ apply step-frame timing (18fps feel)
  ↓ apply bloom to light sources
  ↓ assemble and export
```

Cost: $10/month. No GPU. No training. No setup beyond Midjourney account.

---

### Tier 2 — Style Locked Permanently (After 10 Videos)

```
SDXL or Flux LoRA (trained on best AFTERIMAGE frames, ~$5 one-time)
  ↓ generates AFTERIMAGE frames automatically, no --sref needed
  ↓ runs via Replicate API or locally in ComfyUI

ComfyUI (free, local)
  ↓ orchestrates generation pipeline
  ↓ IP-Adapter holds protagonist silhouette across all generations
  ↓ ControlNet locks composition (figure scale, depth relationship)
  ↓ batch-generate 20 frames per session

Runway or Kling API
  ↓ programmatic I2V — no manual uploading

DaVinci Power Grade (automated via scripting)
```

Cost: ~$5 one-time for LoRA training + ~$0.01/generation. 100 videos ≈ $1.

---

### Tier 3 — Full Automation (When Pipeline Is Proven)

```
n8n workflow
  ↓ Claude generates episode concept + feeling tag
  ↓ n8n calls Replicate API (ComfyUI workflow) → generates frames
  ↓ n8n calls Kling/Runway API → animates frames
  ↓ output lands in /packages/EP-XXX/

Owner reviews package
  ↓ applies DaVinci grade (2 min)
  ↓ schedules in Buffer
```

Human time per video: 15–20 minutes.

---

## INTEGRATION WITH CLAUDE (MCP/API/n8n)

| Tool | Integration Method | What Claude Controls |
|------|------------------|---------------------|
| **Replicate** | HTTP API, n8n HTTP node | Triggers SDXL/Flux/LoRA generation, receives image URLs |
| **ComfyUI** | HTTP API (`--listen` flag) | Submits full generation workflows as JSON |
| **Kling AI** | HTTP API | Submits I2V jobs, polls for completion |
| **Runway** | HTTP API | Submits I2V jobs |
| **Midjourney** | Discord API (unofficial) or ImagineAPI (paid wrapper) | Less clean — better to keep Midjourney manual |

The automation-ready path: **Replicate + Kling** via n8n HTTP nodes, orchestrated by Claude. Both have clean REST APIs that n8n can call natively. No custom code required — just HTTP Request nodes.

ComfyUI API is the most powerful integration — you define the entire generation workflow as a JSON graph and submit it programmatically. n8n sends the workflow JSON, ComfyUI executes, returns image URLs. Claude writes the workflow JSON per episode.

---

## THE HONEST ASSESSMENT

**The Midjourney --sref path is the correct immediate action.**

Reason: you need to see whether AFTERIMAGE's visual identity can be achieved with AI tools at all before building automation infrastructure. Midjourney's --sref gives you the fastest answer. If 10 Midjourney generations with --sref produce frames that pass the "could only be AFTERIMAGE" test — then you build the LoRA, then you build the automation.

If they don't — the problem is the reference images, not the tools. The reference images need to come first.

**The order:**
1. Generate 10 Midjourney images using the Visual Bible as prompts (no --sref yet)
2. Find the 3-5 that come closest to the target aesthetic
3. Use those as --sref for the next 10 generations
4. If those pass the identity test: train the LoRA, build the pipeline
5. If they don't: the reference images need revision first

You cannot automate your way to a style that hasn't been manually found yet. Find the style first. Lock it second. Automate third.
