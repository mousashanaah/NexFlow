# AFTERIMAGE — n8n Workflow Setup
## Step-by-step build guide for the automated pipeline

---

## PREREQUISITES

1. n8n installed (self-hosted: `npx n8n` — free, runs locally)
   OR n8n cloud account (free tier: 5 active workflows, 2,500 executions/month)
2. Anthropic API key (you need this separately from Claude Pro — get at console.anthropic.com)
3. Notion API key + database IDs (set up the 4 databases first)
4. A folder: `~/afterimage-packages/` where output files will be saved

**Note on Claude Pro vs Claude API:** Claude Pro (your subscription) is for manual work at claude.ai. The pipeline uses the **Claude API** — separate billing, ~$0.001 per full pipeline run using Haiku. You need both. Get the API key at console.anthropic.com → API Keys.

---

## WORKFLOW STRUCTURE

Create one workflow in n8n called `AFTERIMAGE Episode Generator`.

### Node 1: Manual Trigger
- Type: Manual Trigger
- Purpose: You click "Run" to generate one episode package
- Configuration: Add one input field: `feeling_tag` (text)
  - This lets you type the feeling tag at trigger time, or leave blank to use automatic rotation

---

### Node 2: Read Episode Log
- Type: Notion → Get Database Items
- Database: `Episode Log`
- Filter: Sort by `posted_date` descending, limit 8
- Output: Last 8 episodes (for non-repetition check)

---

### Node 3: Read Anchor Usage
- Type: Notion → Get Database Items  
- Database: `Anchor Map`
- Output: All 6 anchors with `last_used_episode` field
- Use: Select the anchor least recently used (or longest since visited)

---

### Node 4: Stage 1 — Episode Generator
- Type: HTTP Request
- URL: `https://api.anthropic.com/v1/messages`
- Method: POST
- Headers:
  ```
  x-api-key: [YOUR_ANTHROPIC_API_KEY]
  anthropic-version: 2023-06-01
  content-type: application/json
  ```
- Body (JSON):
  ```json
  {
    "model": "claude-haiku-4-5-20251001",
    "max_tokens": 500,
    "system": "[FULL CONTENT OF UNIVERSE_CONTEXT.md — paste here]",
    "messages": [{
      "role": "user",
      "content": "Generate one AFTERIMAGE episode concept.\n\nFeeling tag: {{$json.feeling_tag || 'auto-select from library'}}\nAnchor: {{$node['Node 3'].json.least_used_anchor}}\nCassette state from last episode: {{$node['Node 2'].json[0].cassette_state}}\nRecent episodes to avoid repeating: {{$node['Node 2'].json.map(e => e.core_moment).join(' | ')}}\n\nOutput valid JSON only. Schema:\n{\"episode_id\": \"EP-XXX\", \"anchor\": \"...\", \"feeling_tag\": \"...\", \"core_moment\": \"...\", \"what_is_withheld\": \"...\", \"symbol_state\": \"held|found|left|missing|absent\", \"other_one_presence\": \"none|implied|trace\", \"emotional_arc\": \"...\"}"
    }]
  }
  ```
- Output: Parse `content[0].text` as JSON → episode_json

---

### Node 5: Stage 2 — Shot Generator
- Type: HTTP Request (same Anthropic endpoint)
- Body: Pass episode_json from Node 4
- User message:
  ```
  Break this episode into the 8-part AFTERIMAGE shot structure.
  
  Episode: {{$node['Node 4'].json}}
  
  Output a JSON array of 8 shot objects. Each object:
  {"part": 1, "duration_seconds": 2, "shot_description": "...", "camera_position": "...", "what_is_visible": "...", "what_is_NOT_visible": "...", "audio_note": "..."}
  ```
- Output: shot_array JSON

---

### Node 6: Stage 3 — Visual Prompt Generator
- Type: HTTP Request
- User message:
  ```
  Convert each shot into Runway ML Gen-3 / Pika video generation prompts.
  
  Shots: {{$node['Node 5'].json}}
  
  Apply to every prompt: "dark teal shadows, warm amber highlights, muted saturation, analog grain, practical lighting, neon reflections, wet surfaces, cinematic, no faces visible"
  
  Output JSON array: [{"part": 1, "runway_prompt": "...", "negative_prompt": "...", "style_tags": [...], "human_filming_note": "..."}]
  ```

---

### Node 7: Stage 4 — Video Production Brief
- Type: HTTP Request
- User message: Combine episode_json + shot_array into production brief
- Output: production_brief JSON

---

### Node 8: Stage 5 — Caption Generator
- Type: HTTP Request
- User message: Generate 3 caption options from episode_json
- Output: captions JSON

---

### Node 9: Stage 6 — Hashtag Generator
- Type: Notion read (get last 10 hashtag sets) → HTTP Request
- User message: Generate 3 hashtags, include #afterimage, avoid recent sets
- Output: hashtags JSON

---

### Node 10: Assemble Upload Package
- Type: Code (JavaScript)
- Function: Combine all outputs into human-readable markdown files
  ```javascript
  const episode = $node['Node 4'].json;
  const shots = $node['Node 5'].json;
  const prompts = $node['Node 6'].json;
  const brief = $node['Node 7'].json;
  const captions = $node['Node 8'].json;
  const hashtags = $node['Node 9'].json;
  
  const packageId = episode.episode_id;
  
  // Build production_brief.md content
  const briefMd = `# ${packageId} — Production Brief
  
  **Feeling:** ${episode.feeling_tag}
  **Anchor:** ${episode.anchor}
  **Core Moment:** ${episode.core_moment}
  **What Is Withheld:** ${episode.what_is_withheld}
  **Cassette:** ${episode.symbol_state}
  **The Other One:** ${episode.other_one_presence}
  
  ## Shot List
  ${shots.map(s => `**Part ${s.part} (${s.duration_seconds}s):** ${s.shot_description}\nCamera: ${s.camera_position}\nAudio: ${s.audio_note}`).join('\n\n')}
  
  ## Production Notes
  Mode: ${brief.production_mode}
  Music: ${brief.music_direction}
  Audio Decay: ${brief.audio_decay_instruction}
  `;
  
  const promptsTxt = prompts.map(p => 
    `PART ${p.part}:\nPrompt: ${p.runway_prompt}\nNegative: ${p.negative_prompt}\nIf filming: ${p.human_filming_note}`
  ).join('\n\n---\n\n');
  
  const captionTxt = `PRIMARY: ${captions.primary}\nALT 1: ${captions.alt_1}\nALT 2: ${captions.alt_2}`;
  const hashtagTxt = hashtags.hashtags.join(' ');
  
  return {packageId, briefMd, promptsTxt, captionTxt, hashtagTxt};
  ```

---

### Node 11: Write Files
- Type: Write Binary File (or use a Google Drive / Notion page creation node)
- Creates:
  - `/afterimage-packages/{episode_id}/production_brief.md`
  - `/afterimage-packages/{episode_id}/runway_prompts.txt`
  - `/afterimage-packages/{episode_id}/caption.txt`
  - `/afterimage-packages/{episode_id}/hashtags.txt`

---

### Node 12: Stage 7 — Lore DB Update
- Type: HTTP Request (Claude API)
- User message: Extract lore data from completed episode
- Output: lore_update JSON

---

### Node 13: Write to Notion Episode Log
- Type: Notion → Create Database Item
- Database: `Episode Log`
- Fields: Map all fields from episode_json + lore_update

---

### Node 14: Notify
- Type: Send Email (Gmail node, free) or Slack (if you use it)
- Message: "Upload package ready: {episode_id} — {anchor} — {feeling_tag} → ~/afterimage-packages/{episode_id}/"

---

## TOTAL API CALLS PER PIPELINE RUN

| Stage | Model | Est. Tokens | Est. Cost |
|-------|-------|-------------|-----------|
| Episode Generator | haiku | ~800 | $0.0001 |
| Shot Generator | haiku | ~1,200 | $0.0002 |
| Visual Prompt Generator | haiku | ~1,500 | $0.0002 |
| Video Brief | haiku | ~800 | $0.0001 |
| Caption Generator | haiku | ~400 | $0.00005 |
| Hashtag Generator | haiku | ~300 | $0.00004 |
| Lore DB Update | haiku | ~600 | $0.0001 |
| **TOTAL** | | **~5,600** | **~$0.001** |

100 videos = $0.10 in API costs.

---

## SETUP TIME ESTIMATE

| Task | Time |
|------|------|
| n8n install + account | 20 min |
| Notion databases setup | 30 min |
| Build n8n workflow (all 14 nodes) | 3-4 hours |
| Paste UNIVERSE_CONTEXT into system prompts | 30 min |
| Test run + prompt refinement | 1-2 hours |
| **Total** | **~6 hours (once)** |

After that: 60 seconds to run the pipeline, 30-40 minutes to produce and schedule the video.
