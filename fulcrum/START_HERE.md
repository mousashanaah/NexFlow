# Fulcrum — Start Here
## Everything you need. All free. In order.

---

## WHAT YOU HAVE

```
fulcrum/
├── START_HERE.md          ← you are here
├── scripts/
│   ├── fetch_opportunities.py   ← pulls live SAM.gov data (free API)
│   ├── generate_digest.py       ← generates weekly brief via Claude
│   └── send_digest.py           ← emails digest to subscribers (free Resend)
├── content/
│   ├── reddit_posts.md          ← copy-paste ready Reddit content
│   ├── twitter_posts.md         ← copy-paste ready Twitter/X threads
│   └── dm_templates.md          ← DM templates + customer discovery guide
├── ops/
│   ├── scoreboard.md            ← weekly validation tracker
│   └── subscribers.json         ← subscriber list (add real ones here)
└── digest/
    └── (weekly digests saved here)
```

---

## FREE TOOLS REQUIRED (setup takes ~30 minutes total)

| Tool | Cost | What For | Sign Up |
|------|------|----------|---------|
| SAM.gov API key | FREE | Pull live contract data | sam.gov/profile |
| Anthropic API | ~$5/month | Generate digests (~$0.05 each) | console.anthropic.com |
| Resend | FREE (3k emails/month) | Send digests | resend.com |
| Proton Mail | FREE | intel@fulcrumintel.com | proton.me |
| Reddit account | FREE | Post content | reddit.com |
| Twitter/X account | FREE | Post content | twitter.com |

**Total monthly cost at validation stage: ~$2-5 (Claude API only)**

---

## YOUR EXACT ACTION SEQUENCE

### TODAY (30 minutes)

**Step 1 — Get your free SAM.gov API key (10 min)**
1. Go to sam.gov/profile
2. Create a free account
3. Go to System Account Management → Create System Account
4. Request API key for "Opportunities"
5. Paste the key into `scripts/fetch_opportunities.py` line 12

**Step 2 — Get your free Anthropic API key (5 min)**
1. Go to console.anthropic.com
2. Create account → API Keys → Create Key
3. Set as environment variable: `export ANTHROPIC_API_KEY=your_key_here`
   OR paste directly into `scripts/generate_digest.py` line 14

**Step 3 — Get your free Resend API key (5 min)**
1. Go to resend.com → Sign up free
2. Add sending domain OR use their test domain for validation
3. Create API key → paste into `scripts/send_digest.py` line 14

**Step 4 — Create your identity (5 min)**
1. Go to proton.me → create free account → email: intel@fulcrumintel.com
2. Create Reddit account (new, not personal) — username: FulcrumIntel or similar
3. Create Twitter/X account — @FulcrumFederal or @FulcrumIntel

**Step 5 — Run the pipeline (5 min)**
```bash
cd fulcrum/scripts
python fetch_opportunities.py    # pulls real SAM.gov data → raw_data.json
python generate_digest.py        # generates digest → digest_YYYY-MM-DD.md
```
Open the digest. Replace any illustrative items with real SAM.gov solicitations.
Save as PDF (this is your validation asset).

---

### WEEK 1-2 (10-15 minutes/day)

**Comment-building phase on Reddit**
- Find SAM.gov frustration threads, Q4/pipeline threads, recompete threads
- Post 1-2 genuine helpful comments per day
- Use the 3 comment templates in `content/reddit_posts.md`
- Goal: 10-15 real comments before first product post
- Do NOT mention Fulcrum yet

---

### WEEK 2-3 (posts go live)

**Day 1: Post #1 on Reddit** (content/reddit_posts.md → POST #1)
**Day 3: Post #2 on Reddit** (content/reddit_posts.md → POST #2)
**Day 1-2: Twitter Thread #1** (content/twitter_posts.md → THREAD #1)
**Day 2-3: Twitter Thread #2** (content/twitter_posts.md → THREAD #2)

**Before first post: load all DM templates** (content/dm_templates.md)
First DM can arrive within hours. Have the response ready.

---

### EVERY MONDAY

```bash
cd fulcrum/scripts
python fetch_opportunities.py
python generate_digest.py
python send_digest.py
```

Three commands. Weekly digest sent automatically.

---

### EVERY FRIDAY

Open `ops/scoreboard.md` and fill in the week's numbers.
Check the decision gate for your current week.
Be honest. The scoreboard is only useful if the numbers are real.

---

## ADDING A SUBSCRIBER

When someone DMs asking for the digest:

1. Open `ops/subscribers.json`
2. Add their entry:
```json
{
  "email": "their@email.com",
  "name": "Their Name",
  "company": "Company LLC",
  "role": "Capture Manager",
  "status": "trial",
  "naics": ["541512"],
  "agencies": ["HHS", "VA"],
  "certifications": ["Small Business"],
  "signed_up": "2026-06-04",
  "trial_ends": "2026-06-18",
  "notes": "Found via Reddit post #1"
}
```
3. Change `"status": "trial"` to `"status": "active"` when they pay

---

## THE DIFFERENTIATION ANSWER

When anyone asks "how is this different from SAM.gov?" or "how is this different from getcapio.io?"

**vs SAM.gov:**
"SAM.gov is a database you navigate. Fulcrum is intelligence that arrives Monday morning, filtered for your specific agencies and NAICS codes, with context — not just raw solicitation links."

**vs getcapio.io:**
"getcapio is a platform you operate. Fulcrum is a curated weekly brief you read. Different buying decision, different use case. If you want to search, use getcapio. If you want intelligence delivered without logging in, that's Fulcrum."

---

## THE NUMBERS THAT MATTER

At Day 30:
- 1 paying customer → GO, build the full pipeline
- 3 WTP signals, 0 paying → test $49/month
- 0 WTP signals → kill this niche, try the next one

The architecture (scripts, templates, scoreboard) transfers to any intelligence niche.
Government contracts is Machine #1. Clinical trials, patents, or executive movement could be Machine #2.
