"""
Fulcrum — Digest Generator
Takes raw opportunity data and generates the weekly intelligence brief.
Requires: Anthropic API key (get free credits at console.anthropic.com)
Cost per digest: ~$0.05 using claude-haiku-4-5 (essentially free at validation stage)
"""

import json
import os
import sys
from datetime import datetime
from anthropic import Anthropic

# --- CONFIG ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "PASTE_YOUR_KEY_HERE")

SUBSCRIBER_PROFILE = {
    "company_size": "10-50 employees",
    "certifications": ["Small Business", "SDVOSB"],  # edit to match subscriber
    "primary_agencies": ["HHS", "VA", "DHS", "DOJ"],
    "naics_codes": ["541512", "541511", "541513", "541519"],
    "contract_range": "$500K - $10M",
}

DIGEST_PROMPT = """You are producing a weekly government contract intelligence brief for Fulcrum.

Subscriber profile:
- Company size: {company_size}
- Certifications: {certifications}
- Primary agencies: {agencies}
- NAICS focus: {naics}
- Contract size range: {contract_range}

Here is this week's raw opportunity data from SAM.gov:
{opportunities}

Here is expiring contract data (recompete intelligence):
{expiring}

Produce a complete weekly intelligence digest in this exact format:

---
FULCRUM — FEDERAL CONTRACT INTELLIGENCE
IT Services | Small Business | Week of {week_date}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THIS WEEK AT A GLANCE
• [Most important item — specific agency, dollar amount, deadline]
• [Second most important — recompete or time-sensitive opportunity]
• [Market observation — pattern in this week's data]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HIGHLIGHTED OPPORTUNITIES

[Select the 3 best opportunities for this subscriber profile.
For each, include:
- Title, Agency, Solicitation Number
- Value | Contract Type | Set-Aside
- Response Deadline
- 2-3 sentence analysis: why it matters for a small IT firm,
  competitive landscape, one specific action to take this week]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXPIRING CONTRACT ALERT

[Identify the single best recompete opportunity from the expiring data.
Include: what it covers, current incumbent, expiration date,
estimated recompete value, why it's an opening, what to do this week]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MARKET PULSE
[One paragraph: a pattern or anomaly in this week's procurement data
that a BD professional should know. Be specific — cite numbers.]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FULL OPPORTUNITY TABLE

| Title | Agency | Value | Deadline | Set-Aside | NAICS |
[All matching opportunities sorted by deadline]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Filtered for: Federal civilian agencies | NAICS {naics} | Small business | $500K–$25M
Reply to update preferences | Unsubscribe
---

Requirements:
- Every opportunity must come from the provided data
- Analysis must be specific and actionable, not generic
- Tone: professional intelligence brief, not a newsletter
- Do not pad with filler or general GovCon advice
"""


def load_raw_data(filepath: str = "raw_data.json") -> dict:
    with open(filepath) as f:
        return json.load(f)


def format_opportunities_for_prompt(opportunities: list) -> str:
    if not opportunities:
        return "No opportunities available this week."

    lines = []
    for i, opp in enumerate(opportunities[:30], 1):  # cap at 30 for context
        lines.append(
            f"{i}. {opp['title']}\n"
            f"   Agency: {opp['agency']}\n"
            f"   Sol #: {opp['solicitation_number']}\n"
            f"   NAICS: {opp['naics']} | Set-aside: {opp['set_aside']}\n"
            f"   Deadline: {opp['response_deadline']}\n"
            f"   URL: {opp['url']}\n"
        )
    return "\n".join(lines)


def format_expiring_for_prompt(expiring: list) -> str:
    if not expiring:
        return "No expiring contracts found this week."

    lines = []
    for contract in expiring[:10]:
        lines.append(
            f"- {contract.get('Award Description', 'N/A')}\n"
            f"  Incumbent: {contract.get('Recipient Name', 'N/A')}\n"
            f"  Agency: {contract.get('awarding_agency_name', 'N/A')}\n"
            f"  Value: {contract.get('Award Amount', 'N/A')}\n"
            f"  Expires: {contract.get('Period of Performance End Date', 'N/A')}\n"
        )
    return "\n".join(lines)


def generate_digest(raw_data: dict, profile: dict = SUBSCRIBER_PROFILE) -> str:
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = DIGEST_PROMPT.format(
        company_size=profile["company_size"],
        certifications=", ".join(profile["certifications"]),
        agencies=", ".join(profile["primary_agencies"]),
        naics=", ".join(profile["naics_codes"]),
        contract_range=profile["contract_range"],
        week_date=datetime.now().strftime("%B %d, %Y"),
        opportunities=format_opportunities_for_prompt(raw_data.get("opportunities", [])),
        expiring=format_expiring_for_prompt(raw_data.get("expiring_contracts", [])),
    )

    # Using Haiku for cost efficiency (~$0.05 per digest)
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text


def save_digest(digest: str):
    week = datetime.now().strftime("%Y-%m-%d")
    filename = f"digest_{week}.md"
    with open(filename, "w") as f:
        f.write(digest)
    print(f"Digest saved to {filename}")
    return filename


def run():
    print("Fulcrum — Generating digest...\n")

    if not os.path.exists("raw_data.json"):
        print("Error: raw_data.json not found. Run fetch_opportunities.py first.")
        sys.exit(1)

    raw_data = load_raw_data()
    opp_count = len(raw_data.get("opportunities", []))
    exp_count = len(raw_data.get("expiring_contracts", []))
    print(f"  Loaded {opp_count} opportunities, {exp_count} expiring contracts")

    print("  Calling Claude API...")
    digest = generate_digest(raw_data)

    filename = save_digest(digest)

    print("\n--- DIGEST PREVIEW (first 500 chars) ---")
    print(digest[:500])
    print("---\n")
    print(f"Full digest in {filename}")
    print("Next step: email this to your pilot subscribers via Resend (free tier)\n")


if __name__ == "__main__":
    run()
