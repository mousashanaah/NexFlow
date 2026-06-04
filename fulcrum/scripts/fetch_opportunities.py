"""
Fulcrum — SAM.gov Opportunity Fetcher
Pulls real federal contract opportunities filtered for small IT contractors.
SAM.gov API is free. Get your key at: https://sam.gov/profile (takes 10 min)
"""

import json
import urllib.request
import urllib.parse
from datetime import datetime, timedelta

# --- CONFIG ---
SAM_API_KEY = "PASTE_YOUR_FREE_SAM_GOV_API_KEY_HERE"

TARGET_NAICS = ["541512", "541511", "541513", "541519", "541690"]

SET_ASIDE_FILTERS = [
    "Total Small Business",
    "Service-Disabled Veteran-Owned Small Business",
    "8(a) Program",
    "Women-Owned Small Business",
    "HUBZone",
]

DAYS_BACK = 21          # opportunities posted in last 3 weeks
MIN_VALUE = 500_000     # ignore micro-contracts
MAX_VALUE = 25_000_000  # ignore large primes


def fetch_opportunities(naics: str, days_back: int = DAYS_BACK) -> list:
    """Pull active opportunities for one NAICS code."""
    posted_from = (datetime.now() - timedelta(days=days_back)).strftime("%m/%d/%Y")
    posted_to = datetime.now().strftime("%m/%d/%Y")

    params = {
        "api_key": SAM_API_KEY,
        "limit": 100,
        "postedFrom": posted_from,
        "postedTo": posted_to,
        "naicsCode": naics,
        "active": "true",
    }

    url = "https://api.sam.gov/opportunities/v2/search?" + urllib.parse.urlencode(params)

    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            data = json.loads(response.read())
            return data.get("opportunitiesData", [])
    except Exception as e:
        print(f"  Error fetching NAICS {naics}: {e}")
        return []


def filter_opportunities(opportunities: list) -> list:
    """Keep only opportunities relevant to small IT contractors."""
    filtered = []

    for opp in opportunities:
        # Skip if no set-aside info (full and open only — harder to win)
        set_aside = opp.get("typeOfSetAsideDescription", "") or ""

        # Keep small business set-asides + full/open if value is in range
        value_str = opp.get("award", {}).get("amount", "") or ""
        try:
            value = float(str(value_str).replace(",", "").replace("$", ""))
            if value < MIN_VALUE or value > MAX_VALUE:
                continue
        except (ValueError, TypeError):
            pass  # no value listed yet — still include, common for pre-solicitations

        # Deduplicate by solicitation number
        sol_num = opp.get("solicitationNumber", "")
        if any(f.get("solicitationNumber") == sol_num for f in filtered):
            continue

        filtered.append({
            "title": opp.get("title", ""),
            "agency": opp.get("fullParentPathName", opp.get("organizationHierarchy", {}).get("name", "")),
            "solicitation_number": sol_num,
            "naics": opp.get("naicsCode", ""),
            "set_aside": set_aside,
            "type": opp.get("type", ""),
            "response_deadline": opp.get("responseDeadLine", ""),
            "posted_date": opp.get("postedDate", ""),
            "description": (opp.get("description", "") or "")[:500],
            "url": f"https://sam.gov/opp/{opp.get('noticeId', '')}/view",
            "notice_id": opp.get("noticeId", ""),
        })

    return filtered


def fetch_expiring_contracts(naics_list: list) -> list:
    """
    Pull contracts expiring in the next 90 days from USASpending.gov
    These are your recompete opportunities.
    """
    url = "https://api.usaspending.gov/api/v2/search/spending_by_award/"

    today = datetime.now().strftime("%Y-%m-%d")
    ninety_days = (datetime.now() + timedelta(days=90)).strftime("%Y-%m-%d")

    payload = json.dumps({
        "filters": {
            "award_type_codes": ["A", "B", "C", "D"],
            "naics_codes": naics_list,
            "time_period": [{"start_date": today, "end_date": ninety_days}],
            "award_amounts": [{"lower_bound": MIN_VALUE, "upper_bound": MAX_VALUE}],
        },
        "fields": [
            "Award ID",
            "Recipient Name",
            "Award Amount",
            "Period of Performance End Date",
            "awarding_agency_name",
            "Award Description",
        ],
        "sort": "Period of Performance End Date",
        "order": "asc",
        "limit": 20,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read())
            return data.get("results", [])
    except Exception as e:
        print(f"  Error fetching expiring contracts: {e}")
        return []


def run():
    print("Fulcrum — Fetching opportunities...\n")

    all_opportunities = []

    for naics in TARGET_NAICS:
        print(f"  Pulling NAICS {naics}...")
        opps = fetch_opportunities(naics)
        filtered = filter_opportunities(opps)
        print(f"    {len(opps)} total → {len(filtered)} after filter")
        all_opportunities.extend(filtered)

    # Deduplicate across NAICS codes
    seen = set()
    unique_opps = []
    for opp in all_opportunities:
        key = opp["solicitation_number"] or opp["notice_id"]
        if key and key not in seen:
            seen.add(key)
            unique_opps.append(opp)

    print(f"\n  Total unique opportunities: {len(unique_opps)}")

    print("\n  Fetching expiring contracts (recompete intelligence)...")
    expiring = fetch_expiring_contracts(TARGET_NAICS)
    print(f"  Found {len(expiring)} contracts expiring in next 90 days")

    # Save raw data
    output = {
        "generated_at": datetime.now().isoformat(),
        "opportunities": unique_opps,
        "expiring_contracts": expiring,
    }

    with open("raw_data.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n  Saved to raw_data.json")
    print("  Next step: run generate_digest.py to create this week's brief\n")

    return output


if __name__ == "__main__":
    run()
