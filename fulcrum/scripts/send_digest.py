"""
Fulcrum — Digest Sender
Sends the weekly digest to all active subscribers via Resend (free tier: 3,000/month)
Get your free Resend API key at: https://resend.com (free, no credit card)
"""

import json
import os
import sys
from datetime import datetime
import urllib.request
import urllib.parse


RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "PASTE_YOUR_FREE_RESEND_KEY_HERE")
FROM_EMAIL = "intel@fulcrumintel.com"   # set up in Resend dashboard
FROM_NAME = "Fulcrum Federal Intelligence"
SUBJECT_TEMPLATE = "Federal Contract Intelligence — Week of {date}"


def load_subscribers(filepath: str = "../ops/subscribers.json") -> list:
    """Load subscriber list. Each subscriber has email + profile."""
    if not os.path.exists(filepath):
        # Bootstrap with empty list
        print(f"No subscribers file found at {filepath}. Creating empty list.")
        with open(filepath, "w") as f:
            json.dump([], f, indent=2)
        return []

    with open(filepath) as f:
        return json.load(f)


def load_digest(filepath: str = None) -> str:
    """Load the most recent digest file."""
    if filepath:
        with open(filepath) as f:
            return f.read()

    # Find most recent digest
    import glob
    files = sorted(glob.glob("digest_*.md"), reverse=True)
    if not files:
        print("No digest file found. Run generate_digest.py first.")
        sys.exit(1)

    with open(files[0]) as f:
        return f.read()


def markdown_to_html(markdown: str) -> str:
    """
    Simple markdown → HTML conversion for email.
    Keeps formatting readable in email clients.
    """
    html = markdown
    html = html.replace("━" * 39, "<hr style='border: 1px solid #333; margin: 20px 0;'>")
    html = html.replace("\n\n", "</p><p>")
    html = html.replace("\n", "<br>")
    html = f"""
    <div style="font-family: 'Courier New', monospace; max-width: 680px;
                margin: 0 auto; padding: 20px; background: #fff; color: #111;">
        <div style="background: #111; color: #fff; padding: 16px;
                    font-size: 18px; font-weight: bold; letter-spacing: 2px;">
            FULCRUM
        </div>
        <div style="padding: 20px; font-size: 14px; line-height: 1.7;">
            <p>{html}</p>
        </div>
        <div style="padding: 16px; font-size: 11px; color: #999; border-top: 1px solid #eee;">
            You're receiving this because you signed up for Fulcrum Federal Intelligence.<br>
            <a href="{{unsubscribe_url}}">Unsubscribe</a>
        </div>
    </div>
    """
    return html


def send_email(to_email: str, subject: str, html_content: str, text_content: str) -> bool:
    """Send one email via Resend API."""
    payload = json.dumps({
        "from": f"{FROM_NAME} <{FROM_EMAIL}>",
        "to": [to_email],
        "subject": subject,
        "html": html_content,
        "text": text_content,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=payload,
        headers={
            "Authorization": f"Bearer {RESEND_API_KEY}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read())
            return True
    except urllib.error.HTTPError as e:
        print(f"    Error sending to {to_email}: {e.code} {e.reason}")
        return False
    except Exception as e:
        print(f"    Error sending to {to_email}: {e}")
        return False


def run():
    print("Fulcrum — Sending weekly digest...\n")

    subscribers = load_subscribers()
    active = [s for s in subscribers if s.get("status") == "active"]

    print(f"  Subscribers: {len(subscribers)} total, {len(active)} active")

    if not active:
        print("  No active subscribers yet.")
        print("  Add subscribers to fulcrum/ops/subscribers.json")
        print("  Format: [{\"email\": \"name@co.com\", \"name\": \"Name\", "
              "\"status\": \"active\", \"naics\": [\"541512\"], \"agencies\": [\"HHS\"]}]")
        return

    digest_text = load_digest()
    week_date = datetime.now().strftime("%B %d, %Y")
    subject = SUBJECT_TEMPLATE.format(date=week_date)
    html_content = markdown_to_html(digest_text)

    sent = 0
    failed = 0

    for subscriber in active:
        email = subscriber.get("email")
        if not email:
            continue

        print(f"  Sending to {email}...")
        success = send_email(email, subject, html_content, digest_text)

        if success:
            sent += 1
            print(f"    ✓ Sent")
        else:
            failed += 1

    print(f"\n  Done. Sent: {sent} | Failed: {failed}")


if __name__ == "__main__":
    run()
