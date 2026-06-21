"""
V9 Confidence — Notification Channel

Thin, dependency-free notification layer.

Responsibilities — exactly three:
  1. Write every notification to the log file (always)
  2. Send email if NEXFLOW_NOTIFY_EMAIL is configured
  3. POST to webhook if NEXFLOW_NOTIFY_WEBHOOK is configured

Notification failure never crashes the bot.
If email/webhook fails, the failure is written to the log and execution continues.

Configuration (environment variables):
  NEXFLOW_NOTIFY_EMAIL    — recipient address (uses local sendmail/SMTP)
  NEXFLOW_NOTIFY_SMTP_HOST    — SMTP host     (default: localhost)
  NEXFLOW_NOTIFY_SMTP_PORT    — SMTP port     (default: 25)
  NEXFLOW_NOTIFY_SMTP_USER    — SMTP username (optional)
  NEXFLOW_NOTIFY_SMTP_PASS    — SMTP password (optional)
  NEXFLOW_NOTIFY_FROM_EMAIL   — sender address (default: nexflow@localhost)
  NEXFLOW_NOTIFY_WEBHOOK  — URL to POST JSON payload to (Slack, Discord, etc.)
  NEXFLOW_NOTIFY_LOG      — log file path (default: /var/nexflow/notify.log)

Levels:
  INFO    — normal operation events (rebalance executed, gate updated)
  WARNING — degraded but not blocked (parity borderline, data lag)
  ERROR   — blocked execution (parity failed, reconciliation failed)
  FATAL   — system cannot proceed (state corrupt, unrecoverable)
"""
from __future__ import annotations

import json
import os
import smtplib
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional


# ── Levels ────────────────────────────────────────────────────────────────────

class Level:
    INFO    = "INFO"
    WARNING = "WARNING"
    ERROR   = "ERROR"
    FATAL   = "FATAL"

    _NOTIFY_THRESHOLD = {INFO, WARNING, ERROR, FATAL}
    _EMAIL_THRESHOLD  = {WARNING, ERROR, FATAL}


# ── Notification ──────────────────────────────────────────────────────────────

@dataclass
class Notification:
    subject: str
    body:    str
    level:   str        = Level.INFO
    date:    str        = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    def to_dict(self) -> dict:
        return {
            "level":   self.level,
            "date":    self.date,
            "subject": self.subject,
            "body":    self.body,
            "sent_at": datetime.now(timezone.utc).isoformat(),
        }

    def should_email(self) -> bool:
        return self.level in Level._EMAIL_THRESHOLD

    def format_text(self) -> str:
        return f"[{self.level}] {self.subject}\n\n{self.body}"


# ── Notifier ──────────────────────────────────────────────────────────────────

class Notifier:
    """
    Sends notifications via configured channels.

    All failures are swallowed and written to the log.
    A broken notification channel must never stop the bot.
    """

    def __init__(
        self,
        log_path:   Optional[Path] = None,
        email_to:   Optional[str]  = None,
        webhook_url: Optional[str] = None,
    ) -> None:
        default_log = Path(os.environ.get("NEXFLOW_NOTIFY_LOG", "/var/nexflow/notify.log"))
        self._log_path   = log_path   or default_log
        self._email_to   = email_to   or os.environ.get("NEXFLOW_NOTIFY_EMAIL")
        self._webhook    = webhook_url or os.environ.get("NEXFLOW_NOTIFY_WEBHOOK")

    def send(self, notification: Notification) -> None:
        """Send notification to all configured channels. Never raises."""
        self._write_log(notification)
        if self._email_to and notification.should_email():
            self._send_email(notification)
        if self._webhook:
            self._send_webhook(notification)

    def info(self, subject: str, body: str = "") -> None:
        self.send(Notification(subject=subject, body=body, level=Level.INFO))

    def warning(self, subject: str, body: str = "") -> None:
        self.send(Notification(subject=subject, body=body, level=Level.WARNING))

    def error(self, subject: str, body: str = "") -> None:
        self.send(Notification(subject=subject, body=body, level=Level.ERROR))

    def fatal(self, subject: str, body: str = "") -> None:
        self.send(Notification(subject=subject, body=body, level=Level.FATAL))

    # ── Channels ──────────────────────────────────────────────────────────────

    def _write_log(self, n: Notification) -> None:
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "a") as f:
                f.write(json.dumps(n.to_dict()) + "\n")
        except Exception:
            pass   # log write failure is not recoverable; silently skip

    def _send_email(self, n: Notification) -> None:
        try:
            host = os.environ.get("NEXFLOW_NOTIFY_SMTP_HOST", "localhost")
            port = int(os.environ.get("NEXFLOW_NOTIFY_SMTP_PORT", "25"))
            user = os.environ.get("NEXFLOW_NOTIFY_SMTP_USER")
            pw   = os.environ.get("NEXFLOW_NOTIFY_SMTP_PASS")
            from_addr = os.environ.get("NEXFLOW_NOTIFY_FROM_EMAIL", "nexflow@localhost")

            msg = MIMEText(n.format_text())
            msg["Subject"] = f"[NexFlow V9] {n.subject}"
            msg["From"]    = from_addr
            msg["To"]      = self._email_to

            with smtplib.SMTP(host, port, timeout=10) as smtp:
                if user and pw:
                    smtp.login(user, pw)
                smtp.sendmail(from_addr, [self._email_to], msg.as_string())

        except Exception as exc:
            self._write_log(Notification(
                subject = "Email delivery failed",
                body    = str(exc),
                level   = Level.WARNING,
            ))

    def _send_webhook(self, n: Notification) -> None:
        try:
            payload = json.dumps({
                "text": f"*[NexFlow V9 {n.level}]* {n.subject}\n{n.body}"
            }).encode("utf-8")
            req = urllib.request.Request(
                self._webhook,
                data    = payload,
                headers = {"Content-Type": "application/json"},
                method  = "POST",
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception as exc:
            self._write_log(Notification(
                subject = "Webhook delivery failed",
                body    = str(exc),
                level   = Level.WARNING,
            ))

    def load_log(self) -> list[dict]:
        """Return all logged notifications in chronological order."""
        if not self._log_path.exists():
            return []
        out = []
        with open(self._log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return out
