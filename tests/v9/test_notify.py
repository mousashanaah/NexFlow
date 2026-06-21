"""
Tests for nexflow.v9.notify — Notification Channel
"""
from __future__ import annotations

import json
import smtplib
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nexflow.v9.notify import Level, Notification, Notifier


# ── Level ─────────────────────────────────────────────────────────────────────

class TestLevel:
    def test_constants_are_strings(self):
        assert Level.INFO    == "INFO"
        assert Level.WARNING == "WARNING"
        assert Level.ERROR   == "ERROR"
        assert Level.FATAL   == "FATAL"

    def test_email_threshold_excludes_info(self):
        assert Level.INFO    not in Level._EMAIL_THRESHOLD
        assert Level.WARNING in  Level._EMAIL_THRESHOLD
        assert Level.ERROR   in  Level._EMAIL_THRESHOLD
        assert Level.FATAL   in  Level._EMAIL_THRESHOLD

    def test_notify_threshold_includes_all(self):
        for lvl in (Level.INFO, Level.WARNING, Level.ERROR, Level.FATAL):
            assert lvl in Level._NOTIFY_THRESHOLD


# ── Notification ──────────────────────────────────────────────────────────────

class TestNotification:
    def _make(self, level=Level.INFO) -> Notification:
        return Notification(subject="Test subject", body="Test body", level=level)

    def test_to_dict_keys(self):
        n = self._make()
        d = n.to_dict()
        assert set(d) == {"level", "date", "subject", "body", "sent_at"}

    def test_to_dict_values(self):
        n = self._make(Level.ERROR)
        d = n.to_dict()
        assert d["level"]   == "ERROR"
        assert d["subject"] == "Test subject"
        assert d["body"]    == "Test body"

    def test_format_text(self):
        n = self._make(Level.WARNING)
        assert n.format_text() == "[WARNING] Test subject\n\nTest body"

    def test_should_email_info_false(self):
        assert not self._make(Level.INFO).should_email()

    def test_should_email_warning_true(self):
        assert self._make(Level.WARNING).should_email()

    def test_should_email_error_true(self):
        assert self._make(Level.ERROR).should_email()

    def test_should_email_fatal_true(self):
        assert self._make(Level.FATAL).should_email()

    def test_date_field_defaults_to_today(self):
        n = self._make()
        import re
        assert re.match(r"\d{4}-\d{2}-\d{2}", n.date)


# ── Notifier — log writing ────────────────────────────────────────────────────

class TestNotifierLog:
    def test_write_log_creates_file(self, tmp_path):
        log = tmp_path / "notify.log"
        n = Notifier(log_path=log)
        n.info("hello", "world")
        assert log.exists()

    def test_write_log_valid_json(self, tmp_path):
        log = tmp_path / "notify.log"
        n = Notifier(log_path=log)
        n.warning("w subject", "w body")
        lines = log.read_text().strip().splitlines()
        assert len(lines) == 1
        d = json.loads(lines[0])
        assert d["subject"] == "w subject"
        assert d["level"]   == "WARNING"

    def test_write_log_multiple_entries(self, tmp_path):
        log = tmp_path / "notify.log"
        n = Notifier(log_path=log)
        n.info("a", "")
        n.error("b", "")
        n.fatal("c", "")
        lines = log.read_text().strip().splitlines()
        assert len(lines) == 3

    def test_write_log_creates_parent_dirs(self, tmp_path):
        log = tmp_path / "deep" / "dir" / "notify.log"
        n = Notifier(log_path=log)
        n.info("hi", "")
        assert log.exists()

    def test_load_log_empty_when_no_file(self, tmp_path):
        n = Notifier(log_path=tmp_path / "missing.log")
        assert n.load_log() == []

    def test_load_log_returns_records(self, tmp_path):
        log = tmp_path / "notify.log"
        n = Notifier(log_path=log)
        n.info("a", "body_a")
        n.warning("b", "body_b")
        records = n.load_log()
        assert len(records) == 2
        assert records[0]["subject"] == "a"
        assert records[1]["subject"] == "b"

    def test_load_log_skips_corrupt_lines(self, tmp_path):
        log = tmp_path / "notify.log"
        log.write_text('{"subject": "good"}\nnot json\n{"subject": "also good"}\n')
        n = Notifier(log_path=log)
        records = n.load_log()
        assert len(records) == 2

    def test_log_write_failure_does_not_raise(self, tmp_path):
        # Point log at an unwritable path
        log = tmp_path / "readonly_dir" / "notify.log"
        log.parent.mkdir()
        log.parent.chmod(0o444)
        n = Notifier(log_path=log)
        try:
            n.info("test", "")   # must not raise
        finally:
            log.parent.chmod(0o755)


# ── Notifier — convenience methods ───────────────────────────────────────────

class TestNotifierConvenienceMethods:
    def test_info_writes_info_level(self, tmp_path):
        log = tmp_path / "n.log"
        n = Notifier(log_path=log)
        n.info("subj", "body")
        d = json.loads(log.read_text().strip())
        assert d["level"] == "INFO"

    def test_warning_writes_warning_level(self, tmp_path):
        log = tmp_path / "n.log"
        n = Notifier(log_path=log)
        n.warning("subj", "body")
        d = json.loads(log.read_text().strip())
        assert d["level"] == "WARNING"

    def test_error_writes_error_level(self, tmp_path):
        log = tmp_path / "n.log"
        n = Notifier(log_path=log)
        n.error("subj", "body")
        d = json.loads(log.read_text().strip())
        assert d["level"] == "ERROR"

    def test_fatal_writes_fatal_level(self, tmp_path):
        log = tmp_path / "n.log"
        n = Notifier(log_path=log)
        n.fatal("subj", "body")
        d = json.loads(log.read_text().strip())
        assert d["level"] == "FATAL"


# ── Notifier — email ─────────────────────────────────────────────────────────

class TestNotifierEmail:
    def test_email_not_sent_for_info(self, tmp_path):
        n = Notifier(log_path=tmp_path / "n.log", email_to="x@example.com")
        with patch("smtplib.SMTP") as mock_smtp:
            n.info("subject", "body")
        mock_smtp.assert_not_called()

    def test_email_sent_for_warning(self, tmp_path):
        n = Notifier(log_path=tmp_path / "n.log", email_to="x@example.com")
        mock_server = MagicMock()
        mock_smtp_cls = MagicMock(return_value=mock_server)
        mock_server.__enter__ = MagicMock(return_value=mock_server)
        mock_server.__exit__ = MagicMock(return_value=False)
        with patch("smtplib.SMTP", mock_smtp_cls):
            n.warning("subject", "body")
        mock_smtp_cls.assert_called_once()

    def test_email_sent_for_fatal(self, tmp_path):
        n = Notifier(log_path=tmp_path / "n.log", email_to="x@example.com")
        mock_server = MagicMock()
        mock_smtp_cls = MagicMock(return_value=mock_server)
        mock_server.__enter__ = MagicMock(return_value=mock_server)
        mock_server.__exit__ = MagicMock(return_value=False)
        with patch("smtplib.SMTP", mock_smtp_cls):
            n.fatal("subject", "body")
        mock_smtp_cls.assert_called_once()

    def test_email_not_sent_when_no_email_configured(self, tmp_path):
        n = Notifier(log_path=tmp_path / "n.log")
        with patch("smtplib.SMTP") as mock_smtp:
            n.error("subject", "body")
        mock_smtp.assert_not_called()

    def test_email_failure_does_not_raise(self, tmp_path):
        log = tmp_path / "n.log"
        n = Notifier(log_path=log, email_to="x@example.com")
        with patch("smtplib.SMTP", side_effect=smtplib.SMTPException("connection refused")):
            n.error("subject", "body")   # must not raise
        # Failure should be written to log
        records = n.load_log()
        # First record = original error; second = email delivery failure
        assert len(records) == 2
        assert records[1]["subject"] == "Email delivery failed"

    def test_email_failure_written_to_log(self, tmp_path):
        log = tmp_path / "n.log"
        n = Notifier(log_path=log, email_to="x@example.com")
        with patch("smtplib.SMTP", side_effect=Exception("boom")):
            n.warning("test", "body")
        records = n.load_log()
        failure_records = [r for r in records if r["subject"] == "Email delivery failed"]
        assert len(failure_records) == 1


# ── Notifier — webhook ────────────────────────────────────────────────────────

class TestNotifierWebhook:
    def test_webhook_called_on_info(self, tmp_path):
        n = Notifier(log_path=tmp_path / "n.log", webhook_url="http://localhost/hook")
        with patch("urllib.request.urlopen") as mock_open:
            n.info("subject", "body")
        mock_open.assert_called_once()

    def test_webhook_called_on_all_levels(self, tmp_path):
        n = Notifier(log_path=tmp_path / "n.log", webhook_url="http://localhost/hook")
        with patch("urllib.request.urlopen") as mock_open:
            n.info("a", "")
            n.warning("b", "")
            n.error("c", "")
            n.fatal("d", "")
        assert mock_open.call_count == 4

    def test_webhook_not_called_when_not_configured(self, tmp_path):
        n = Notifier(log_path=tmp_path / "n.log")
        with patch("urllib.request.urlopen") as mock_open:
            n.fatal("subject", "body")
        mock_open.assert_not_called()

    def test_webhook_failure_does_not_raise(self, tmp_path):
        log = tmp_path / "n.log"
        n = Notifier(log_path=log, webhook_url="http://localhost/hook")
        with patch("urllib.request.urlopen", side_effect=Exception("network error")):
            n.info("subject", "body")   # must not raise

    def test_webhook_failure_written_to_log(self, tmp_path):
        log = tmp_path / "n.log"
        n = Notifier(log_path=log, webhook_url="http://localhost/hook")
        with patch("urllib.request.urlopen", side_effect=Exception("timeout")):
            n.info("test", "body")
        records = n.load_log()
        failure_records = [r for r in records if r["subject"] == "Webhook delivery failed"]
        assert len(failure_records) == 1

    def test_webhook_payload_is_valid_json(self, tmp_path):
        n = Notifier(log_path=tmp_path / "n.log", webhook_url="http://localhost/hook")
        captured = []
        def fake_urlopen(req, timeout=None):
            captured.append(req.data)
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            n.error("my subject", "my body")
        assert len(captured) == 1
        payload = json.loads(captured[0].decode("utf-8"))
        assert "text" in payload
        assert "my subject" in payload["text"]


# ── Notifier — env-var configuration ─────────────────────────────────────────

class TestNotifierEnvConfig:
    def test_reads_email_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NEXFLOW_NOTIFY_EMAIL", "env@example.com")
        n = Notifier(log_path=tmp_path / "n.log")
        assert n._email_to == "env@example.com"

    def test_reads_webhook_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NEXFLOW_NOTIFY_WEBHOOK", "http://hook.example.com")
        n = Notifier(log_path=tmp_path / "n.log")
        assert n._webhook == "http://hook.example.com"

    def test_explicit_args_override_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NEXFLOW_NOTIFY_EMAIL", "env@example.com")
        n = Notifier(log_path=tmp_path / "n.log", email_to="override@example.com")
        assert n._email_to == "override@example.com"

    def test_reads_log_path_from_env(self, tmp_path, monkeypatch):
        log = tmp_path / "custom.log"
        monkeypatch.setenv("NEXFLOW_NOTIFY_LOG", str(log))
        n = Notifier()
        assert n._log_path == log
