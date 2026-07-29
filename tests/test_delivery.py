"""Mocked SMTP delivery tests with no network or credentials."""

from __future__ import annotations

import smtplib
from datetime import date
from email.message import EmailMessage
from pathlib import Path

import pytest

from arxiv_digest.config import DeliveryConfig
from arxiv_digest.delivery import SMTPDeliveryProvider, validate_delivery_ready
from arxiv_digest.exceptions import ConfigurationError, DeliveryError


class FakeSMTP:
    """Record SMTP operations while implementing the small provider surface."""

    def __init__(self, host: str, port: int, **kwargs: object) -> None:
        self.host = host
        self.port = port
        self.kwargs = kwargs
        self.calls: list[str] = []
        self.message: EmailMessage | None = None
        self.from_addr: str | None = None
        self.to_addrs: list[str] = []

    def starttls(self, *, context: object) -> None:
        assert context is not None
        self.calls.append("starttls")

    def ehlo(self) -> None:
        self.calls.append("ehlo")

    def login(self, username: str, password: str) -> None:
        assert username and password
        self.calls.append("login")

    def send_message(
        self,
        message: EmailMessage,
        *,
        from_addr: str,
        to_addrs: list[str],
    ) -> None:
        self.calls.append("send_message")
        self.message = message
        self.from_addr = from_addr
        self.to_addrs = to_addrs

    def quit(self) -> None:
        self.calls.append("quit")

    def close(self) -> None:
        self.calls.append("close")


def _config(**overrides: object) -> DeliveryConfig:
    values: dict[str, object] = {
        "smtp_host": "smtp.test.invalid",
        "smtp_port": 587,
        "smtp_username": "digest-user",
        "smtp_password": "not-a-real-password",
        "smtp_use_tls": True,
        "smtp_use_ssl": False,
        "from_email": "digest@test.invalid",
        "to_emails": ["one@test.invalid", "two@test.invalid"],
        "timeout_seconds": 12.0,
    }
    values.update(overrides)
    return DeliveryConfig.model_validate(values)


def _reports(tmp_path: Path) -> tuple[Path, Path]:
    markdown = tmp_path / "latest.md"
    html = tmp_path / "latest.html"
    markdown.write_text("# Weekly digest\n\nPlain report.", encoding="utf-8")
    html.write_text("<h1>Weekly digest</h1><p>HTML report.</p>", encoding="utf-8")
    return markdown, html


def test_starttls_delivery_is_multipart_authenticated_and_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    created: list[FakeSMTP] = []

    def factory(host: str, port: int, **kwargs: object) -> FakeSMTP:
        created.append(FakeSMTP(host, port, **kwargs))
        return created[-1]

    monkeypatch.setattr(smtplib, "SMTP", factory)
    markdown, html = _reports(tmp_path)
    result = SMTPDeliveryProvider(_config()).send(markdown, html, report_date=date(2026, 7, 29))
    client = created[0]
    assert result.recipient_count == 2
    assert client.calls == ["starttls", "ehlo", "login", "send_message", "quit"]
    assert client.kwargs["timeout"] == 12.0
    assert client.message is not None and client.message.is_multipart()
    assert client.message["Subject"] == "Weekly arXiv Digest — 2026-07-29"
    assert client.message.get_body(preferencelist=("plain",)) is not None
    assert client.message.get_body(preferencelist=("html",)) is not None
    assert client.to_addrs == ["one@test.invalid", "two@test.invalid"]


def test_implicit_ssl_uses_ssl_factory_without_starttls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    created: list[FakeSMTP] = []

    def factory(host: str, port: int, **kwargs: object) -> FakeSMTP:
        created.append(FakeSMTP(host, port, **kwargs))
        return created[-1]

    monkeypatch.setattr(smtplib, "SMTP_SSL", factory)
    markdown, html = _reports(tmp_path)
    SMTPDeliveryProvider(_config(smtp_port=465, smtp_use_tls=False, smtp_use_ssl=True)).send(
        markdown, html, report_date=date(2026, 7, 29)
    )
    assert "context" in created[0].kwargs
    assert "starttls" not in created[0].calls
    assert created[0].calls[-1] == "quit"


def test_authentication_failure_is_safe_and_still_closes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = FakeSMTP("smtp.test.invalid", 587)

    def fail_login(_username: str, _password: str) -> None:
        raise smtplib.SMTPAuthenticationError(535, b"credential rejected")

    client.login = fail_login  # type: ignore[method-assign]
    monkeypatch.setattr(smtplib, "SMTP", lambda *_args, **_kwargs: client)
    markdown, html = _reports(tmp_path)
    with pytest.raises(DeliveryError, match="SMTPAuthenticationError") as captured:
        SMTPDeliveryProvider(_config()).send(markdown, html, report_date=date(2026, 7, 29))
    assert "not-a-real-password" not in str(captured.value)
    assert "one@test.invalid" not in str(captured.value)
    assert client.calls[-1] == "quit"


def test_connection_timeout_and_missing_report_are_actionable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        smtplib,
        "SMTP",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("timed out")),
    )
    markdown, html = _reports(tmp_path)
    with pytest.raises(DeliveryError, match="TimeoutError"):
        SMTPDeliveryProvider(_config()).send(markdown, html, report_date=date(2026, 7, 29))
    with pytest.raises(DeliveryError, match="Cannot read completed report"):
        SMTPDeliveryProvider(_config()).send(
            tmp_path / "missing.md", html, report_date=date(2026, 7, 29)
        )


def test_cleanup_falls_back_to_close_when_quit_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = FakeSMTP("smtp.test.invalid", 587)

    def fail_quit() -> None:
        client.calls.append("quit")
        raise smtplib.SMTPServerDisconnected("already disconnected")

    def fail_close() -> None:
        client.calls.append("close")
        raise OSError("socket already closed")

    client.quit = fail_quit  # type: ignore[method-assign]
    client.close = fail_close  # type: ignore[method-assign]
    monkeypatch.setattr(smtplib, "SMTP", lambda *_args, **_kwargs: client)
    markdown, html = _reports(tmp_path)
    SMTPDeliveryProvider(_config()).send(markdown, html, report_date=date(2026, 7, 29))
    assert client.calls[-2:] == ["quit", "close"]


def test_delivery_configuration_rejects_unsafe_or_incomplete_values() -> None:
    with pytest.raises(ValueError, match="cannot both"):
        _config(smtp_use_tls=True, smtp_use_ssl=True)
    with pytest.raises(ValueError, match="configured together"):
        _config(smtp_password=None)
    with pytest.raises(ValueError, match="email-shaped"):
        _config(to_emails=["not-an-email"])
    with pytest.raises(ConfigurationError, match="SMTP_HOST"):
        validate_delivery_ready(DeliveryConfig())
