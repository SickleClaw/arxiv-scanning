"""Optional SMTP delivery behind an injectable provider interface."""

from __future__ import annotations

import smtplib
import ssl
from contextlib import suppress
from dataclasses import dataclass
from datetime import date
from email.message import EmailMessage
from pathlib import Path
from typing import Protocol

from arxiv_digest.config import DeliveryConfig
from arxiv_digest.exceptions import ConfigurationError, DeliveryError


class DeliveryProvider(Protocol):
    """Deliver one already-rendered report without changing report state."""

    def send(self, markdown_path: Path, html_path: Path, *, report_date: date) -> DeliveryResult:
        """Send the multipart report or raise an actionable delivery error."""


@dataclass(frozen=True, slots=True)
class DeliveryResult:
    """Non-sensitive result metadata for one successful delivery."""

    recipient_count: int


def validate_delivery_ready(config: DeliveryConfig) -> None:
    """Require the settings needed for an explicitly requested SMTP send."""
    missing: list[str] = []
    if config.smtp_host is None:
        missing.append("SMTP_HOST")
    if config.from_email is None:
        missing.append("DIGEST_FROM_EMAIL")
    if not config.to_emails:
        missing.append("DIGEST_TO_EMAIL")
    if missing:
        raise ConfigurationError(
            "SMTP delivery is incomplete; configure " + ", ".join(missing) + "."
        )


class SMTPDeliveryProvider:
    """Send multipart plain-text/HTML reports with bounded standard-library SMTP I/O."""

    def __init__(self, config: DeliveryConfig) -> None:
        validate_delivery_ready(config)
        if config.smtp_host is None:
            raise ConfigurationError("SMTP_HOST is required for SMTP delivery")
        self._config = config
        self._smtp_host = config.smtp_host

    def _message(self, markdown: str, html: str, report_date: date) -> EmailMessage:
        message = EmailMessage()
        message["Subject"] = f"Weekly arXiv Digest — {report_date.isoformat()}"
        message["From"] = self._config.from_email
        message["To"] = ", ".join(self._config.to_emails)
        message.set_content(markdown)
        message.add_alternative(html, subtype="html")
        return message

    def send(self, markdown_path: Path, html_path: Path, *, report_date: date) -> DeliveryResult:
        """Read completed reports, send them once, and always close the SMTP client."""
        try:
            markdown = markdown_path.read_text(encoding="utf-8")
            html = html_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise DeliveryError(
                f"Cannot read completed report for email delivery ({type(exc).__name__})."
            ) from exc

        client: smtplib.SMTP | smtplib.SMTP_SSL | None = None
        try:
            context = ssl.create_default_context()
            if self._config.smtp_use_ssl:
                client = smtplib.SMTP_SSL(
                    self._smtp_host,
                    self._config.smtp_port,
                    timeout=self._config.timeout_seconds,
                    context=context,
                )
            else:
                client = smtplib.SMTP(
                    self._smtp_host,
                    self._config.smtp_port,
                    timeout=self._config.timeout_seconds,
                )
                if self._config.smtp_use_tls:
                    client.starttls(context=context)
                    client.ehlo()
            if self._config.smtp_username is not None:
                client.login(self._config.smtp_username, self._config.smtp_password or "")
            client.send_message(
                self._message(markdown, html, report_date),
                from_addr=self._config.from_email,
                to_addrs=self._config.to_emails,
            )
        except (OSError, smtplib.SMTPException) as exc:
            raise DeliveryError(
                f"SMTP delivery failed ({type(exc).__name__}); completed reports remain available."
            ) from exc
        finally:
            if client is not None:
                try:
                    client.quit()
                except (OSError, smtplib.SMTPException):
                    with suppress(OSError, smtplib.SMTPException):
                        client.close()
        return DeliveryResult(recipient_count=len(self._config.to_emails))
