"""Sending generated emails over SMTP.

Deliberately separate from generation. Nothing here knows how an email was
written, and nothing in the generator knows how it is delivered - which is
what lets the demo (everything to one address) become production (each row to
its own recruiter) without touching either side.

Every credential comes from the environment. No address, host, password or
account name appears in this file, and none should ever be added to it.
"""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
import time
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import formataddr, parseaddr
from typing import Any, Callable, Optional

logger = logging.getLogger("ats.sender")

# A gap between messages. A hundred emails delivered as fast as the socket
# allows looks like spam to the provider and gets throttled or blocked; spread
# over a few minutes it does not.
DEFAULT_SEND_DELAY_SECONDS = 2.0

# Providers cut the connection on long-running batches, so it is rebuilt every
# so often rather than held open for a hundred sends.
_RECONNECT_EVERY = 20

_CONNECT_TIMEOUT = 30


@dataclass
class SmtpSettings:
    """SMTP configuration, read from the environment only."""

    host: str
    port: int
    username: str
    password: str
    from_address: str
    from_name: str = ""
    use_tls: bool = True      # STARTTLS on 587
    use_ssl: bool = False     # implicit TLS on 465

    @property
    def sender(self) -> str:
        return formataddr((self.from_name, self.from_address)) if self.from_name else self.from_address


class SenderNotConfigured(RuntimeError):
    """Raised when the environment does not describe a usable mail account."""


def load_smtp_settings() -> SmtpSettings:
    """Build settings from environment variables.

    Expected in .env (never in source control):

        SMTP_HOST=smtp.gmail.com
        SMTP_PORT=587
        SMTP_USER=<the sending account>
        SMTP_PASSWORD=<an app password, not the account password>
        SMTP_FROM=<optional, defaults to SMTP_USER>
        SMTP_FROM_NAME=<optional display name>

    Raises SenderNotConfigured with a specific message rather than failing
    halfway through a batch.
    """
    host = (os.environ.get("SMTP_HOST") or "").strip()
    username = (os.environ.get("SMTP_USER") or "").strip()
    password = os.environ.get("SMTP_PASSWORD") or ""

    missing = [
        name
        for name, value in (
            ("SMTP_HOST", host),
            ("SMTP_USER", username),
            ("SMTP_PASSWORD", password),
        )
        if not value
    ]
    if missing:
        raise SenderNotConfigured(
            "Email sending is not configured. Add "
            + ", ".join(missing)
            + " to your .env file and restart the server."
        )

    try:
        port = int((os.environ.get("SMTP_PORT") or "587").strip())
    except ValueError:
        raise SenderNotConfigured("SMTP_PORT must be a number, for example 587.") from None

    from_address = (os.environ.get("SMTP_FROM") or username).strip()

    return SmtpSettings(
        host=host,
        port=port,
        username=username,
        password=password,
        from_address=from_address,
        from_name=(os.environ.get("SMTP_FROM_NAME") or "").strip(),
        # 465 is implicit TLS; everything else uses STARTTLS.
        use_ssl=port == 465,
        use_tls=port != 465,
    )


def is_configured() -> bool:
    """Whether sending is possible, without raising."""
    try:
        load_smtp_settings()
        return True
    except SenderNotConfigured:
        return False


def _connect(settings: SmtpSettings) -> smtplib.SMTP:
    context = ssl.create_default_context()
    if settings.use_ssl:
        server = smtplib.SMTP_SSL(settings.host, settings.port,
                                  timeout=_CONNECT_TIMEOUT, context=context)
    else:
        server = smtplib.SMTP(settings.host, settings.port, timeout=_CONNECT_TIMEOUT)
        server.ehlo()
        if settings.use_tls:
            server.starttls(context=context)
            server.ehlo()
    server.login(settings.username, settings.password)
    return server


def verify_connection() -> dict[str, Any]:
    """Log in and disconnect, so a bad password is found before a batch runs.

    Worth doing first: discovering the credentials are wrong on message one of
    a hundred is a poor way to find out.
    """
    try:
        settings = load_smtp_settings()
    except SenderNotConfigured as exc:
        return {"ok": False, "error": str(exc)}

    try:
        server = _connect(settings)
        server.quit()
        return {"ok": True, "host": settings.host, "port": settings.port,
                "from": settings.from_address}
    except smtplib.SMTPAuthenticationError:
        return {
            "ok": False,
            "error": (
                "The mail server rejected those credentials. For Gmail this "
                "usually means an ordinary account password was used - an app "
                "password is required, generated at "
                "myaccount.google.com/apppasswords with 2-Step Verification on."
            ),
        }
    except Exception as exc:
        return {"ok": False, "error": f"Could not reach the mail server ({type(exc).__name__}): {exc}"}


def build_message(
    *,
    to_address: str,
    subject: str,
    body: str,
    settings: SmtpSettings,
    reply_to: str = "",
) -> EmailMessage:
    message = EmailMessage()
    message["From"] = settings.sender
    message["To"] = to_address
    message["Subject"] = subject or "(no subject)"
    if reply_to:
        message["Reply-To"] = reply_to
    message.set_content(body or "")
    return message


def send_batch(
    emails: list[dict[str, Any]],
    *,
    delay_seconds: float = DEFAULT_SEND_DELAY_SECONDS,
    progress: Optional[Callable[[int, int, str], None]] = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Send already-generated emails. Generates nothing.

    Each item needs `recipient`, `subject` and `body`; anything else is
    carried through to the report. One failure is recorded against its row and
    the rest continue - a single bad address must not cost the other
    ninety-nine.

    dry_run performs every step except the send itself, which is the safe way
    to rehearse a hundred-message batch.
    """
    # A dry run sends nothing, so it must not require credentials - the whole
    # point is to rehearse the batch before the mail account is set up.
    settings = None if dry_run else load_smtp_settings()
    total = len(emails)
    results: list[dict[str, Any]] = []
    server: Optional[smtplib.SMTP] = None
    sent_since_connect = 0

    try:
        for position, item in enumerate(emails, start=1):
            row = item.get("row", position)
            recipient = (item.get("recipient") or "").strip()
            outcome = {
                "row": row,
                "recipient": recipient,
                "subject": item.get("subject", ""),
                "role": item.get("role", ""),
                "company": item.get("company", ""),
                "status": "pending",
                "error": "",
            }

            if progress:
                progress(position, total, recipient)

            if not recipient or "@" not in parseaddr(recipient)[1]:
                outcome["status"] = "skipped"
                outcome["error"] = "No valid recipient address."
                results.append(outcome)
                continue

            if not (item.get("body") or "").strip():
                outcome["status"] = "skipped"
                outcome["error"] = "The email body is empty."
                results.append(outcome)
                continue

            if dry_run:
                outcome["status"] = "dry-run"
                results.append(outcome)
                continue

            try:
                if server is None or sent_since_connect >= _RECONNECT_EVERY:
                    if server is not None:
                        try:
                            server.quit()
                        except Exception:
                            pass
                    server = _connect(settings)
                    sent_since_connect = 0

                server.send_message(
                    build_message(
                        to_address=recipient,
                        subject=item.get("subject", ""),
                        body=item.get("body", ""),
                        settings=settings,
                    )
                )
                outcome["status"] = "sent"
                sent_since_connect += 1

            except smtplib.SMTPAuthenticationError as exc:
                # Credentials will not fix themselves mid-batch; stop early
                # rather than logging the same failure a hundred times.
                outcome["status"] = "failed"
                outcome["error"] = "Authentication rejected by the mail server."
                results.append(outcome)
                logger.error("SMTP authentication failed: %s", exc)
                for remaining in emails[position:]:
                    results.append({
                        "row": remaining.get("row", ""),
                        "recipient": remaining.get("recipient", ""),
                        "subject": remaining.get("subject", ""),
                        "role": remaining.get("role", ""),
                        "company": remaining.get("company", ""),
                        "status": "failed",
                        "error": "Not attempted - sending stopped after an authentication failure.",
                    })
                break

            except Exception as exc:
                logger.exception("Send failed for row %s", row)
                outcome["status"] = "failed"
                outcome["error"] = f"{type(exc).__name__}: {exc}"[:200]
                server = None  # force a fresh connection for the next message

            results.append(outcome)

            if delay_seconds and position < total:
                time.sleep(delay_seconds)

    finally:
        if server is not None:
            try:
                server.quit()
            except Exception:
                pass

    return {
        "total": total,
        "sent": sum(1 for item in results if item["status"] == "sent"),
        "skipped": sum(1 for item in results if item["status"] == "skipped"),
        "failed": sum(1 for item in results if item["status"] == "failed"),
        "dry_run": dry_run,
        "results": results,
    }