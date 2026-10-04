"""Sending generated emails over SMTP.

Generation and sending remain separate.

This module:
- loads SMTP settings from environment variables
- verifies SMTP connectivity
- builds email messages
- optionally attaches a CV/document
- sends individual or batch emails
- isolates per-email failures
- reconnects periodically for long batches

No credentials should ever be hard-coded here.
"""

from __future__ import annotations

import logging
import mimetypes
import os
import smtplib
import ssl
import time

from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import formataddr, parseaddr
from pathlib import Path
from typing import Any, Callable, Optional


logger = logging.getLogger("ats.sender")


DEFAULT_SEND_DELAY_SECONDS = 2.0

_RECONNECT_EVERY = 20

_CONNECT_TIMEOUT = 30


@dataclass
class SmtpSettings:
    host: str
    port: int
    username: str
    password: str
    from_address: str
    from_name: str = ""
    use_tls: bool = True
    use_ssl: bool = False

    @property
    def sender(self) -> str:
        if self.from_name:
            return formataddr(
                (
                    self.from_name,
                    self.from_address,
                )
            )

        return self.from_address


class SenderNotConfigured(RuntimeError):
    pass


def load_smtp_settings() -> SmtpSettings:
    host = (
        os.environ.get("SMTP_HOST")
        or ""
    ).strip()

    username = (
        os.environ.get("SMTP_USER")
        or ""
    ).strip()

    password = (
        os.environ.get("SMTP_PASSWORD")
        or ""
    )

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
            "Email sending is not configured. "
            "Missing: "
            + ", ".join(missing)
        )

    try:
        port = int(
            (
                os.environ.get("SMTP_PORT")
                or "587"
            ).strip()
        )

    except ValueError:
        raise SenderNotConfigured(
            "SMTP_PORT must be numeric."
        ) from None

    from_address = (
        os.environ.get("SMTP_FROM")
        or username
    ).strip()

    from_name = (
        os.environ.get("SMTP_FROM_NAME")
        or ""
    ).strip()

    return SmtpSettings(
        host=host,
        port=port,
        username=username,
        password=password,
        from_address=from_address,
        from_name=from_name,
        use_ssl=(port == 465),
        use_tls=(port != 465),
    )


def is_configured() -> bool:
    try:
        load_smtp_settings()
        return True

    except SenderNotConfigured:
        return False


def _connect(
    settings: SmtpSettings,
) -> smtplib.SMTP:

    context = ssl.create_default_context()

    if settings.use_ssl:
        server = smtplib.SMTP_SSL(
            settings.host,
            settings.port,
            timeout=_CONNECT_TIMEOUT,
            context=context,
        )

    else:
        server = smtplib.SMTP(
            settings.host,
            settings.port,
            timeout=_CONNECT_TIMEOUT,
        )

        server.ehlo()

        if settings.use_tls:
            server.starttls(
                context=context
            )

            server.ehlo()

    server.login(
        settings.username,
        settings.password,
    )

    return server


def verify_connection() -> dict[str, Any]:

    try:
        settings = load_smtp_settings()

    except SenderNotConfigured as exc:
        return {
            "ok": False,
            "error": str(exc),
        }

    try:
        server = _connect(settings)

        server.quit()

        return {
            "ok": True,
            "host": settings.host,
            "port": settings.port,
            "from": settings.from_address,
        }

    except smtplib.SMTPAuthenticationError:
        return {
            "ok": False,
            "error": (
                "The mail server rejected "
                "the SMTP credentials."
            ),
        }

    except Exception as exc:
        return {
            "ok": False,
            "error": (
                "Could not connect to SMTP "
                f"({type(exc).__name__}): "
                f"{exc}"
            ),
        }


def _validate_attachment(
    attachment_path: str,
) -> Path:

    path = Path(
        attachment_path
    ).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Attachment not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            "Attachment path is not a file."
        )

    return path


def _add_attachment(
    message: EmailMessage,
    attachment_path: str,
    attachment_name: str = "",
) -> None:

    if not attachment_path:
        return

    path = _validate_attachment(
        attachment_path
    )

    mime_type, _ = (
        mimetypes.guess_type(
            str(path)
        )
    )

    if mime_type:
        maintype, subtype = (
            mime_type.split("/", 1)
        )

    else:
        maintype = "application"
        subtype = "octet-stream"

    filename = (
        attachment_name.strip()
        if attachment_name
        else path.name
    )

    with path.open("rb") as file_handle:
        message.add_attachment(
            file_handle.read(),
            maintype=maintype,
            subtype=subtype,
            filename=filename,
        )


def build_message(
    *,
    to_address: str,
    subject: str,
    body: str,
    settings: SmtpSettings,
    reply_to: str = "",
    attachment_path: str = "",
    attachment_name: str = "",
) -> EmailMessage:

    message = EmailMessage()

    message["From"] = (
        settings.sender
    )

    message["To"] = (
        to_address
    )

    message["Subject"] = (
        subject
        or "(no subject)"
    )

    if reply_to:
        message["Reply-To"] = (
            reply_to
        )

    message.set_content(
        body or ""
    )

    if attachment_path:
        _add_attachment(
            message=message,
            attachment_path=attachment_path,
            attachment_name=attachment_name,
        )

    return message


def send_batch(
    emails: list[dict[str, Any]],
    *,
    delay_seconds: float = (
        DEFAULT_SEND_DELAY_SECONDS
    ),
    progress: Optional[
        Callable[[int, int, str], None]
    ] = None,
    dry_run: bool = False,
) -> dict[str, Any]:

    settings = (
        None
        if dry_run
        else load_smtp_settings()
    )

    total = len(emails)

    results: list[
        dict[str, Any]
    ] = []

    server: Optional[
        smtplib.SMTP
    ] = None

    sent_since_connect = 0

    try:

        for position, item in enumerate(
            emails,
            start=1,
        ):

            row = item.get(
                "row",
                position,
            )

            recipient = (
                item.get("recipient")
                or ""
            ).strip()

            subject = (
                item.get("subject")
                or ""
            )

            body = (
                item.get("body")
                or ""
            )

            attachment_path = (
                item.get(
                    "attachment_path"
                )
                or ""
            ).strip()

            attachment_name = (
                item.get(
                    "attachment_name"
                )
                or ""
            ).strip()

            reply_to = (
                item.get("reply_to")
                or ""
            ).strip()

            outcome = {
                "row": row,
                "recipient": recipient,
                "subject": subject,
                "role": item.get(
                    "role",
                    "",
                ),
                "company": item.get(
                    "company",
                    "",
                ),
                "attachment": (
                    attachment_name
                    or (
                        Path(
                            attachment_path
                        ).name
                        if attachment_path
                        else ""
                    )
                ),
                "status": "pending",
                "error": "",
            }

            if progress:
                progress(
                    position,
                    total,
                    recipient,
                )

            parsed_email = (
                parseaddr(
                    recipient
                )[1]
            )

            if (
                not recipient
                or "@"
                not in parsed_email
            ):
                outcome[
                    "status"
                ] = "skipped"

                outcome[
                    "error"
                ] = (
                    "No valid recipient "
                    "email address."
                )

                results.append(
                    outcome
                )

                continue

            if not body.strip():
                outcome[
                    "status"
                ] = "skipped"

                outcome[
                    "error"
                ] = (
                    "The email body "
                    "is empty."
                )

                results.append(
                    outcome
                )

                continue

            if attachment_path:
                try:
                    _validate_attachment(
                        attachment_path
                    )

                except Exception as exc:
                    outcome[
                        "status"
                    ] = "failed"

                    outcome[
                        "error"
                    ] = (
                        "Attachment error: "
                        f"{exc}"
                    )[:300]

                    results.append(
                        outcome
                    )

                    continue

            if dry_run:
                outcome[
                    "status"
                ] = "dry-run"

                results.append(
                    outcome
                )

                continue

            try:

                if (
                    server is None
                    or sent_since_connect
                    >= _RECONNECT_EVERY
                ):

                    if server is not None:
                        try:
                            server.quit()
                        except Exception:
                            pass

                    server = _connect(
                        settings
                    )

                    sent_since_connect = 0

                message = build_message(
                    to_address=recipient,
                    subject=subject,
                    body=body,
                    settings=settings,
                    reply_to=reply_to,
                    attachment_path=(
                        attachment_path
                    ),
                    attachment_name=(
                        attachment_name
                    ),
                )

                server.send_message(
                    message
                )

                outcome[
                    "status"
                ] = "sent"

                sent_since_connect += 1

            except smtplib.SMTPAuthenticationError as exc:

                outcome[
                    "status"
                ] = "failed"

                outcome[
                    "error"
                ] = (
                    "Authentication rejected "
                    "by the mail server."
                )

                results.append(
                    outcome
                )

                logger.error(
                    "SMTP authentication "
                    "failed: %s",
                    exc,
                )

                for remaining in emails[
                    position:
                ]:

                    results.append(
                        {
                            "row": (
                                remaining.get(
                                    "row",
                                    "",
                                )
                            ),
                            "recipient": (
                                remaining.get(
                                    "recipient",
                                    "",
                                )
                            ),
                            "subject": (
                                remaining.get(
                                    "subject",
                                    "",
                                )
                            ),
                            "role": (
                                remaining.get(
                                    "role",
                                    "",
                                )
                            ),
                            "company": (
                                remaining.get(
                                    "company",
                                    "",
                                )
                            ),
                            "attachment": (
                                remaining.get(
                                    "attachment_name",
                                    "",
                                )
                            ),
                            "status": "failed",
                            "error": (
                                "Not attempted. "
                                "Sending stopped "
                                "after SMTP "
                                "authentication "
                                "failed."
                            ),
                        }
                    )

                break

            except Exception as exc:

                logger.exception(
                    "Email send failed "
                    "for row %s",
                    row,
                )

                outcome[
                    "status"
                ] = "failed"

                outcome[
                    "error"
                ] = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )[:300]

                if server is not None:
                    try:
                        server.quit()
                    except Exception:
                        pass

                server = None

                sent_since_connect = 0

            results.append(
                outcome
            )

            if (
                delay_seconds
                and position < total
            ):
                time.sleep(
                    delay_seconds
                )

    finally:

        if server is not None:
            try:
                server.quit()

            except Exception:
                pass

    return {
        "total": total,
        "sent": sum(
            1
            for item in results
            if item["status"]
            == "sent"
        ),
        "skipped": sum(
            1
            for item in results
            if item["status"]
            == "skipped"
        ),
        "failed": sum(
            1
            for item in results
            if item["status"]
            == "failed"
        ),
        "dry_run": dry_run,
        "results": results,
    }