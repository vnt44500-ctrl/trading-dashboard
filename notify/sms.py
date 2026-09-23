"""SMS alerts for generated signals (opt-in, side-effect free by default).

The signal pipeline never calls this module. Signal generation stays pure;
a separate notifier reads finished Signal objects and sends texts. Disabled
unless the operator opts in via env vars. Twilio is lazily imported only
when a send is attempted, so installs without it keep working. The email
fallback needs only the Python standard library.

Two delivery providers:

1. Twilio (paid, instant, most reliable)
   TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   TWILIO_FROM_NUMBER=+1XXXXXXXXXX      # a Twilio number you own
   NOTIFY_SMS_TO=+16477167264           # your Canadian handset

2. Email-to-SMS gateway (completely free; arrives as a normal text)
   SMTP_HOST=smtp.gmail.com             # any SMTP account works
   SMTP_PORT=587
   SMTP_USER=you@gmail.com
   SMTP_PASS=<app password>             # Gmail: create an App Password
   SMS_GATEWAY_TO=6477167264@pcs.rogers.com
   Canadian carrier domains: Rogers/Fido @pcs.rogers.com,
   Bell/Virgin @txt.bell.ca, Telus/Koodo/Public @msg.telus.com,
   Freedom @txt.freedommobile.ca

Manual provider switch (SMS_PROVIDER):
   auto    (default) Twilio first; on failure falls back to email and the
           failed provider cools down for SMS_PROVIDER_COOLDOWN_MIN (15).
   twilio  Twilio only (forced).
   email   email-to-SMS gateway only (forced).

Optional behaviour flags (all safe defaults; the notifier never alters a
signal, score, or action — it only reads finished Signal objects):
   NOTIFY_SMS_INCLUDE_HOLD=false        # set true to also text HOLD outcomes
   NOTIFY_SMS_INCLUDE_RESEARCH=true     # research-only (e.g. short) ideas texted
   NOTIFY_SMS_DEDUP_HOURS=24            # same signal id re-texted after this long

Manual test from the project root:
   python -m notify.sms                    # sends the built-in test text
   python -m notify.sms "custom message"
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

SENT_LOG = Path(__file__).resolve().parent / "sent_alerts.json"
_LOCK = threading.Lock()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_str(name: str) -> str:
    raw = os.getenv(name)
    return raw.strip() if raw else ""


def _twilio_ready() -> bool:
    return all(_env_str(name) for name in (
        "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN",
        "TWILIO_FROM_NUMBER", "NOTIFY_SMS_TO"))


def _email_ready() -> bool:
    return all(_env_str(name) for name in (
        "SMTP_HOST", "SMTP_USER", "SMTP_PASS", "SMS_GATEWAY_TO"))


_READY = {"twilio": _twilio_ready, "email": _email_ready}


def provider_status() -> dict:
    """Snapshot of provider readiness and the active SMS_PROVIDER selection."""
    requested = _env_str("SMS_PROVIDER").lower() or "auto"
    if requested not in ("auto", "twilio", "email"):
        requested = "auto"
    return {
        "requested": requested,
        "twilio_ready": _twilio_ready(),
        "email_ready": _email_ready(),
        "twilio_in_cooldown": _in_cooldown("twilio"),
        "email_in_cooldown": _in_cooldown("email"),
    }


def sms_configured() -> bool:
    """True only when the operator opted in AND one provider is fully configured."""
    return _env_bool("NOTIFY_SMS_ENABLED", False) and (_twilio_ready() or _email_ready())


def _load_sent() -> dict:
    try:
        if SENT_LOG.exists():
            with open(SENT_LOG, "r", encoding="utf-8") as handle:
                data = json.load(handle)
                if isinstance(data, dict):
                    return data
    except Exception as error:
        logger.warning("SMS sent-log unreadable, starting fresh: %s", error)
    return {}


def _save_sent(data: dict) -> None:
    try:
        SENT_LOG.parent.mkdir(parents=True, exist_ok=True)
        temporary = SENT_LOG.with_suffix(".tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        os.replace(temporary, SENT_LOG)
    except Exception as error:
        logger.warning("SMS sent-log could not be saved: %s", error)


def _alert_id(signal) -> str:
    if getattr(signal, "id", ""):
        return str(signal.id)
    price = getattr(signal, "price", 0) or 0
    asset = getattr(signal, "asset", None)
    return (f"{getattr(asset, 'symbol', '?')}:"
            f"{getattr(signal, 'timeframe', '?')}:{getattr(signal, 'action', '?')}:"
            f"{round(float(price), 2)}")


def format_signal_message(signal) -> str:
    """One compact SMS (single segment where possible) per signal."""
    asset = getattr(signal, "asset", None)
    symbol = getattr(asset, "symbol", "?")
    action = getattr(signal, "action", "HOLD")
    timeframe = getattr(signal, "timeframe", "?")
    price = getattr(signal, "price", None)
    confidence = getattr(signal, "confidence", None)
    parts = [f"{symbol} {timeframe}: {action}"]
    if price:
        parts.append(f"@ {float(price):,.2f}")
    if confidence is not None:
        parts.append(f"conf {float(confidence):.0%}")
    if action == "HOLD" and getattr(signal, "raw_action", "HOLD") != "HOLD":
        parts.append(f"(suggests {signal.raw_action})")
    return " ".join(parts)[:300]


def _send_via_twilio(body: str) -> str:
    """Lazy Twilio import: only required when an SMS is actually sent."""
    try:
        from twilio.rest import Client  # type: ignore
    except ImportError as error:
        raise RuntimeError("twilio package not installed (pip install twilio)") from error
    client = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
    message = client.messages.create(
        body=body,
        from_=os.environ["TWILIO_FROM_NUMBER"],
        to=os.environ["NOTIFY_SMS_TO"],
    )
    return getattr(message, "sid", "")


def _send_via_email(body: str) -> str:
    """Deliver body via the carrier's email-to-SMS bridge. Stdlib only."""
    import smtplib
    from email.message import EmailMessage

    try:
        port = int(_env_str("SMTP_PORT") or 587)
    except ValueError:
        port = 587
    message = EmailMessage()
    message["From"] = _env_str("SMTP_USER")
    message["To"] = _env_str("SMS_GATEWAY_TO")  # no Subject: carriers prepend it
    message.set_content(body)
    with smtplib.SMTP(_env_str("SMTP_HOST"), port, timeout=20) as server:
        server.starttls()
        server.login(_env_str("SMTP_USER"), _env_str("SMTP_PASS"))
        server.send_message(message)
    return f"email-{datetime.now().strftime('%Y%m%d%H%M%S')}"


def _cooldown_seconds() -> int:
    try:
        return max(0, int(float(os.getenv("SMS_PROVIDER_COOLDOWN_MIN", "15") or 15) * 60))
    except ValueError:
        return 15 * 60


_PROVIDER_FAILURES: dict = {}  # provider name -> time.monotonic() of last failure


def _in_cooldown(provider: str) -> bool:
    failed_at = _PROVIDER_FAILURES.get(provider)
    if failed_at is None:
        return False
    if time.monotonic() - failed_at >= _cooldown_seconds():
        _PROVIDER_FAILURES.pop(provider, None)
        return False
    return True


def _provider_order() -> tuple:
    requested = _env_str("SMS_PROVIDER").lower() or "auto"
    if requested == "twilio":
        return ("twilio",)
    if requested == "email":
        return ("email",)
    return ("twilio", "email")  # auto: Twilio first, free gateway as fallback


def _deliver(body: str) -> tuple:
    """Send body honoring SMS_PROVIDER. Returns (reference, provider_used).

    auto skips providers in cooldown; a forced provider is always attempted.
    Raises the last error when every ready provider failed.
    """
    requested = _env_str("SMS_PROVIDER").lower() or "auto"
    last_error = None
    for provider in _provider_order():
        if not _READY[provider]():
            last_error = RuntimeError(f"{provider} not fully configured (see .env)")
            continue
        if requested == "auto" and _in_cooldown(provider):
            last_error = RuntimeError(f"{provider} cooling down after a recent failure")
            continue
        try:
            if provider == "twilio":
                return _send_via_twilio(body), "twilio"
            return _send_via_email(body), "email"
        except Exception as error:  # remember failure, try the next provider
            _PROVIDER_FAILURES[provider] = time.monotonic()
            logger.warning("%s send failed: %s", provider, error)
            last_error = error
    raise last_error or RuntimeError("no SMS provider configured")


def send_test_sms(text: str = "") -> dict:
    """Send a test text through the real delivery path (bypasses dedup).

    Never raises. Example: send_test_sms("hello") or `python -m notify.sms`.
    """
    body = (text or "[TradingDash] test - SMS alerts are working.").strip()
    if not sms_configured():
        return {"sent": False,
                "reason": "NOTIFY_SMS_ENABLED is false or no provider fully configured",
                "status": provider_status()}
    try:
        reference, provider = _deliver(body)
        return {"sent": True, "provider": provider, "sid": reference, "message": body}
    except Exception as error:
        return {"sent": False, "reason": str(error), "status": provider_status()}


def notify_signals(signals: dict, dry_run: bool = False) -> dict:
    """Text the operator about generated signals. Never raises.

    Args:
        signals: mapping like {"short": Signal, "medium": Signal, ...}.
        dry_run: build messages and report what *would* send, without sending.

    Returns a per-timeframe report, e.g.
    {"medium": {"sent": True, "sid": "SM...", "message": "..."}, ...}.
    Skipped timeframes report {"sent": False, "reason": "..."}.
    """
    report: dict = {}
    if not isinstance(signals, dict) or not signals:
        return report

    if not sms_configured() and not dry_run:
        for timeframe in signals:
            report[timeframe] = {"sent": False, "reason": "SMS alerts not enabled"}
        return report

    include_hold = _env_bool("NOTIFY_SMS_INCLUDE_HOLD", False)
    include_research = _env_bool("NOTIFY_SMS_INCLUDE_RESEARCH", True)
    try:
        dedup_hours = int(os.getenv("NOTIFY_SMS_DEDUP_HOURS", "24") or 24)
    except ValueError:
        dedup_hours = 24

    with _LOCK:
        sent = _load_sent()
        now = datetime.now()
        dirty = False
        for timeframe, signal in signals.items():
            action = getattr(signal, "action", "HOLD")
            if action == "HOLD" and not include_hold:
                report[timeframe] = {"sent": False,
                                     "reason": "HOLD skipped (opt-in via NOTIFY_SMS_INCLUDE_HOLD=true)"}
                continue
            if (not getattr(signal, "live_eligible", True)
                    and not include_research):
                report[timeframe] = {"sent": False,
                                     "reason": "research-only suggestion skipped "
                                               "(opt-out via NOTIFY_SMS_INCLUDE_RESEARCH=false)"}
                continue
            key = _alert_id(signal)
            last = sent.get(key)
            if last and not dry_run:
                try:
                    age_hours = (now - datetime.fromisoformat(last)).total_seconds() / 3600
                    if age_hours < dedup_hours:
                        report[timeframe] = {"sent": False,
                                             "reason": f"duplicate within {dedup_hours}h"}
                        continue
                except Exception:
                    pass
            message = format_signal_message(signal)
            if dry_run:
                report[timeframe] = {"sent": False, "reason": "dry-run", "message": message}
                continue
            try:
                reference, provider = _deliver(f"[TradingDash] {message}")
                sent[key] = now.isoformat()
                dirty = True
                report[timeframe] = {"sent": True, "sid": reference,
                                     "provider": provider, "message": message}
            except Exception as error:  # SMS failure must never break the app
                logger.warning("SMS send failed for %s: %s", timeframe, error)
                report[timeframe] = {"sent": False, "reason": f"send failed: {error}",
                                     "provider": (_env_str("SMS_PROVIDER").lower() or "auto")}
        if dirty:
            _save_sent(sent)
    return report


def notify_signals_async(signals: dict) -> None:
    """Fire-and-forget SMS fan-out that can never block or break the caller.

    Spawns one daemon thread per notification batch; the caller returns
    immediately and any failure is logged only. Safe to call from the
    Streamlit render path, schedulers, or batch jobs.
    """
    def _worker() -> None:
        try:
            notify_signals(signals)
        except Exception as error:  # last-resort guard; notify_signals already swallows
            logger.warning("Async SMS batch failed: %s", error)

    try:
        threading.Thread(target=_worker, name="sms-notify", daemon=True).start()
    except Exception as error:
        logger.warning("Could not start SMS thread: %s", error)


def _main(argv: list) -> int:
    """`python -m notify.sms [message]` — manual test switch from project root."""
    try:
        from dotenv import load_dotenv
        # Load .env explicitly (cwd/runpy-proof). Stale OS-level vars like
        # NOTIFY_SMS_ENABLED=false would otherwise silently win, so the file
        # is loaded with override=True; then any value that was genuinely set
        # in this process's environment BEFORE the load is restored on top,
        # so explicit env vars still take precedence over .env empties.
        dotenv_path = Path(__file__).resolve().parent.parent / ".env"
        sms_keys = ("NOTIFY_SMS_ENABLED", "NOTIFY_SMS_TO", "SMS_PROVIDER",
                    "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER",
                    "SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS",
                    "SMS_GATEWAY_TO", "NOTIFY_SMS_INCLUDE_HOLD",
                    "NOTIFY_SMS_INCLUDE_RESEARCH", "NOTIFY_SMS_DEDUP_HOURS",
                    "SMS_PROVIDER_COOLDOWN_MIN")
        preexisting = {key: os.environ[key] for key in sms_keys if os.environ.get(key)}
        load_dotenv(dotenv_path=dotenv_path, override=True)
        for key, value in preexisting.items():
            os.environ[key] = value
    except ImportError:
        pass
    result = send_test_sms(argv[1] if len(argv) > 1 else "")
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("sent") else 1


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv))
