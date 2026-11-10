"""Application logging setup that keeps credentials out of log output."""

import logging
import re

_BOT_TOKEN_IN_URL = re.compile(r"bot\d+:[A-Za-z0-9_-]+")
_BARE_BOT_TOKEN = re.compile(r"\b\d{6,}:[A-Za-z0-9_-]{30,}\b")


class RedactingFormatter(logging.Formatter):
    """Remove Telegram bot tokens from messages and tracebacks after formatting."""

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        formatted = _BOT_TOKEN_IN_URL.sub("bot<redacted>", formatted)
        return _BARE_BOT_TOKEN.sub("<redacted>", formatted)


def configure_logging() -> None:
    """Configure root logging and silence libraries that log full request URLs."""
    logging.basicConfig(level=logging.INFO)
    formatter = RedactingFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)

    # httpx logs every request URL at INFO, and Telegram URLs contain the bot token.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


__all__ = ["RedactingFormatter", "configure_logging"]
