"""Telegram Bot API HTTP client methods for sending messages, answering callbacks, and downloading media."""

import logging
from pathlib import Path
import tempfile
from typing import Any
import httpx
from src.config import settings

logger = logging.getLogger(__name__)


async def send_telegram_message(
    chat_id: int,
    text: str,
    reply_markup: dict[str, Any] | None = None,
) -> bool:
    """Send text message back to user via Telegram Bot API with optional inline keyboard.

    Args:
        chat_id: Telegram chat ID.
        text: Text response to send.
        reply_markup: Optional inline keyboard or reply markup dictionary.

    Returns:
        True if sent successfully, False otherwise.
    """
    if not settings.TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN not configured; logging message instead:\n%s", text)
        return False

    url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=payload)
            if response.status_code == 200:
                logger.info("Successfully sent Telegram message to chat %d", chat_id)
                return True

            # If Markdown parsing fails, retry as plain text format
            logger.warning("Markdown parse failed (%d); retrying as plain text.", response.status_code)
            payload.pop("parse_mode", None)
            retry_response = await client.post(url, json=payload)
            if retry_response.status_code == 200:
                logger.info("Sent plain text Telegram message to chat %d", chat_id)
                return True
            else:
                logger.error(
                    "Failed to send Telegram message (%d): %s",
                    retry_response.status_code,
                    retry_response.text,
                )
                return False
    except Exception as err:
        logger.exception("Unexpected error sending Telegram message to chat %d: %s", chat_id, err)
        return False


async def answer_telegram_callback_query(
    callback_query_id: str,
    text: str | None = None,
) -> bool:
    """Acknowledge Telegram callback query button click.

    Args:
        callback_query_id: Unique callback query identifier.
        text: Optional notification toast text to display on user's screen.

    Returns:
        True if acknowledged successfully, False otherwise.
    """
    if not settings.TELEGRAM_BOT_TOKEN:
        return False

    url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/answerCallbackQuery"
    payload: dict[str, Any] = {"callback_query_id": callback_query_id}
    if text:
        payload["text"] = text

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            res = await client.post(url, json=payload)
            return res.status_code == 200
    except Exception as err:
        logger.warning("Failed to answer callback query: %s", err)
        return False


async def edit_telegram_message_text(
    chat_id: int,
    message_id: int,
    text: str,
    reply_markup: dict[str, Any] | None = None,
) -> bool:
    """Edit existing Telegram message text in-place.

    Args:
        chat_id: Telegram chat ID.
        message_id: Message ID of message to edit.
        text: Updated markdown text.
        reply_markup: Optional updated inline keyboard buttons.

    Returns:
        True if edited successfully, False otherwise.
    """
    if not settings.TELEGRAM_BOT_TOKEN:
        return False

    url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/editMessageText"
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": text,
        "parse_mode": "Markdown",
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.post(url, json=payload)
            return res.status_code == 200
    except Exception as err:
        logger.warning("Failed to edit Telegram message: %s", err)
        return False


async def download_telegram_file(file_id: str) -> Path | None:
    """Download a file from Telegram Bot API into a local temporary path.

    Args:
        file_id: Telegram file ID.

    Returns:
        Path to downloaded file, or None on failure.
    """
    if not settings.TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN missing; cannot download audio file.")
        return None

    try:
        # 1. Get file path from Telegram
        get_file_url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/getFile?file_id={file_id}"
        async with httpx.AsyncClient(timeout=15.0) as client:
            res = await client.get(get_file_url)
            if res.status_code != 200:
                logger.error("Failed to get file info for %s (%d): %s", file_id, res.status_code, res.text)
                return None

            file_info = res.json().get("result", {})
            file_path = file_info.get("file_path")
            if not file_path:
                logger.error("No file_path returned in Telegram getFile result.")
                return None

            # 2. Download file bytes
            download_url = f"https://api.telegram.org/file/bot{settings.TELEGRAM_BOT_TOKEN}/{file_path}"
            dl_res = await client.get(download_url)
            if dl_res.status_code != 200:
                logger.error("Failed to download file from %s (%d)", download_url, dl_res.status_code)
                return None

            extension = Path(file_path).suffix or ".ogg"
            temp_file = Path(tempfile.gettempdir()) / f"tg_audio_{file_id}{extension}"
            temp_file.write_bytes(dl_res.content)
            logger.info("Downloaded Telegram audio file (%d bytes) to %s", len(dl_res.content), temp_file)
            return temp_file

    except Exception as err:
        logger.exception("Error downloading Telegram file %s: %s", file_id, err)
        return None


__all__ = [
    "send_telegram_message",
    "answer_telegram_callback_query",
    "edit_telegram_message_text",
    "download_telegram_file",
]
