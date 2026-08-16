"""Telegram Bot API HTTP client methods for sending messages, answering callbacks, and downloading media."""

import logging
from pathlib import Path
import tempfile
from typing import Any
import httpx
from src.config import settings

logger = logging.getLogger(__name__)


def split_telegram_message(text: str, max_chunk_size: int = 4000) -> list[str]:
    """Split long markdown or plain text into clean chunks within Telegram's 4096-character limit.

    Args:
        text: The complete message text.
        max_chunk_size: Maximum size per message chunk (default: 4000).

    Returns:
        List of message chunk strings.
    """
    if not text:
        return [""]
    if len(text) <= max_chunk_size:
        return [text]

    chunks: list[str] = []
    remaining = text.strip()

    while len(remaining) > max_chunk_size:
        split_idx = -1

        # 1. Prefer splitting on Markdown headings and paragraph boundaries
        for delim in ["\n\n## ", "\n\n# ", "\n\n"]:
            pos = remaining.rfind(delim, 0, max_chunk_size)
            if pos != -1 and pos > 0:
                split_idx = pos
                break

        # 2. Fall back to newline or space if reasonably far into the chunk
        if split_idx == -1:
            for delim in ["\n", " "]:
                pos = remaining.rfind(delim, 0, max_chunk_size)
                if pos != -1 and pos >= max_chunk_size // 2:
                    split_idx = pos
                    break

        # 3. Hard limit cut if no reasonable boundary found
        if split_idx == -1:
            split_idx = max_chunk_size

        chunk = remaining[:split_idx].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_idx:].strip()

    if remaining:
        chunks.append(remaining)

    return chunks or [text]


async def _send_single_telegram_chunk(
    client: httpx.AsyncClient,
    chat_id: int,
    text: str,
    reply_markup: dict[str, Any] | None = None,
) -> bool:
    """Send a single chunk to Telegram API with HTML formatting and graceful fallbacks."""
    from src.telegram.formatters import convert_markdown_to_telegram_html

    url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"

    # 1. Try HTML formatting first (safely preserves WikiLinks and special characters)
    html_text = convert_markdown_to_telegram_html(text)
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": html_text,
        "parse_mode": "HTML",
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup

    try:
        response = await client.post(url, json=payload)
        if response.status_code == 200:
            return True

        logger.warning(
            "Telegram HTML parse failed (%d: %s); retrying chunk with Markdown.",
            response.status_code,
            response.text[:120],
        )

        # 2. Try legacy Markdown fallback
        payload["text"] = text
        payload["parse_mode"] = "Markdown"
        retry_response = await client.post(url, json=payload)
        if retry_response.status_code == 200:
            return True

        # 3. If Markdown fails, retry as plain text format
        logger.warning(
            "Telegram Markdown parse failed (%d: %s); retrying chunk as plain text.",
            retry_response.status_code,
            retry_response.text[:120],
        )
        payload.pop("parse_mode", None)
        plain_response = await client.post(url, json=payload)
        if plain_response.status_code == 200:
            return True
        else:
            logger.error(
                "Failed to send Telegram message chunk (%d): %s",
                plain_response.status_code,
                plain_response.text,
            )
            return False
    except Exception as err:
        logger.exception("Unexpected error sending Telegram message chunk to chat %d: %s", chat_id, err)
        return False


async def send_telegram_message(
    chat_id: int,
    text: str,
    reply_markup: dict[str, Any] | None = None,
) -> bool:
    """Send text message back to user via Telegram Bot API with automatic chunking for long content.

    Args:
        chat_id: Telegram chat ID.
        text: Text response to send (will be chunked cleanly if >4000 characters).
        reply_markup: Optional inline keyboard or reply markup dictionary (attached to the final chunk).

    Returns:
        True if all chunks sent successfully, False otherwise.
    """
    if not settings.TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN not configured; logging message instead:\n%s", text)
        return False

    chunks = split_telegram_message(text, max_chunk_size=4000)
    all_success = True

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            for idx, chunk in enumerate(chunks):
                # Attach reply_markup only to the final chunk
                chunk_markup = reply_markup if (idx == len(chunks) - 1) else None
                success = await _send_single_telegram_chunk(
                    client=client,
                    chat_id=chat_id,
                    text=chunk,
                    reply_markup=chunk_markup,
                )
                if not success:
                    all_success = False
                    break
            if all_success:
                logger.info("Successfully sent %d Telegram message chunk(s) to chat %d", len(chunks), chat_id)
            return all_success
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
    """Edit existing Telegram message text in-place with HTML formatting and fallback.

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

    from src.telegram.formatters import convert_markdown_to_telegram_html

    url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/editMessageText"
    html_text = convert_markdown_to_telegram_html(text)
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": html_text,
        "parse_mode": "HTML",
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.post(url, json=payload)
            if res.status_code == 200:
                return True
            # Fallback to plain text if HTML edit fails
            payload["text"] = text
            payload.pop("parse_mode", None)
            retry_res = await client.post(url, json=payload)
            return retry_res.status_code == 200
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
            await asyncio.to_thread(temp_file.write_bytes, dl_res.content)
            logger.info("Downloaded Telegram audio file (%d bytes) to %s", len(dl_res.content), temp_file)
            return temp_file

    except Exception as err:
        logger.exception("Error downloading Telegram file %s: %s", file_id, err)
        return None


__all__ = [
    "split_telegram_message",
    "send_telegram_message",
    "answer_telegram_callback_query",
    "edit_telegram_message_text",
    "download_telegram_file",
]
