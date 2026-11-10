"""Long-polling Telegram mode for local development, with no public HTTPS endpoint required."""

import asyncio
import logging
from typing import Any

from fastapi import FastAPI
import httpx

from src.config import settings

logger = logging.getLogger(__name__)

LONG_POLL_SECONDS = 30
HTTP_TIMEOUT_SECONDS = LONG_POLL_SECONDS + 10
PROCESSING_TIMEOUT_SECONDS = 600
RETRY_DELAY_SECONDS = 5


class PollingError(RuntimeError):
    """Raised when polling cannot continue and retrying would not help."""


async def fetch_updates(client: httpx.AsyncClient, offset: int | None) -> list[dict[str, Any]]:
    """Fetch pending updates from Telegram. Passing an offset confirms all earlier updates."""
    url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/getUpdates"
    payload: dict[str, Any] = {
        "timeout": LONG_POLL_SECONDS,
        "allowed_updates": ["message", "callback_query"],
    }
    if offset is not None:
        payload["offset"] = offset

    response = await client.post(url, json=payload, timeout=HTTP_TIMEOUT_SECONDS)
    if response.status_code == 409:
        raise PollingError(
            "Telegram refused polling: this bot has a webhook set, or another poller is running. "
            "Use a separate development bot. Polling never removes a webhook."
        )
    if response.status_code in (401, 404):
        raise PollingError("Telegram rejected the bot token. Check TELEGRAM_BOT_TOKEN.")
    response.raise_for_status()
    return response.json().get("result", [])


async def deliver_update(app_client: httpx.AsyncClient, update: dict[str, Any]) -> None:
    """Send an update through the same handler the webhook uses."""
    response = await app_client.post(
        "/webhook/telegram",
        json=update,
        headers={"X-Telegram-Bot-Api-Secret-Token": settings.TELEGRAM_SECRET_TOKEN},
    )
    if response.status_code >= 400:
        logger.warning("App returned HTTP %d for update %s.", response.status_code, update.get("update_id"))


async def poll_forever(
    telegram_client: httpx.AsyncClient,
    app_client: httpx.AsyncClient,
    max_cycles: int | None = None,
) -> None:
    """Process updates one at a time, confirming them only after they have been handled."""
    offset: int | None = None
    cycles = 0
    while max_cycles is None or cycles < max_cycles:
        cycles += 1
        try:
            updates = await fetch_updates(telegram_client, offset)
        except httpx.HTTPError as err:
            # Only the error type is logged, because httpx errors can contain the bot token URL.
            logger.warning("Telegram polling failed (%s); retrying.", type(err).__name__)
            await asyncio.sleep(RETRY_DELAY_SECONDS)
            continue

        for update in updates:
            update_id = update["update_id"]
            try:
                await deliver_update(app_client, update)
            except Exception as err:
                logger.error("Could not process update %s: %s", update_id, type(err).__name__)
            offset = update_id + 1


async def run_polling(app: FastAPI) -> None:
    """Run the app's startup and shutdown around a Telegram polling loop."""
    if not settings.TELEGRAM_BOT_TOKEN:
        raise PollingError("TELEGRAM_BOT_TOKEN is not set.")
    if not settings.TELEGRAM_SECRET_TOKEN:
        raise PollingError("TELEGRAM_SECRET_TOKEN is not set.")

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with (
            httpx.AsyncClient(transport=transport, base_url="http://local", timeout=PROCESSING_TIMEOUT_SECONDS)
        ) as app_client, httpx.AsyncClient() as telegram_client:
            logger.info("Telegram polling started. Press Ctrl+C to stop.")
            await poll_forever(telegram_client, app_client)


__all__ = ["PollingError", "deliver_update", "fetch_updates", "poll_forever", "run_polling"]
