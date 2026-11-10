"""Tests for the local development Telegram polling mode."""

import json
import logging
import sys
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.telegram import polling

# Built at runtime so secret scanners don't mistake a test value for a real token.
FAKE_TOKEN = "123456789:" + "x" * 35


def _telegram_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _ok(updates: list[dict]) -> httpx.Response:
    return httpx.Response(200, json={"ok": True, "result": updates})


@pytest.fixture(autouse=True)
def _token(monkeypatch):
    monkeypatch.setattr(polling.settings, "TELEGRAM_BOT_TOKEN", FAKE_TOKEN)
    monkeypatch.setattr(polling.settings, "TELEGRAM_SECRET_TOKEN", "test-secret")


@pytest.mark.asyncio
async def test_fetch_requests_messages_and_buttons_from_the_offset():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return _ok([{"update_id": 7}])

    async with _telegram_client(handler) as client:
        updates = await polling.fetch_updates(client, offset=7)

    assert updates == [{"update_id": 7}]
    assert seen["url"].endswith("/getUpdates")
    assert seen["body"]["offset"] == 7
    assert set(seen["body"]["allowed_updates"]) == {"message", "callback_query"}
    assert seen["body"]["timeout"] > 0


@pytest.mark.parametrize(
    ("status", "expected_text"),
    [(409, "webhook"), (401, "token")],
)
@pytest.mark.asyncio
async def test_unrecoverable_telegram_errors_stop_with_a_clear_message(status, expected_text):
    async with _telegram_client(lambda request: httpx.Response(status, json={"ok": False})) as client:
        with pytest.raises(polling.PollingError) as error:
            await polling.fetch_updates(client, offset=None)

    assert expected_text in str(error.value).lower()
    assert FAKE_TOKEN not in str(error.value)


@pytest.mark.asyncio
async def test_polling_never_deletes_a_webhook():
    paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(409, json={"ok": False})

    async with _telegram_client(handler) as client:
        with pytest.raises(polling.PollingError):
            await polling.poll_forever(client, AsyncMock())

    assert all("Webhook" not in path for path in paths)


@pytest.mark.asyncio
async def test_update_is_sent_to_the_app_with_the_webhook_secret():
    app_client = AsyncMock()
    app_client.post.return_value = httpx.Response(200, json={"status": "ok"})

    await polling.deliver_update(app_client, {"update_id": 1})

    args, kwargs = app_client.post.call_args
    assert args[0] == "/webhook/telegram"
    assert kwargs["json"] == {"update_id": 1}
    assert kwargs["headers"]["X-Telegram-Bot-Api-Secret-Token"] == "test-secret"


@pytest.mark.asyncio
async def test_update_is_only_acknowledged_after_it_has_been_processed():
    events: list[str] = []
    batches = [[{"update_id": 10}, {"update_id": 11}], []]

    def handler(request: httpx.Request) -> httpx.Response:
        offset = json.loads(request.content).get("offset")
        events.append(f"fetch offset={offset}")
        return _ok(batches.pop(0) if batches else [])

    async def deliver(_app_client, update: dict) -> None:
        events.append(f"deliver {update['update_id']}")

    async with _telegram_client(handler) as client:
        with patch.object(polling, "deliver_update", deliver):
            await polling.poll_forever(client, AsyncMock(), max_cycles=2)

    assert events == ["fetch offset=None", "deliver 10", "deliver 11", "fetch offset=12"]


@pytest.mark.asyncio
async def test_failed_delivery_is_logged_and_does_not_block_later_updates(caplog):
    delivered: list[int] = []
    batches = [[{"update_id": 1}, {"update_id": 2}]]

    async def deliver(_app_client, update: dict) -> None:
        delivered.append(update["update_id"])
        if update["update_id"] == 1:
            raise RuntimeError("boom")

    def handler(request: httpx.Request) -> httpx.Response:
        return _ok(batches.pop(0) if batches else [])

    async with _telegram_client(handler) as client:
        with patch.object(polling, "deliver_update", deliver), caplog.at_level(logging.ERROR):
            await polling.poll_forever(client, AsyncMock(), max_cycles=1)

    assert delivered == [1, 2]
    assert "update 1" in caplog.text


@pytest.mark.asyncio
async def test_network_errors_are_retried_without_logging_the_token(caplog):
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ConnectError(f"cannot reach {request.url}")
        return _ok([])

    async with _telegram_client(handler) as client:
        with patch.object(polling.asyncio, "sleep", new_callable=AsyncMock) as slept, caplog.at_level(logging.WARNING):
            await polling.poll_forever(client, AsyncMock(), max_cycles=2)

    assert calls["count"] == 2
    slept.assert_awaited_once()
    assert FAKE_TOKEN not in caplog.text


@pytest.mark.asyncio
async def test_missing_token_stops_before_starting_the_app(monkeypatch):
    monkeypatch.setattr(polling.settings, "TELEGRAM_BOT_TOKEN", "")
    with pytest.raises(polling.PollingError):
        await polling.run_polling(object())


def test_poll_command_starts_polling_and_not_the_web_server():
    import src.main as app_main

    with (
        patch("src.main.run_polling", new_callable=AsyncMock) as run_polling,
        patch("uvicorn.run") as uvicorn_run,
        patch.object(sys, "argv", ["main.py", "poll"]),
    ):
        app_main.cli_entrypoint()

    run_polling.assert_awaited_once_with(app_main.app)
    uvicorn_run.assert_not_called()


def test_dev_env_file_overrides_the_other_env_files():
    from src.config import Settings

    env_files = Settings.model_config["env_file"]
    assert env_files[-1] == ".env.dev.local"
    assert ".env.local" in env_files
