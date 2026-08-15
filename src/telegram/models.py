"""Pydantic v2 data models for Telegram webhook updates and payloads."""

from pydantic import BaseModel, ConfigDict, Field


class TelegramUser(BaseModel):
    """Schema for Telegram user/sender details."""

    id: int = Field(description="Unique Telegram user identifier")
    is_bot: bool = Field(default=False, description="Whether the user is a bot")
    first_name: str | None = Field(default=None, description="User first name")
    username: str | None = Field(default=None, description="Telegram username")

    model_config = ConfigDict(str_strip_whitespace=True)


class TelegramChat(BaseModel):
    """Schema for Telegram chat details."""

    id: int = Field(description="Telegram chat ID")
    first_name: str | None = Field(default=None, description="Chat first name")
    username: str | None = Field(default=None, description="Chat username")
    type: str | None = Field(default=None, description="Chat type (e.g. private)")

    model_config = ConfigDict(str_strip_whitespace=True)


class TelegramVoice(BaseModel):
    """Schema for voice memo metadata."""

    file_id: str = Field(description="Telegram file ID")
    file_unique_id: str = Field(description="Unique file identifier")
    duration: int | None = Field(default=None, description="Audio duration in seconds")
    mime_type: str | None = Field(default=None, description="MIME type")
    file_size: int | None = Field(default=None, description="File size in bytes")

    model_config = ConfigDict(str_strip_whitespace=True)


class TelegramAudio(BaseModel):
    """Schema for general audio file metadata."""

    file_id: str = Field(description="Telegram file ID")
    file_unique_id: str = Field(description="Unique file identifier")
    duration: int | None = Field(default=None, description="Audio duration in seconds")
    file_name: str | None = Field(default=None, description="Original filename")
    mime_type: str | None = Field(default=None, description="MIME type")
    file_size: int | None = Field(default=None, description="File size in bytes")

    model_config = ConfigDict(str_strip_whitespace=True)


class TelegramMessage(BaseModel):
    """Schema for Telegram message payload."""

    message_id: int = Field(description="Unique message identifier")
    chat: TelegramChat = Field(description="Chat recipient information")
    from_user: TelegramUser | None = Field(default=None, alias="from", description="Sender user details")
    text: str | None = Field(default=None, description="Message text content")
    caption: str | None = Field(default=None, description="Caption for media files")
    voice: TelegramVoice | None = Field(default=None, description="Voice memo payload")
    audio: TelegramAudio | None = Field(default=None, description="Audio file payload")
    date: int | None = Field(default=None, description="Unix timestamp of message")

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)


class TelegramCallbackQuery(BaseModel):
    """Schema for Telegram interactive button callback query."""

    id: str = Field(description="Unique callback query identifier")
    from_user: TelegramUser = Field(alias="from", description="User who pressed the button")
    message: TelegramMessage | None = Field(default=None, description="Parent message")
    data: str | None = Field(default=None, description="Data attached to the inline button")

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)


class TelegramUpdate(BaseModel):
    """Schema for incoming Telegram webhook update."""

    update_id: int = Field(description="Unique update identifier")
    message: TelegramMessage | None = Field(default=None, description="Incoming message object")
    callback_query: TelegramCallbackQuery | None = Field(default=None, description="Incoming callback query")

    model_config = ConfigDict(str_strip_whitespace=True)


__all__ = [
    "TelegramUser",
    "TelegramChat",
    "TelegramVoice",
    "TelegramAudio",
    "TelegramMessage",
    "TelegramCallbackQuery",
    "TelegramUpdate",
]
