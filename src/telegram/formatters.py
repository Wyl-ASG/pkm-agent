"""Telegram message and interactive inline keyboard formatters for tasks and briefings."""

from datetime import date, datetime, timedelta
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.vault.md_writer import TaskItem


def format_pending_tasks_message(
    tasks: list["TaskItem"],
    today: date | None = None,
) -> tuple[str, list[list[dict[str, str]]]]:
    """Generate structured markdown template and inline keyboard for all pending tasks in vault.

    Args:
        tasks: List of active TaskItem instances.
        today: Current date reference. Defaults to today's date.

    Returns:
        Tuple of (formatted_markdown_message, inline_keyboard_button_matrix).
    """
    ref_date = today or datetime.now().date()
    today_str = ref_date.strftime("%Y-%m-%d")
    tomorrow_str = (ref_date + timedelta(days=1)).strftime("%Y-%m-%d")
    day_after_str = (ref_date + timedelta(days=2)).strftime("%Y-%m-%d")

    if not tasks:
        msg = (
            "📋 **Pending Tasks Overview**\n"
            "───────────────────────\n"
            "✨ **All clear!** You currently have no pending tasks in your vault.\n"
            "Enjoy your day or capture a new task anytime! 🚀"
        )
        return msg, []

    # Partition tasks
    overdue_tasks: list[TaskItem] = []
    today_tasks: list[TaskItem] = []
    next_2_days_tasks: list[TaskItem] = []
    upcoming_tasks: list[TaskItem] = []

    for t in tasks:
        if t.due_date:
            if t.due_date < today_str:
                overdue_tasks.append(t)
            elif t.due_date == today_str:
                today_tasks.append(t)
            elif t.due_date in (tomorrow_str, day_after_str):
                next_2_days_tasks.append(t)
            else:
                upcoming_tasks.append(t)
        else:
            if t.daily_date == today_str:
                today_tasks.append(t)
            else:
                upcoming_tasks.append(t)

    lines: list[str] = [
        "📋 **Pending Tasks Overview**",
        "───────────────────────",
        f"📊 *Total Active Tasks: {len(tasks)}*\n",
    ]

    ordered_tasks_for_buttons: list[TaskItem] = []

    # 1. Overdue section
    if overdue_tasks:
        lines.append(f"🔴 **Overdue ({len(overdue_tasks)})**")
        for t in overdue_tasks:
            lines.append(f"• {t.task_text}\n  📅 *Due: {t.due_date}* • 📝 [[{t.source_note_display}]]")
            ordered_tasks_for_buttons.append(t)
        lines.append("")

    # 2. Due Today section
    if today_tasks:
        lines.append(f"🟡 **Due Today ({today_str}) ({len(today_tasks)})**")
        for t in today_tasks:
            due_disp = t.due_date or today_str
            lines.append(f"• {t.task_text}\n  📅 *Due: {due_disp}* • 📝 [[{t.source_note_display}]]")
            ordered_tasks_for_buttons.append(t)
        lines.append("")

    # 3. Due in Next 2 Days section
    if next_2_days_tasks:
        lines.append(f"🔵 **Due Next 2 Days ({len(next_2_days_tasks)})**")
        for t in next_2_days_tasks:
            lines.append(f"• {t.task_text}\n  📅 *Due: {t.due_date}* • 📝 [[{t.source_note_display}]]")
            ordered_tasks_for_buttons.append(t)
        lines.append("")

    # 4. Upcoming & Backlog section
    if upcoming_tasks:
        lines.append(f"⚪ **Upcoming & Backlog ({len(upcoming_tasks)})**")
        for t in upcoming_tasks:
            due_info = f" • 📅 *Due: {t.due_date}*" if t.due_date else ""
            lines.append(f"• {t.task_text}\n  📝 [[{t.source_note_display}]]{due_info}")
            ordered_tasks_for_buttons.append(t)
        lines.append("")

    lines.append("───────────────────────")
    lines.append("👇 *Tap a button below to mark any task complete:*")

    # Build action buttons (capped at 15 to stay within Telegram UI limits)
    buttons: list[list[dict[str, str]]] = []
    for idx, t in enumerate(ordered_tasks_for_buttons[:15], 1):
        target_hint = t.daily_date or t.source_note_display
        cb_data = f"done:{target_hint}:{t.task_id}"
        btn_label = f"✅ {idx}. {t.clean_text_for_button}"
        buttons.append([{"text": btn_label, "callback_data": cb_data}])

    return "\n".join(lines), buttons


def format_daily_scheduled_message(
    tasks: list["TaskItem"],
    today: date | datetime | None = None,
) -> tuple[str, list[list[dict[str, str]]]]:
    """Generate 8:30 AM daily scheduled briefing message for tasks due today and next 2 days.

    Args:
        tasks: List of active TaskItem instances across the vault.
        today: Current date reference. Defaults to today's date.

    Returns:
        Tuple of (formatted_briefing_message, inline_keyboard_buttons).
    """
    if isinstance(today, datetime):
        ref_date = today.date()
    elif isinstance(today, date):
        ref_date = today
    else:
        ref_date = datetime.now().date()

    today_str = ref_date.strftime("%Y-%m-%d")
    tomorrow = ref_date + timedelta(days=1)
    tomorrow_str = tomorrow.strftime("%Y-%m-%d")
    day_after = ref_date + timedelta(days=2)
    day_after_str = day_after.strftime("%Y-%m-%d")

    weekday_today = ref_date.strftime("%A, %d %b %Y")
    weekday_tomorrow = tomorrow.strftime("%A, %d %b")
    weekday_day_after = day_after.strftime("%A, %d %b")

    # Filter tasks due today, tomorrow, day after tomorrow, and overdue
    overdue_tasks = [t for t in tasks if t.due_date and t.due_date < today_str]
    today_tasks = [t for t in tasks if t.due_date == today_str or (not t.due_date and t.daily_date == today_str)]
    tomorrow_tasks = [t for t in tasks if t.due_date == tomorrow_str]
    day_after_tasks = [t for t in tasks if t.due_date == day_after_str]

    due_soon_count = len(today_tasks) + len(tomorrow_tasks) + len(day_after_tasks)

    if due_soon_count == 0 and not overdue_tasks:
        msg = (
            "🌅 **Daily Morning Briefing (08:30 AM)**\n"
            f"📅 *{weekday_today}*\n"
            "───────────────────────\n"
            "✨ **All clear!** You have no tasks due for today or the next 2 days.\n"
            "🚀 Have a productive and great day!"
        )
        return msg, []

    lines: list[str] = [
        "🌅 **Daily Morning Briefing (08:30 AM)**",
        f"📅 *{weekday_today}*",
        "───────────────────────",
        f"🎯 **Tasks Due Today & Next 2 Days ({due_soon_count}):**\n",
    ]

    actionable_tasks: list[TaskItem] = []

    # 1. Overdue section (if any)
    if overdue_tasks:
        lines.append(f"🔴 **Overdue Tasks ({len(overdue_tasks)})**")
        for t in overdue_tasks:
            lines.append(f"• {t.task_text}\n  📅 *Due: {t.due_date}* • 📝 [[{t.source_note_display}]]")
            actionable_tasks.append(t)
        lines.append("")

    # 2. Today's tasks
    if today_tasks:
        lines.append(f"🟡 **Today ({today_str}) ({len(today_tasks)})**")
        for t in today_tasks:
            lines.append(f"• {t.task_text} — 📝 [[{t.source_note_display}]]")
            actionable_tasks.append(t)
        lines.append("")
    else:
        lines.append(f"🟡 **Today ({today_str})**: ✨ *No tasks due today*")
        lines.append("")

    # 3. Tomorrow's tasks
    if tomorrow_tasks:
        lines.append(f"🔵 **Tomorrow — {weekday_tomorrow} ({len(tomorrow_tasks)})**")
        for t in tomorrow_tasks:
            lines.append(f"• {t.task_text} — 📝 [[{t.source_note_display}]]")
            actionable_tasks.append(t)
        lines.append("")

    # 4. Day After Tomorrow tasks
    if day_after_tasks:
        lines.append(f"⚪ **{weekday_day_after} ({len(day_after_tasks)})**")
        for t in day_after_tasks:
            lines.append(f"• {t.task_text} — 📝 [[{t.source_note_display}]]")
            actionable_tasks.append(t)
        lines.append("")

    lines.append("───────────────────────")
    lines.append("✨ *Mark tasks complete as you finish them:*")

    # Build action buttons
    buttons: list[list[dict[str, str]]] = []
    for idx, t in enumerate(actionable_tasks[:15], 1):
        target_hint = t.daily_date or t.source_note_display
        cb_data = f"done:{target_hint}:{t.task_id}"
        btn_label = f"✅ {idx}. {t.clean_text_for_button}"
        buttons.append([{"text": btn_label, "callback_data": cb_data}])

    return "\n".join(lines), buttons


def is_task_query_intent(text: str) -> bool:
    """Check if message is inquiring about pending tasks, to-dos, or morning task briefing.

    Args:
        text: Incoming user message text.

    Returns:
        True if intent matches a task inquiry or briefing command, False otherwise.
    """
    t = text.lower().strip()
    # 1. Exact command matches
    if t in (
        "/tasks", "/task", "/todo", "/todos", "/pending", "/due",
        "/briefing", "/digest", "/schedule", "/today",
    ):
        return True

    # 2. Command prefix matches
    if t.startswith(("/tasks", "/todo", "/pending", "/due", "/briefing", "/digest")):
        return True

    # 3. Natural language query patterns
    patterns = [
        r"\b(what('s| is| are)? (my|the|all)? ?(pending|active|due)? ?(tasks|todos|to-dos|action items))\b",
        r"\b(show|list|get|view|give|check|display) (me )?(all )?(my )?(pending |due |active )?(tasks|todos|to-dos|action items)\b",
        r"\b(pending|active|due) (tasks|todos|to-dos|action items)\b",
        r"\b(what('s| is| are) due (today|tomorrow|soon))\b",
        r"\b(what do i have due)\b",
        r"\b(any (pending )?(tasks|todos|to-dos) (due )?(today|soon)?)\b",
    ]
    for p in patterns:
        if re.search(p, t):
            return True

    return False


def format_obsidian_for_telegram(text: str) -> str:
    """Format and polish raw Obsidian markdown into clean Telegram message markdown."""
    if not text:
        return text

    # 1. Normalize obsidian markdown first (fix single brackets, dataview fields, etc.)
    from src.agents.parser import normalize_obsidian_markdown
    formatted = normalize_obsidian_markdown(text)

    # 2. Replace horizontal rules with visual divider line
    formatted = re.sub(r"^(?:---|\*\*\*|___)\s*$", "───────────────────────", formatted, flags=re.MULTILINE)

    # 3. Convert markdown headers to clean styled headers for Telegram
    # ### Heading -> 🔹 *Heading*
    formatted = re.sub(r"^###\s+(.+)$", r"🔹 *\1*", formatted, flags=re.MULTILINE)
    # ## Heading -> 📌 *Heading*
    formatted = re.sub(r"^##\s+(.+)$", r"📌 *\1*", formatted, flags=re.MULTILINE)
    # # Heading -> 🏆 *Heading*
    formatted = re.sub(r"^#\s+(.+)$", r"🏆 *\1*", formatted, flags=re.MULTILINE)

    # 4. Convert callouts to styled alert blocks
    formatted = re.sub(r"^>\s*\[!NOTE\]\s*(.*)$", r"ℹ️ *Note: \1*", formatted, flags=re.MULTILINE)
    formatted = re.sub(r"^>\s*\[!WARNING\]\s*(.*)$", r"⚠️ *Warning: \1*", formatted, flags=re.MULTILINE)
    formatted = re.sub(r"^>\s*\[!TIP\]\s*(.*)$", r"💡 *Tip: \1*", formatted, flags=re.MULTILINE)
    formatted = re.sub(r"^>\s*\[!IMPORTANT\]\s*(.*)$", r"❗ *Important: \1*", formatted, flags=re.MULTILINE)
    formatted = re.sub(r"^>\s*\[!CAUTION\]\s*(.*)$", r"🛑 *Caution: \1*", formatted, flags=re.MULTILINE)

    # 5. Fix bullet list markers for loose lines
    formatted = re.sub(r"^\s+([A-Za-z0-9_\-\s]+:)\s*$", r"• *\1*", formatted, flags=re.MULTILINE)
    formatted = re.sub(r"^\s{1,2}(?=[A-Za-z0-9_\[])", r"• ", formatted, flags=re.MULTILINE)

    return formatted


def convert_markdown_to_telegram_html(text: str) -> str:
    """Convert Markdown / Obsidian text into safe Telegram-supported HTML.

    Telegram HTML supports:
      <b>bold</b>, <i>italic</i>, <code>inline code</code>, <pre>preformatted code</pre>,
      <s>strikethrough</s>, <u>underline</u>, <blockquote>quote</blockquote>, <a href="...">link</a>.
    All brackets [[WikiLink]] and special characters are preserved safely as text without entity parsing errors.
    """
    if not text:
        return text

    # Pre-process obsidian structure
    text = format_obsidian_for_telegram(text)

    # Extract code blocks to protect them from HTML escaping and tag transformations
    code_blocks: list[str] = []
    def save_code_block(match: re.Match) -> str:
        code = match.group(2)
        escaped_code = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        idx = len(code_blocks)
        code_blocks.append(f"<pre><code>{escaped_code}</code></pre>")
        return f"___CODE_BLOCK_{idx}___"

    text = re.sub(r"```([a-zA-Z0-9_\-\+]*)\n([\s\S]*?)```", save_code_block, text)

    # Extract inline code
    inline_codes: list[str] = []
    def save_inline_code(match: re.Match) -> str:
        code = match.group(1)
        escaped_code = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        idx = len(inline_codes)
        inline_codes.append(f"<code>{escaped_code}</code>")
        return f"___INLINE_CODE_{idx}___"

    text = re.sub(r"`([^`\n]+)`", save_inline_code, text)

    # HTML escape standard text
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Markdown links: [Title](url) -> <a href="url">Title</a>
    text = re.sub(r"\[([^\]\n]+)\]\((https?:\/\/[^\s\)]+)\)", r'<a href="\2">\1</a>', text)

    # Bold: **text** -> <b>text</b>
    text = re.sub(r"\*\*([^\*\n]+)\*\*", r"<b>\1</b>", text)
    # Bold: *text* -> <b>text</b>
    text = re.sub(r"(?<![\w\*])\*([^\*\n]+)\*(?![\w\*])", r"<b>\1</b>", text)

    # Italic: _text_ -> <i>text</i>
    text = re.sub(r"(?<![\w_])_([^_\n]+)_(?![\w_])", r"<i>\1</i>", text)

    # Strikethrough: ~~text~~ -> <s>text</s>
    text = re.sub(r"~~([^~\n]+)~~", r"<s>\1</s>", text)

    # Restore code blocks and inline code
    for idx, block in enumerate(code_blocks):
        text = text.replace(f"___CODE_BLOCK_{idx}___", block)
    for idx, inline in enumerate(inline_codes):
        text = text.replace(f"___INLINE_CODE_{idx}___", inline)

    return text


__all__ = [
    "format_pending_tasks_message",
    "format_daily_scheduled_message",
    "is_task_query_intent",
    "format_obsidian_for_telegram",
    "convert_markdown_to_telegram_html",
]
