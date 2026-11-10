"""Entry parser agent for turning raw text inputs into structured InterstitialEntry models with grounded WikiLinks."""

from datetime import datetime, timedelta
import logging
import re
from src.agents.models import InterstitialEntry, normalize_category
from src.config import settings
from src.graphrag.resolver import WikiLinkResolver
from src.llm.base import LLMProvider
from src.llm.factory import get_llm_provider

logger = logging.getLogger(__name__)

PARSER_SYSTEM_PROMPT = """You are an expert Personal Knowledge Management (PKM) entry parser for an Obsidian vault.
Your task is to parse raw user text into a structured InterstitialEntry object adhering strictly to the Obsidian Feature Rulebook.

CRITICAL INSTRUCTIONS FOR CONTENT FORMATTING & OBSIDIAN SYNTAX:

0. PRESERVE THE USER'S WORDS:
   - Copy the user's exact text into 'content'. Do not add links, checkboxes, dates, or any other formatting. The system adds these itself.

1. NATURAL LANGUAGE TASK DETECTION:
   - Use category='task' for a future action the user intends to do, including short to-do phrases (e.g. 'I need to file my taxes', 'Remember to renew the passport', 'Clean the garage').
   - Ask: is this something the user personally still has to do? If not, it is not a task.
   - Knowledge, facts, explanations, observations, and past events are NOT tasks, even when they contain words like 'must', 'should', 'have to', 'due', or a date (e.g. 'Rust values must have one owner', 'Paid the invoice on 3 May').
   - General rules and best practices are knowledge, not tasks (e.g. 'Backups must be tested regularly').
   - Decisions already made are not tasks (e.g. 'Chose Rust over Go for the CLI').
   - DO NOT output assistant fallback phrases like 'Awaiting user instructions', 'Session initiated', or 'Ready to assist'.
   - Set 'due_date' as YYYY-MM-DD only when the user stated a deadline. Never invent a due date.

2. WIKILINKS:
   - Do not create links. Leave 'extracted_wikilinks' empty.

3. DATAVIEW INLINE KEYS:
   - Extract only [key:: value] properties the user wrote into the 'dataview_fields' map. Do not invent properties.

4. BLOCK IDENTIFIERS:
   - Do not create block anchors. Leave 'block_id' empty.

5. CALLOUTS:
   - For urgent alerts or critical warnings, set 'callout_type' to 'WARNING' or 'IMPORTANT'. Otherwise leave it empty.

6. TAGS & CATEGORIZATION & MEMORY TYPE:
   - Extract all topic hashtags (e.g. '#aws', '#infrastructure') into 'extracted_tags'.
   - Choose exactly one category:
     'task' = a future action the user intends to do.
     'discovery' = knowledge, a technical fact, or something learned, including fragments that explain how something works.
     'log' = something that happened, or a passing thought.
     'inbox' = you cannot tell which of the others applies.
   - Classify 'memory_type' into: 'fact', 'observation', 'decision', 'task', or 'ai_inference'.

7. ATOMIC NOTES & ZETTELKASTEN:
   - Most captures do NOT need an atomic note. Set 'requires_atomic_note' = True only for a substantial, self-contained concept or major decision that makes sense alone, and provide 'atomic_note_title' and 'atomic_note_content'.
   - Short facts, statistics, fragments, and tasks do not qualify.
   - Set 'atomic_note_confidence' honestly from 0.0 to 1.0. Use 0.85 or more only when the input alone clearly justifies a standalone note.

8. TIMESTAMP:
   - Set 'timestamp' to the provided input timestamp or current time in 'YYYY-MM-DD HH:MM' format.
"""


def extract_due_date_from_natural_language(text: str, ref_date: datetime | None = None) -> str | None:
    """Extract YYYY-MM-DD due date from natural language time expressions like 'by 30 aug', 'by Sunday', 'by tomorrow'."""
    now = ref_date or datetime.now()
    t = text.lower()

    # 1. Explicit ISO date format: YYYY-MM-DD or YYYY/MM/DD
    iso_match = re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", t)
    if iso_match:
        y, m, d = int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3))
        try:
            return datetime(y, m, d).strftime("%Y-%m-%d")
        except ValueError:
            pass

    # 2. Relative keywords
    if re.search(r"\btomorrow\b", t):
        return (now + timedelta(days=1)).strftime("%Y-%m-%d")
    if re.search(r"\btoday\b", t):
        return now.strftime("%Y-%m-%d")

    months_map = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }
    months_regex = "|".join(sorted(months_map.keys(), key=lambda x: -len(x)))

    # 3. Day + Month: e.g. "30 aug", "30th august", "30 aug 2026", "by 30th of august"
    day_month_match = re.search(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({months_regex})\b(?:,?\s+(\d{{4}}))?",
        t,
    )
    if day_month_match:
        d = int(day_month_match.group(1))
        m = months_map[day_month_match.group(2)]
        y = int(day_month_match.group(3)) if day_month_match.group(3) else now.year
        if not day_month_match.group(3) and (m < now.month or (m == now.month and d < now.day)):
            try:
                dt_temp = datetime(y, m, d)
                if (now - dt_temp).days > 7:
                    y += 1
            except ValueError:
                pass
        try:
            return datetime(y, m, d).strftime("%Y-%m-%d")
        except ValueError:
            pass

    # 4. Month + Day: e.g. "aug 30", "august 30th", "aug 30 2026"
    month_day_match = re.search(
        rf"\b({months_regex})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?\b",
        t,
    )
    if month_day_match:
        m = months_map[month_day_match.group(1)]
        d = int(month_day_match.group(2))
        y = int(month_day_match.group(3)) if month_day_match.group(3) else now.year
        if not month_day_match.group(3) and (m < now.month or (m == now.month and d < now.day)):
            try:
                dt_temp = datetime(y, m, d)
                if (now - dt_temp).days > 7:
                    y += 1
            except ValueError:
                pass
        try:
            return datetime(y, m, d).strftime("%Y-%m-%d")
        except ValueError:
            pass

    # 5. Short numeric date format: DD/MM or DD-MM (e.g., 30/08 or 30-08)
    short_date_match = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{4}))?\b", t)
    if short_date_match:
        val1, val2 = int(short_date_match.group(1)), int(short_date_match.group(2))
        y = int(short_date_match.group(3)) if short_date_match.group(3) else now.year
        if val1 > 12:
            d, m = val1, val2
        elif val2 > 12:
            m, d = val1, val2
        else:
            d, m = val1, val2
        try:
            return datetime(y, m, d).strftime("%Y-%m-%d")
        except ValueError:
            pass

    # 6. Day names: e.g. "by Sunday", "on Monday"
    days_map = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
    for day_name, day_num in days_map.items():
        current_day = now.weekday()
        days_ahead = (day_num - current_day) % 7 or 7

        # "next friday" means the Friday of next week, even if one remains this week.
        if re.search(rf"\bnext\s+{day_name}\b", t):
            if day_num > current_day:
                days_ahead += 7
            return (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

        if re.search(rf"\b(?:by|on|this|due)\s+{day_name}\b", t):
            return (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    return None


EXPLICIT_COMMITMENT_PATTERN = re.compile(
    r"^\s*(?:"
    r"i(?:'m| am)?\s+(?:need|have|required|supposed)\s+to\b"
    r"|i\s+must\b"
    r"|(?:remember|don't forget)\s+to\b"
    r"|(?:todo|to-do)\b"
    r")",
    re.IGNORECASE,
)
IMPERATIVE_WITH_DEADLINE_PATTERN = re.compile(
    r"^\s*(?:finish|submit|complete|upload|send|pay|call|book|buy|email|review|prepare|write|do)\b"
    r".*\b(?:by|before|due)\b",
    re.IGNORECASE,
)


def is_natural_language_task(text: str) -> bool:
    """Detect an explicit commitment; used only when no trustworthy LLM classification exists."""
    normalized = text.replace("’", "'")
    if EXPLICIT_COMMITMENT_PATTERN.search(normalized):
        return True
    return bool(
        IMPERATIVE_WITH_DEADLINE_PATTERN.search(normalized)
        and extract_due_date_from_natural_language(normalized) is not None
    )


def normalize_obsidian_markdown(text: str) -> str:
    """Normalize Obsidian markdown syntax in text for WikiLinks, Callouts, Dataview fields, and Tasks."""
    if not text:
        return text

    normalized = text

    # 1. Fix malformed callouts: '> !WARNING Title' -> '> [!WARNING] Title'
    normalized = re.sub(
        r"^>\s*!([A-Z]+)\b(.*)$",
        r"> [!\1]\2",
        normalized,
        flags=re.MULTILINE,
    )

    # 2. Fix single-bracket aliased links: '[Actual Target|Alias]' -> '[[Actual Target|Alias]]'
    normalized = re.sub(
        r"(?<!\[)\[([^\[\]\n\|]+)\|([^\[\]\n]+)\](?!\])",
        r"[[\1|\2]]",
        normalized,
    )

    # 3. Fix unbracketed inline Dataview fields: e.g. 'category:: work' -> '[category:: work]'
    normalized = re.sub(
        r"(?<![\[\(])\b([a-zA-Z0-9_\-]+)::\s*([a-zA-Z0-9_\-\/]+)(?![\]\)])",
        r"[\1:: \2]",
        normalized,
    )

    # 4. Fix single bracketed entities that should be WikiLinks:
    def fix_single_bracket_link(match: re.Match) -> str:
        content = match.group(1).strip()
        if content in ("", " ", "x", "X", "/", "-") or content.startswith("!"):
            return match.group(0)
        if "::" in content:
            return match.group(0)
        if content.startswith("^"):
            return match.group(0)
        if content.isdigit() or re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", content):
            return match.group(0)
        return f"[[{content}]]"

    normalized = re.sub(
        r"(?<!\[)\[([^\[\]\n\|]+)\](?!\s*[\(\]])",
        fix_single_bracket_link,
        normalized,
    )

    # 5. Fix accidental triple brackets or nested brackets: '[[[Note]]]' -> '[[Note]]'
    normalized = re.sub(r"\[{3,}([^\[\]\n]+)\]{3,}", r"[[\1]]", normalized)

    return normalized


def extract_wikilinks_from_text(text: str) -> list[str]:
    """Extract Obsidian WikiLink target note names from text, stripping aliases."""
    matches = re.findall(r"\[\[([^\]\|]+)(?:\|[^\]]+)?\]\]", text)
    return [m.strip() for m in matches if m.strip()]


def extract_block_id_from_text(text: str) -> str | None:
    """Extract Obsidian block identifier anchor from end of text string."""
    match = re.search(r"\^([a-zA-Z0-9_-]+)\s*$", text.strip())
    return match.group(1) if match else None


def extract_dataview_fields_from_text(text: str) -> dict[str, str]:
    """Extract Dataview inline fields formatted as [key:: value] from text."""
    fields: dict[str, str] = {}
    matches = re.findall(r"\[([a-zA-Z0-9_\-\s]+)::\s*([^\]]+)\]", text)
    for k, v in matches:
        fields[k.strip()] = v.strip()
    return fields


def enforce_task_syntax(content: str, timestamp_str: str, due_date: str | None = None) -> str:
    """Enforce strict Obsidian Tasks syntax: - [ ] {Task} ➕ {CreatedDate} 📅 {DueDate}."""
    date_part = timestamp_str.split(" ")[0] if " " in timestamp_str else timestamp_str[:10]

    result_lines = []
    for line in content.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        created_match = re.search(r"➕\s*(\d{4}-\d{2}-\d{2})", line)
        due_match = re.search(r"📅\s*(\d{4}-\d{2}-\d{2})", line)

        created_date = created_match.group(1) if created_match else date_part
        final_due_date = due_match.group(1) if due_match else due_date

        task_body = re.sub(r"^[-*+]\s*(\[\s*[xX]?\s*\])?\s*", "", line)
        task_body = re.sub(r"➕\s*\d{4}-\d{2}-\d{2}", "", task_body)
        task_body = re.sub(r"📅\s*\d{4}-\d{2}-\d{2}", "", task_body)
        task_body = task_body.strip()

        block_match = re.search(r"\s+(\^[A-Za-z0-9_-]+)$", task_body)
        block_suffix = f" {block_match.group(1)}" if block_match else ""
        if block_match:
            task_body = task_body[: block_match.start()]

        due_suffix = f" 📅 {final_due_date}" if final_due_date else ""
        result_lines.append(f"- [ ] {task_body} ➕ {created_date}{due_suffix}{block_suffix}")

    return "\n".join(result_lines)


_TRAILING_BLOCK_ID = re.compile(r"(?:^|\s+)\^([A-Za-z0-9_-]+)\s*$")


def _split_trailing_block_id(text: str) -> tuple[str, str | None]:
    """Separate a block anchor the user wrote at the end of their text, so it can stay last in the note."""
    match = _TRAILING_BLOCK_ID.search(text)
    if not match:
        return text, None
    return text[: match.start()].rstrip(), match.group(1)


def _parse_reference_date(timestamp: str) -> datetime:
    """Parse a capture timestamp so relative dates resolve from when the user wrote them."""
    try:
        return datetime.fromisoformat(timestamp).replace(tzinfo=None)
    except ValueError:
        return datetime.now()


class EntryParserAgent:
    """Agent that parses raw user inputs into structured InterstitialEntry models with grounded WikiLinks."""

    def __init__(
        self,
        llm: LLMProvider | None = None,
        resolver: WikiLinkResolver | None = None,
    ) -> None:
        """Initialize EntryParserAgent."""
        self.llm = llm or get_llm_provider()
        self.resolver = resolver or WikiLinkResolver()

    async def parse(
        self,
        raw_text: str,
        timestamp: str | None = None,
        category_hint: str | None = None,
        existing_notes: list[str] | None = None,
        note_aliases: dict[str, str] | None = None,
    ) -> InterstitialEntry:
        """Parse raw user input text into a validated InterstitialEntry model."""
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry_timestamp = timestamp if timestamp else current_time_str

        if existing_notes is not None:
            self.resolver.update_notes(existing_notes, note_aliases)

        user_prompt_parts = [
            f"Timestamp: {entry_timestamp}",
        ]
        if category_hint:
            user_prompt_parts.append(f"Category Hint: {category_hint}")

        user_prompt_parts.extend([
            "Raw Input Content:",
            raw_text.strip(),
        ])

        user_prompt = "\n".join(user_prompt_parts)

        logger.info("Parsing raw text into InterstitialEntry via EntryParserAgent")
        try:
            classification_trusted = True
            try:
                entry = await self.llm.generate_json(
                    prompt=user_prompt,
                    schema_model=InterstitialEntry,
                    system_prompt=PARSER_SYSTEM_PROMPT,
                )
            except Exception as parse_err:
                logger.warning("EntryParserAgent LLM structured parsing failed (%s). Falling back to raw text entry.", parse_err)
                entry = InterstitialEntry(
                    timestamp=entry_timestamp,
                    category=normalize_category(category_hint),
                    content=raw_text.strip(),
                    memory_type="observation",
                )
                classification_trusted = False

            if not entry.timestamp:
                entry.timestamp = entry_timestamp

            # Check if LLM output contains conversational meta-chatter fallback phrases
            meta_chatter_patterns = [
                r"\bawaiting user\b",
                r"\bno task was specified\b",
                r"\bready to assist\b",
                r"\bsession initiated\b",
                r"\bno specific task\b",
                r"\bawaiting task\b",
                r"\bawaiting instructions\b",
                r"\bcompleted response\b",
                r"\buser request\b",
            ]
            content_lower = (entry.content or "").lower()
            if any(re.search(pat, content_lower) for pat in meta_chatter_patterns) or not entry.content:
                logger.warning("LLM returned conversational meta-chatter fallback. Restoring raw text content.")
                entry.content = raw_text.strip()
                entry.category = normalize_category(category_hint)
                classification_trusted = False

            # The note always keeps the user's own text. The LLM only classifies it.
            entry.content, entry.block_id = _split_trailing_block_id(raw_text.strip())

            # The LLM decides what is a task; the strict heuristic only applies when it gave no usable answer.
            is_task_category = entry.category == "task"
            if not classification_trusted and is_natural_language_task(raw_text):
                is_task_category = True

            if is_task_category:
                entry.category = "task"
                entry.memory_type = "task"

                nl_due = extract_due_date_from_natural_language(
                    raw_text, ref_date=_parse_reference_date(entry_timestamp)
                )
                if nl_due:
                    entry.due_date = nl_due

                entry.content = enforce_task_syntax(
                    content=entry.content,
                    timestamp_str=entry.timestamp,
                    due_date=entry.due_date,
                )
                due_match = re.search(r"📅\s*(\d{4}-\d{2}-\d{2})", entry.content)
                if due_match:
                    entry.due_date = due_match.group(1)

            # Link mentions of notes that really exist
            entry.content = self.resolver.link_known_notes(entry.content)
            entry.extracted_wikilinks = sorted(set(extract_wikilinks_from_text(entry.content)))

            # Extract Dataview fields
            inline_dv = extract_dataview_fields_from_text(entry.content)
            if inline_dv:
                for k, v in inline_dv.items():
                    if k not in entry.dataview_fields:
                        entry.dataview_fields[k] = v

            return entry

        except Exception as err:
            logger.exception("EntryParserAgent failed to parse text: %s", err)
            raise

    def parse_sync(
        self,
        raw_text: str,
        timestamp: str | None = None,
        category_hint: str | None = None,
        existing_notes: list[str] | None = None,
    ) -> InterstitialEntry:
        """Synchronous wrapper for parse method."""
        import asyncio
        return asyncio.run(self.parse(raw_text, timestamp, category_hint, existing_notes))


__all__ = [
    "EntryParserAgent",
    "normalize_obsidian_markdown",
    "extract_wikilinks_from_text",
    "extract_block_id_from_text",
    "extract_dataview_fields_from_text",
    "enforce_task_syntax",
]
