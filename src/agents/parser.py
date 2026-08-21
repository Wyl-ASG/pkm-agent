"""Entry parser agent for turning raw text inputs into structured InterstitialEntry models with grounded WikiLinks."""

from datetime import datetime, timedelta
import logging
import re
from src.agents.models import InterstitialEntry
from src.config import settings
from src.graphrag.resolver import WikiLinkResolver
from src.llm.base import LLMProvider
from src.llm.factory import get_llm_provider

logger = logging.getLogger(__name__)

PARSER_SYSTEM_PROMPT = """You are an expert Personal Knowledge Management (PKM) entry parser for an Obsidian vault.
Your task is to parse raw user text into a structured InterstitialEntry object adhering strictly to the Obsidian Feature Rulebook.

CRITICAL INSTRUCTIONS FOR CONTENT FORMATTING & OBSIDIAN SYNTAX:

1. NATURAL LANGUAGE TASK DETECTION & FORMATTING:
   - Treat ANY natural language statement expressing a commitment, obligation, requirement, to-do item, deadline, or task (e.g. 'I am required to...', 'I need to...', 'I have to...', 'Remember to...', 'Must finish...', 'by Sunday', 'by tomorrow') as category='task'.
   - DO NOT output assistant fallback phrases like 'Awaiting user instructions', 'Session initiated', or 'Ready to assist'.
   - If category = 'task', format 'content' strictly as: - [ ] {Task Name with [[WikiLinks]]} ➕ YYYY-MM-DD 📅 YYYY-MM-DD
     where '- [ ] ' is the Markdown task checkbox, '➕ YYYY-MM-DD' is the Created Date and '📅 YYYY-MM-DD' is the Due Date.

2. WIKILINKS & ALIASES:
   - Identify all technical terms, services, platforms, software tools, frameworks, and key concepts, and wrap them in Obsidian DOUBLE-BRACKET WikiLinks (e.g. [[FastAPI]], [[Qdrant]], [[Docker]]).
   - If a list of 'Existing Obsidian Vault Notes' is provided, PRIORITIZE linking to those exact note titles (using aliases if phrasing differs, e.g. [[AWS Architecture|AWS setup]]). Do not invent redundant new titles when an existing note already matches the concept.
   - Use aliased links [[Actual Note Name|Display Title]] when transforming jargon, synonyms, or abbreviations (e.g. [[Amazon Web Services|AWS]], [[Application Load Balancer|ALB]], [[Amazon Elastic Container Service|ECS]]).
   - NEVER use single brackets for WikiLinks or aliases.
   - In 'extracted_wikilinks', extract the target note name without the alias.

3. DATAVIEW INLINE KEYS:
   - Support inline properties formatted strictly as [key:: value] with brackets (e.g. [category:: work], [platform:: AWS], [status:: blocked]).
   - Extract key-value pairs into the 'dataview_fields' map.

4. BLOCK IDENTIFIERS:
   - If appropriate for referencing or transclusion, append a block anchor ^block-id at the end of the entry (e.g. ^log-20260815-0208, ^task-01).

5. CALLOUTS:
   - Wrap urgent alerts, critical warnings, key tips, or AI summaries in valid Obsidian callout syntax (> [!NOTE], > [!WARNING]).

6. TAGS & CATEGORIZATION & MEMORY TYPE:
   - Extract all topic hashtags (e.g. '#aws', '#infrastructure') into 'extracted_tags'.
   - Select an appropriate category (e.g. 'work', 'thought', 'task', 'journal', 'discovery', 'inbox').
   - Classify 'memory_type' into: 'fact', 'observation', 'decision', 'task', or 'ai_inference'.

7. ATOMIC NOTES & ZETTELKASTEN:
   - If input contains an architectural insight, standalone concept, or major decision requiring an independent note, set 'requires_atomic_note' = True.

8. TIMESTAMP:
   - Set 'timestamp' to the provided input timestamp or current time in 'YYYY-MM-DD HH:MM' format.
"""


def extract_due_date_from_natural_language(text: str, ref_date: datetime | None = None) -> str | None:
    """Extract YYYY-MM-DD due date from natural language time expressions like 'by Sunday', 'by tomorrow'."""
    now = ref_date or datetime.now()
    t = text.lower()
    days_map = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}

    if "tomorrow" in t:
        return (now + timedelta(days=1)).strftime("%Y-%m-%d")
    if "today" in t:
        return now.strftime("%Y-%m-%d")

    for day_name, day_num in days_map.items():
        pattern = rf"\b(?:by|on|this|due)\s+{day_name}\b"
        if re.search(pattern, t):
            current_day = now.weekday()
            days_ahead = (day_num - current_day) % 7
            if days_ahead == 0:
                days_ahead = 7
            return (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    return None


def is_natural_language_task(text: str) -> bool:
    """Check if raw text string contains task/to-do indicators or deadline expressions."""
    t = text.lower()
    indicators = [
        "required to", "need to", "needs to", "have to", "has to", "must ",
        "remember to", "todo", "to-do", "should ", "due ", "finish ",
        "by sunday", "by monday", "by tuesday", "by wednesday", "by thursday",
        "by friday", "by saturday", "by tomorrow", "by today"
    ]
    return any(ind in t for ind in indicators)


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
        if content.isdigit():
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
    normalized = normalize_obsidian_markdown(text)
    matches = re.findall(r"\[\[([^\]\|]+)(?:\|[^\]]+)?\]\]", normalized)
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
        final_due_date = due_match.group(1) if due_match else (due_date or date_part)

        task_body = re.sub(r"^[-*+]\s*(\[\s*[xX]?\s*\])?\s*", "", line)
        task_body = re.sub(r"➕\s*\d{4}-\d{2}-\d{2}", "", task_body)
        task_body = re.sub(r"📅\s*\d{4}-\d{2}-\d{2}", "", task_body)
        task_body = task_body.strip()

        task_body = normalize_obsidian_markdown(task_body)
        result_lines.append(f"- [ ] {task_body} ➕ {created_date} 📅 {final_due_date}")

    return "\n".join(result_lines)


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
    ) -> InterstitialEntry:
        """Parse raw user input text into a validated InterstitialEntry model."""
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry_timestamp = timestamp if timestamp else current_time_str

        if existing_notes:
            self.resolver.update_notes(existing_notes)

        user_prompt_parts = [
            f"Timestamp: {entry_timestamp}",
        ]
        if category_hint:
            user_prompt_parts.append(f"Category Hint: {category_hint}")

        if existing_notes:
            sampled_notes = existing_notes[:80]
            notes_str = ", ".join([f"[[{n}]]" for n in sampled_notes])
            user_prompt_parts.append(f"Existing Obsidian Vault Notes to Prioritize Linking:\n{notes_str}")

        user_prompt_parts.extend([
            "Raw Input Content:",
            raw_text.strip(),
        ])

        user_prompt = "\n".join(user_prompt_parts)

        logger.info("Parsing raw text into InterstitialEntry via EntryParserAgent")
        try:
            entry = await self.llm.generate_json(
                prompt=user_prompt,
                schema_model=InterstitialEntry,
                system_prompt=PARSER_SYSTEM_PROMPT,
            )

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

            # Natural language task detection & relative date extraction
            is_nl_task = is_natural_language_task(raw_text)
            is_task_category = entry.category.lower().strip() in (
                "task", "tasks", "priority", "priorities", "todo"
            )

            if is_nl_task or is_task_category:
                entry.category = "task"
                entry.memory_type = "task"

                # Ensure task content retains the user's actual raw text if LLM modified it into generic text
                if is_nl_task:
                    entry.content = raw_text.strip()

                nl_due = extract_due_date_from_natural_language(raw_text)
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

            # Normalize markdown syntax
            entry.content = normalize_obsidian_markdown(entry.content)

            # Ground WikiLinks against existing notes
            extracted_from_content = extract_wikilinks_from_text(entry.content)
            all_links = set()
            for link in entry.extracted_wikilinks + extracted_from_content:
                clean_link = link.replace("[[", "").replace("]]", "").replace("[", "").replace("]", "")
                target = clean_link.split("|")[0].strip()
                if target:
                    canonical, score = self.resolver.resolve(target)
                    all_links.add(canonical if canonical else target)

            entry.extracted_wikilinks = sorted(all_links)

            # Extract Dataview fields
            inline_dv = extract_dataview_fields_from_text(entry.content)
            if inline_dv:
                for k, v in inline_dv.items():
                    if k not in entry.dataview_fields:
                        entry.dataview_fields[k] = v

            # Extract block ID
            if not entry.block_id:
                found_block = extract_block_id_from_text(entry.content)
                if found_block:
                    entry.block_id = found_block

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
