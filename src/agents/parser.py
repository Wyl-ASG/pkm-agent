"""Entry parser agent for turning raw text inputs into structured InterstitialEntry models with grounded WikiLinks."""

from datetime import datetime
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

1. WIKILINKS & ALIASES:
   - Identify all technical terms, services, platforms, software tools, frameworks, and key concepts, and wrap them in Obsidian DOUBLE-BRACKET WikiLinks (e.g. [[FastAPI]], [[Qdrant]], [[Docker]]).
   - If a list of 'Existing Obsidian Vault Notes' is provided, PRIORITIZE linking to those exact note titles (using aliases if phrasing differs, e.g. [[AWS Architecture|AWS setup]]). Do not invent redundant new titles when an existing note already matches the concept.
   - Use aliased links [[Actual Note Name|Display Title]] when transforming jargon, synonyms, or abbreviations (e.g. [[Amazon Web Services|AWS]], [[Application Load Balancer|ALB]], [[Amazon Elastic Container Service|ECS]]).
   - NEVER use single brackets for WikiLinks or aliases (e.g. NEVER output [AWS] or [Web Application|web application]; ALWAYS output [[Amazon Web Services|AWS]] or [[Web Application|web application]]).
   - Strictly format WikiLinks without broken or nested brackets (NEVER output [[[Concept]]]).
   - In 'extracted_wikilinks', extract the target note name without the alias (e.g. 'Amazon Web Services' from '[[Amazon Web Services|AWS]]').

2. DATAVIEW INLINE KEYS:
   - Support inline properties formatted strictly as [key:: value] with brackets (e.g. [category:: work], [platform:: AWS], [status:: blocked]).
   - Do NOT omit brackets around inline Dataview fields in text.
   - Extract key-value pairs into the 'dataview_fields' map.

3. BLOCK IDENTIFIERS:
   - If appropriate for referencing or transclusion, append a block anchor ^block-id at the end of the entry (e.g. ^log-20260815-0208, ^task-01).
   - Set 'block_id' to the identifier string without the leading carat (e.g. 'log-20260815-0208').

4. CALLOUTS:
   - Wrap urgent alerts, critical warnings, key tips, or AI summaries in valid Obsidian callout syntax:
     > [!WARNING] Optional Title
     > Content of the warning...
     or
     > [!NOTE] Optional Title
     > Summary or key insight...
   - Valid callout types: NOTE, WARNING, TIP, IMPORTANT, CAUTION, INFO.
   - MUST include square brackets around ![TYPE], e.g. '> [!WARNING]', NEVER '> !WARNING'.
   - Set 'callout_type' to the uppercase callout keyword if applicable.

5. OBSIDIAN TASKS FORMATTING:
   - If the entry is a task or action item (category = 'task'):
     Format 'content' strictly as: - [ ] {Task Name with [[WikiLinks]]} ➕ YYYY-MM-DD 📅 YYYY-MM-DD
     where '- [ ] ' is the Markdown task checkbox, '➕ YYYY-MM-DD' is the Created Date and '📅 YYYY-MM-DD' is the Due Date.
     NEVER omit the checkbox '- [ ]'.
   - Set 'due_date' to the due date 'YYYY-MM-DD'.

6. TAGS & CATEGORIZATION & MEMORY TYPE:
   - Extract all topic hashtags (e.g. '#aws', '#infrastructure', '#cloud', '#networking') into 'extracted_tags'.
   - Select an appropriate category (e.g. 'work', 'thought', 'task', 'journal', 'discovery', 'inbox').
   - Classify 'memory_type' into: 'fact', 'observation', 'decision', 'task', or 'ai_inference'.

7. ATOMIC NOTES & ZETTELKASTEN:
   - If input contains an architectural insight, standalone concept, or major decision requiring an independent note:
     Set 'requires_atomic_note' = True, provide 'atomic_note_title', 'atomic_note_confidence' (0.0 to 1.0), 'atomic_note_reason', and write clean Markdown body content for 'atomic_note_content'.
   - Otherwise, set 'requires_atomic_note' = False, 'atomic_note_title' = None, and 'atomic_note_content' = None.

8. TIMESTAMP:
   - Set 'timestamp' to the provided input timestamp or current time in 'YYYY-MM-DD HH:MM' format.
"""


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
    # Converts [Target] to [[Target]], while preserving markdown links [Title](url), tasks [ ], [x], callouts [!NOTE], dataview [k::v]
    def fix_single_bracket_link(match: re.Match) -> str:
        content = match.group(1).strip()
        if content in ("", " ", "x", "X", "/", "-") or content.startswith("!"):
            return match.group(0)
        if "::" in content:
            return match.group(0)
        if content.startswith("^"):
            return match.group(0)
        # Preserve numeric citations like [1], [2] unless it's a date like 2026-08-15
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

    created_match = re.search(r"➕\s*(\d{4}-\d{2}-\d{2})", content)
    due_match = re.search(r"📅\s*(\d{4}-\d{2}-\d{2})", content)

    created_date = created_match.group(1) if created_match else date_part
    final_due_date = due_match.group(1) if due_match else (due_date or date_part)

    task_body = content.strip()
    task_body = re.sub(r"^[-*+]\s*(\[\s*[xX]?\s*\])?\s*", "", task_body)
    task_body = re.sub(r"➕\s*\d{4}-\d{2}-\d{2}", "", task_body)
    task_body = re.sub(r"📅\s*\d{4}-\d{2}-\d{2}", "", task_body)
    task_body = task_body.strip()

    task_body = normalize_obsidian_markdown(task_body)

    return f"- [ ] {task_body} ➕ {created_date} 📅 {final_due_date}"


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

            # Defensive enforcement of Obsidian Tasks syntax if category is task
            is_task = entry.category.lower().strip() in (
                "task", "tasks", "priority", "priorities", "todo"
            )
            if is_task:
                entry.content = enforce_task_syntax(
                    content=entry.content,
                    timestamp_str=entry.timestamp,
                    due_date=entry.due_date,
                )
                due_match = re.search(r"📅\s*(\d{4}-\d{2}-\d{2})", entry.content)
                if due_match:
                    entry.due_date = due_match.group(1)

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
