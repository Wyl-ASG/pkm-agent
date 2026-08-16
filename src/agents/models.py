"""Pydantic data models for agent schema validation, provenance, and LLM interactions."""

from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class MemoryType(str, Enum):
    """Classification of memory/provenance in PKM knowledge base."""

    USER_AUTHORED_FACT = "fact"
    OBSERVATION = "observation"
    DECISION = "decision"
    TASK = "task"
    AI_SUMMARY = "ai_summary"
    AI_INFERENCE = "ai_inference"
    IMPORTED_EXTERNAL = "imported_external"
    NOTE = "note"


class SourceType(str, Enum):
    """Origin of knowledge chunk."""

    USER_AUTHORED = "user_authored"
    AI_GENERATED = "ai_generated"
    IMPORTED_EXTERNAL = "imported_external"


class SourceCitation(BaseModel):
    """Fine-grained citation referencing note, heading, block ID, and memory type."""

    note_title: str = Field(description="Display title of the referenced note")
    note_path: str = Field(description="Relative vault path to the source note")
    heading: str | None = Field(default=None, description="Section heading where content resides")
    heading_path: list[str] = Field(default_factory=list, description="Hierarchy of section headings")
    block_id: str | None = Field(default=None, description="Obsidian block identifier anchor (e.g. ^abc123)")
    memory_type: MemoryType | str = Field(
        default=MemoryType.USER_AUTHORED_FACT,
        description="Memory classification",
    )
    source_type: SourceType | str = Field(
        default=SourceType.USER_AUTHORED,
        description="Origin source type",
    )

    def format_citation(self) -> str:
        """Format citation into clean Obsidian Markdown with block transclusion if available."""
        lines = [f"• [[{self.note_title}]]"]
        if self.heading and self.heading != self.note_title:
            lines.append(f"  → ## {self.heading}")
        if self.block_id:
            lines.append(f"  → {self.block_id}")
        return "\n".join(lines)


class InterstitialEntry(BaseModel):
    """Schema representing an interstitial journal entry with extracted metadata and atomic note options."""

    timestamp: str = Field(
        description="Timestamp of the entry in ISO 8601 or HH:MM format.",
    )
    category: str = Field(
        default="general",
        description="Category classification of the entry (e.g. log, task, discovery, thought).",
    )
    content: str = Field(
        description=(
            "Raw or processed markdown text content of the interstitial entry. "
            "For tasks (category='task'), format with Obsidian Tasks syntax: "
            "'- [ ] {Task Name} ➕ YYYY-MM-DD 📅 YYYY-MM-DD'."
        ),
    )
    extracted_tags: list[str] = Field(
        default_factory=list,
        description="List of tags extracted from the content (e.g. ['#pkm', '#idea']).",
    )
    extracted_wikilinks: list[str] = Field(
        default_factory=list,
        description="List of Obsidian wikilinks extracted from the content (e.g. ['[[Project]]']).",
    )
    requires_atomic_note: bool = Field(
        default=False,
        description="Flag indicating whether an independent atomic note should be generated.",
    )
    atomic_note_confidence: float = Field(
        default=0.9,
        description="Confidence score (0.0 to 1.0) regarding whether this concept warrants an atomic note.",
    )
    atomic_note_title: str | None = Field(
        default=None,
        description="Title of the standalone atomic note if requires_atomic_note is True.",
    )
    atomic_note_content: str | None = Field(
        default=None,
        description="Markdown body content of the standalone atomic note if requires_atomic_note is True.",
    )
    atomic_note_reason: str | None = Field(
        default=None,
        description="Brief explanation of why this concept warrants an atomic note.",
    )
    due_date: str | None = Field(
        default=None,
        description="Optional due date for tasks formatted as 'YYYY-MM-DD'.",
    )
    dataview_fields: dict[str, str] = Field(
        default_factory=dict,
        description="Inline Dataview key-value properties formatted as [key:: value] (e.g. {'category': 'Log', 'source': 'Telegram'}).",
    )
    block_id: str | None = Field(
        default=None,
        description="Optional block identifier anchor without leading carat (e.g. 'log-1030', 'task-01') for Obsidian block transclusion ^block-id.",
    )
    callout_type: str | None = Field(
        default=None,
        description="Optional Obsidian callout type (e.g. 'NOTE', 'WARNING', 'TIP', 'IMPORTANT', 'CAUTION') for alerts or summaries.",
    )
    memory_type: str = Field(
        default="observation",
        description="Classification of information type (fact, observation, decision, task, inference).",
    )

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )


class QueryRequest(BaseModel):
    """Schema for knowledge base query requests."""

    query: str = Field(
        description="Natural language query or question from the user.",
    )
    top_k: int = Field(
        default=5,
        description="Maximum number of context chunks or search results to retrieve.",
    )
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Optional key-value filters to restrict vector search results.",
    )
    expand_graph: bool = Field(
        default=True,
        description="Whether to perform 1-hop graph expansion for context enrichment.",
    )

    model_config = ConfigDict(
        str_strip_whitespace=True,
    )


class QueryResponse(BaseModel):
    """Schema for knowledge base query responses with fine-grained citations."""

    query: str = Field(
        description="Original natural language query string.",
    )
    answer: str = Field(
        description="Synthesized answer generated by the LLM agent based on vault context.",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="List of Obsidian file paths or wikilinks referenced to form the answer.",
    )
    citations: list[SourceCitation] = Field(
        default_factory=list,
        description="Structured citations with note path, heading, and block ID where available.",
    )
    context_chunks: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Retrieved vault context chunks with metadata used in answering.",
    )

    model_config = ConfigDict(
        str_strip_whitespace=True,
    )


class ConsolidationProposal(BaseModel):
    """Report generated by knowledge consolidation engine."""

    repeated_concepts: list[str] = Field(
        default_factory=list,
        description="Concepts or ideas appearing frequently across recent notes.",
    )
    potential_duplicates: list[dict[str, str]] = Field(
        default_factory=list,
        description="Pairs of note titles that may be duplicate concepts.",
    )
    knowledge_evolutions: list[dict[str, str]] = Field(
        default_factory=list,
        description="Detected shifts in decisions or observations over time.",
    )
    unresolved_questions: list[str] = Field(
        default_factory=list,
        description="Questions or open inquiries recorded in notes without answers.",
    )
    summary_markdown: str = Field(
        default="",
        description="Formatted Markdown report for user review.",
    )


__all__ = [
    "MemoryType",
    "SourceType",
    "SourceCitation",
    "InterstitialEntry",
    "QueryRequest",
    "QueryResponse",
    "ConsolidationProposal",
]
