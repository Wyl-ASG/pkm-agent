"""Knowledge consolidation and evolution analysis engine for non-destructive maintenance."""

from datetime import datetime
from difflib import SequenceMatcher
import logging
from pathlib import Path
import re
from typing import Any

from src.agents.models import ConsolidationProposal
from src.config import settings
from src.graphrag.graph import VaultKnowledgeGraph
from src.llm.base import LLMProvider
from src.llm.factory import get_llm_provider

logger = logging.getLogger(__name__)

CONSOLIDATION_SYSTEM_PROMPT = """You are an expert PKM Knowledge Maintenance Agent for an Obsidian vault.
Analyze the provided notes and identify:
1. Repeated concepts that appear across multiple notes.
2. Knowledge evolution: shifts in thinking, evolving beliefs, or changing decisions over time (e.g., earlier vs later observations).
3. Potential contradictions or conflicting statements across notes.
4. Unresolved questions mentioned in the notes.

Format your output into structured JSON matching the ConsolidationProposal schema.
Do NOT suggest destructive deletions. Suggest helpful consolidations or review items."""


class KnowledgeConsolidator:
    """Non-destructive periodic/manual knowledge consolidation and evolution engine."""

    def __init__(
        self,
        vault_path: str | Path | None = None,
        knowledge_graph: VaultKnowledgeGraph | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        """Initialize KnowledgeConsolidator."""
        self.vault_path = Path(vault_path or settings.VAULT_PATH).resolve()
        self.knowledge_graph = knowledge_graph or VaultKnowledgeGraph(self.vault_path)
        self.llm = llm or get_llm_provider()

    def find_potential_duplicate_notes(self) -> list[dict[str, str]]:
        """Identify note titles with high lexical similarity using token inverted indexing (O(N*k) instead of O(N^2))."""
        note_titles = list(self.knowledge_graph.nodes.keys())
        duplicates: list[dict[str, str]] = []
        seen_pairs = set()

        # Build token -> title index for candidate pairing
        token_to_titles: dict[str, list[str]] = {}
        for t in note_titles:
            words = set(re.findall(r"\w+", t.lower()))
            for w in words:
                if len(w) > 2:  # Skip 1-2 char words
                    token_to_titles.setdefault(w, []).append(t)

        candidate_pairs: set[tuple[str, str]] = set()
        for titles_with_token in token_to_titles.values():
            if 1 < len(titles_with_token) <= 50:
                for i in range(len(titles_with_token)):
                    for j in range(i + 1, len(titles_with_token)):
                        t1, t2 = titles_with_token[i], titles_with_token[j]
                        if t1.lower() != t2.lower():
                            candidate_pairs.add(tuple(sorted([t1, t2])))

        for t1, t2 in candidate_pairs:
            w1 = set(re.findall(r"\w+", t1.lower()))
            w2 = set(re.findall(r"\w+", t2.lower()))
            is_subset = len(w1) >= 2 and (w1.issubset(w2) or w2.issubset(w1))

            sim = SequenceMatcher(None, t1.lower(), t2.lower()).ratio()
            if sim >= 0.70 or is_subset:
                pair_key = (t1, t2)
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    duplicates.append({
                        "note_a": t1,
                        "note_b": t2,
                        "similarity": f"{sim:.2f}",
                        "reason": f"Potential conceptual duplicate between [[{t1}]] and [[{t2}]]",
                    })

        return duplicates

    def _prepare_consolidation_context(
        self,
        recent_days: int = 14,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Synchronously scan graph, duplicates, and note excerpts on a worker thread."""
        duplicates = self.find_potential_duplicate_notes()

        scored_nodes = []
        now_ts = datetime.now().timestamp()
        cutoff_ts = now_ts - (recent_days * 86400)

        for title, node in self.knowledge_graph.nodes.items():
            file_path = self.vault_path / str(node.rel_path).lstrip('/')
            if file_path.exists():
                try:
                    mtime = file_path.stat().st_mtime
                    is_recent = mtime >= cutoff_ts
                    scored_nodes.append((mtime, is_recent, title, node, file_path))
                except OSError:
                    continue

        scored_nodes.sort(key=lambda x: x[0], reverse=True)

        sample_texts = []
        for _, _, title, node, file_path in scored_nodes[:30]:
            try:
                content = file_path.read_text(encoding="utf-8")[:600]
                sample_texts.append(f"### Note: [[{title}]] ({node.created or 'Date Unknown'})\n{content}\n")
            except (OSError, UnicodeError) as read_err:
                logger.warning("Could not read note file %s for consolidation: %s", file_path, read_err)

        return duplicates, sample_texts

    async def generate_consolidation_report(
        self,
        recent_days: int = 14,
    ) -> ConsolidationProposal:
        """Analyze vault notes and generate a comprehensive consolidation and evolution report."""
        await asyncio.to_thread(self.knowledge_graph.build_graph)

        duplicates, sample_texts = await asyncio.to_thread(
            self._prepare_consolidation_context, recent_days
        )

        if not sample_texts:
            return ConsolidationProposal(
                summary_markdown="✨ **Vault Maintenance Report**: No notes available for consolidation analysis."
            )

        prompt = (
            "Analyze the following vault notes for repeated concepts, knowledge evolution over time, "
            "and unresolved questions:\n\n" + "\n".join(sample_texts)
        )

        try:
            proposal = await self.llm.generate_json(
                prompt=prompt,
                schema_model=ConsolidationProposal,
                system_prompt=CONSOLIDATION_SYSTEM_PROMPT,
            )

            # Merge detected duplicates
            if duplicates:
                proposal.potential_duplicates.extend(duplicates)

            # Format markdown report
            report_lines = [
                "🧠 **Second Brain Knowledge Maintenance & Evolution Report**",
                "────────────────────────────────────────────────────────",
            ]

            if proposal.repeated_concepts:
                report_lines.append("\n📈 **Emerging & Frequently Referenced Concepts:**")
                for c in proposal.repeated_concepts:
                    report_lines.append(f"• [[{c}]]")

            if proposal.potential_duplicates:
                report_lines.append("\n🔍 **Potential Duplicate Notes:**")
                for d in proposal.potential_duplicates[:5]:
                    report_lines.append(f"• [[{d.get('note_a')}]] ↔ [[{d.get('note_b')}]] ({d.get('reason', '')})")

            if proposal.knowledge_evolutions:
                report_lines.append("\n🔄 **Knowledge & Decision Evolution Detected:**")
                for ev in proposal.knowledge_evolutions:
                    topic = ev.get("topic", "Evolution")
                    earlier = ev.get("earlier", "")
                    latest = ev.get("latest", "")
                    report_lines.append(f"• **{topic}**:\n  - Earlier: *{earlier}*\n  - Latest: *{latest}*")

            if proposal.unresolved_questions:
                report_lines.append("\n❓ **Unresolved Inquiries:**")
                for q in proposal.unresolved_questions:
                    report_lines.append(f"• {q}")

            report_lines.append("\n────────────────────────────────────────────────────────")
            report_lines.append("💡 *This report is purely advisory. No notes have been modified.*")

            proposal.summary_markdown = "\n".join(report_lines)
            return proposal

        except Exception as err:
            logger.exception("Failed generating consolidation report: %s", err)
            fallback_md = "⚠️ Knowledge consolidation analysis encountered an error."
            return ConsolidationProposal(
                potential_duplicates=duplicates,
                summary_markdown=fallback_md,
            )


__all__ = ["KnowledgeConsolidator"]
