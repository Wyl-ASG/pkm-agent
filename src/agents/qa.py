"""Knowledge base question-answering agent performing multi-stage hybrid search, block provenance, and LLM synthesis."""

import logging
import re
from typing import Any

from src.agents.models import QueryResponse, SourceCitation
from src.agents.parser import normalize_obsidian_markdown
from src.config import settings
from src.graphrag.embedder import TextEmbedder
from src.graphrag.retriever import HybridRetriever
from src.graphrag.vector_db import QdrantVectorStore, get_vector_store
from src.llm.base import LLMProvider
from src.llm.factory import get_llm_provider

logger = logging.getLogger(__name__)

QA_SYSTEM_PROMPT = """You are an expert Personal Knowledge Management (PKM) AI assistant for an Obsidian vault.
Answer the user's question clearly, structured, and concisely using the provided vault context.

CRITICAL ANSWERING RULES:
1. Ground every statement strictly in the provided Vault Context. Never invent facts or hallucinate external information.
2. If the context does NOT contain enough information to answer, state clearly:
   "I couldn't find evidence in your vault that confirms this."
3. Distinguish direct user evidence from inference: if making a deduction, explicitly state it as an inference.
4. STRUCTURE & FORMATTING:
   - Format lists with standard Markdown bullet points ('- ') or numbers ('1. ') for every item and sub-item. Never use bare indented text without bullet markers.
   - Use clear bold section titles (e.g. '### 1. Core Infrastructure' or '**1. Core Infrastructure**').
5. WIKILINKS SYNTAX:
   - Wrap technical terms, cloud services, concepts, and note references in Obsidian DOUBLE-BRACKET WikiLinks (e.g. [[Microsoft Azure]], [[Azure Functions]], [[Serverless Computing|Serverless]]).
   - NEVER output single brackets for WikiLinks or aliases (NEVER output [Azure] or [Serverless|serverless]).
6. Sources and fine-grained citations will be automatically appended by the provenance engine, so do not invent a Sources section."""


class KnowledgeBaseQAAgent:
    """Agent that handles natural language questions using multi-stage hybrid retrieval and LLM synthesis."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        embedder: TextEmbedder | None = None,
        vector_store: QdrantVectorStore | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        """Initialize KnowledgeBaseQAAgent."""
        self.embedder = embedder or TextEmbedder()
        self.vector_store = vector_store or get_vector_store()
        self.retriever = retriever or HybridRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
        )
        self.llm = llm or get_llm_provider()

    async def query(
        self,
        query_text: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        expand_graph: bool = True,
    ) -> QueryResponse:
        """Process search query, perform hybrid multi-stage retrieval, and synthesize answer with citations."""
        clean_query = re.sub(r"^/(?:ask|query)\s*", "", query_text, flags=re.IGNORECASE).strip()
        if not clean_query:
            return QueryResponse(
                query=query_text,
                answer="Please provide a query after `/ask` (e.g., `/ask How do I set up a web application on aws?`).",
                sources=[],
                citations=[],
                context_chunks=[],
            )

        target_top_k = top_k or getattr(settings, "RETRIEVAL_FINAL_TOP_K", 5)
        logger.info("Executing knowledge base query: '%s' (top_k=%d)", clean_query, target_top_k)

        try:
            # 1. Multi-stage hybrid search
            search_results = await self.retriever.search(
                query=clean_query,
                top_k=target_top_k,
                filters=filters,
                expand_graph=expand_graph,
            )

            context_blocks: list[str] = []
            context_chunks: list[dict[str, Any]] = []
            citations: list[SourceCitation] = []
            sources: list[str] = []

            for idx, res in enumerate(search_results, start=1):
                payload = res.get("payload", {})
                content = res.get("content") or payload.get("content", "")
                
                note_title = (
                    payload.get("title")
                    or payload.get("atomic_note")
                    or payload.get("daily_note")
                    or payload.get("file_name", f"Note #{idx}").replace(".md", "")
                )
                clean_title = str(note_title).replace(".md", "").strip()
                note_path = payload.get("file_path", f"Notes/{clean_title}.md")
                heading = payload.get("heading")
                heading_path = payload.get("heading_path", [])
                block_ids = payload.get("block_ids", [])
                first_block_id = block_ids[0] if block_ids else None
                memory_type = payload.get("memory_type", "user_authored")
                source_type = payload.get("source_type", "user_authored")

                wikilink_title = f"[[{clean_title}]]"
                if wikilink_title not in sources and clean_title:
                    sources.append(wikilink_title)

                citation = SourceCitation(
                    note_title=clean_title,
                    note_path=note_path,
                    heading=heading,
                    heading_path=heading_path,
                    block_id=first_block_id,
                    snippet=content[:150] if content else None,
                    memory_type=memory_type,
                    source_type=source_type,
                )
                citations.append(citation)

                # Context presentation
                header_info = f" -> ## {heading}" if heading and heading != clean_title else ""
                block_info = f" ({first_block_id})" if first_block_id else ""
                context_blocks.append(
                    f"--- Source [{idx}]: [[{clean_title}]]{header_info}{block_info} ---\n{content}"
                )

                context_chunks.append({
                    "source": clean_title,
                    "file_path": note_path,
                    "heading": heading or "",
                    "block_id": first_block_id or "",
                    "content": content,
                    "score": str(res.get("rerank_score", res.get("rrf_score", res.get("score", 0.0)))),
                })

            if not context_blocks:
                return QueryResponse(
                    query=clean_query,
                    answer="I couldn't find evidence in your vault that confirms this.",
                    sources=[],
                    citations=[],
                    context_chunks=[],
                )

            context_str = "\n\n".join(context_blocks)
            user_prompt = (
                f"User Question:\n{clean_query}\n\n"
                f"Retrieved Vault Context:\n{context_str}\n"
            )

            raw_answer = await self.llm.generate_text(
                prompt=user_prompt,
                system_prompt=QA_SYSTEM_PROMPT,
            )
            normalized_answer = normalize_obsidian_markdown(raw_answer.strip())

            # Append block-level provenance citations section
            if getattr(settings, "PROVENANCE_ENABLED", True) and citations:
                citation_lines = ["\n\n**Sources:**"]
                seen_citations = set()
                for c in citations:
                    c_formatted = c.format_citation()
                    if c_formatted not in seen_citations:
                        seen_citations.add(c_formatted)
                        citation_lines.append(c_formatted)
                
                normalized_answer += "\n" + "\n".join(citation_lines)

            return QueryResponse(
                query=clean_query,
                answer=normalized_answer,
                sources=sources,
                citations=citations,
                context_chunks=context_chunks,
            )

        except Exception as err:
            logger.exception("Failed to query knowledge base for '%s': %s", clean_query, err)
            return QueryResponse(
                query=clean_query,
                answer="⚠️ Sorry, an error occurred while processing your query.",
                sources=[],
                citations=[],
                context_chunks=[],
            )


async def query_knowledge_base(
    query_text: str,
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
    expand_graph: bool = True,
    retriever: HybridRetriever | None = None,
    embedder: TextEmbedder | None = None,
    vector_store: QdrantVectorStore | None = None,
    llm: LLMProvider | None = None,
) -> QueryResponse:
    """Convenience function to query knowledge base."""
    agent = KnowledgeBaseQAAgent(
        retriever=retriever,
        embedder=embedder,
        vector_store=vector_store,
        llm=llm,
    )
    return await agent.query(
        query_text=query_text,
        top_k=top_k,
        filters=filters,
        expand_graph=expand_graph,
    )


__all__ = [
    "KnowledgeBaseQAAgent",
    "query_knowledge_base",
    "QA_SYSTEM_PROMPT",
]
