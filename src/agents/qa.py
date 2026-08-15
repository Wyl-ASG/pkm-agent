"""Knowledge base question-answering agent performing hybrid search and LLM synthesis."""

import logging
import re
from typing import Any

from src.agents.models import QueryResponse
from src.agents.parser import normalize_obsidian_markdown
from src.graphrag.embedder import TextEmbedder
from src.graphrag.vector_db import QdrantVectorStore, get_vector_store
from src.llm.antigravity_llm import AntigravityLLM

logger = logging.getLogger(__name__)

QA_SYSTEM_PROMPT = """You are an expert Personal Knowledge Management (PKM) AI assistant for an Obsidian vault.
Answer the user's question clearly, structured, and concisely using the provided vault context.
CRITICAL FORMATTING RULES:
1. Wrap technical terms, cloud services, and concepts in Obsidian double-bracket WikiLinks with aliases where appropriate (e.g. [[Amazon Web Services|AWS]], [[Amazon S3|S3]], [[Amazon CloudFront|CloudFront]], [[Application Load Balancer|ALB]], [[Amazon Elastic Container Service|ECS]], [[Amazon VPC|VPC]], [[VPC Endpoint|VPC Endpoints]], [[Classless Inter-Domain Routing|CIDR]]).
2. NEVER use single brackets for WikiLinks (NEVER [AWS] or [Amazon S3|S3]).
3. If relevant sources exist, list them at the end of the answer in the format:
Sources: [[Source Note 1]], [[Source Note 2]]"""


class KnowledgeBaseQAAgent:
    """Agent that handles natural language questions using hybrid vector search and LLM synthesis."""

    def __init__(
        self,
        embedder: TextEmbedder | None = None,
        vector_store: QdrantVectorStore | None = None,
        llm: AntigravityLLM | None = None,
    ) -> None:
        """Initialize KnowledgeBaseQAAgent.

        Args:
            embedder: TextEmbedder instance. Defaults to default TextEmbedder.
            vector_store: QdrantVectorStore instance. Defaults to singleton vector store.
            llm: AntigravityLLM instance. Defaults to default AntigravityLLM.
        """
        self.embedder = embedder or TextEmbedder()
        self.vector_store = vector_store or get_vector_store()
        self.llm = llm or AntigravityLLM()

    async def query(
        self,
        query_text: str,
        top_k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> QueryResponse:
        """Process search query, perform hybrid search, and synthesize LLM answer with WikiLinks.

        Args:
            query_text: Natural language query string.
            top_k: Maximum number of relevant chunks to retrieve.
            filters: Optional filters for vector search.

        Returns:
            QueryResponse containing synthesized answer, cited sources, and context chunks.
        """
        clean_query = re.sub(r"^/(?:ask|query)\s*", "", query_text, flags=re.IGNORECASE).strip()
        if not clean_query:
            return QueryResponse(
                query=query_text,
                answer="Please provide a query after `/ask` (e.g., `/ask How do I set up a web application on aws?`).",
                sources=[],
                context_chunks=[],
            )

        logger.info("Executing knowledge base query: '%s'", clean_query)

        try:
            # 1. Generate query vector
            query_vec = await self.embedder.encode_async(clean_query)

            # 2. Perform hybrid search in Qdrant
            search_results = await self.vector_store.hybrid_search_async(
                query_text=clean_query,
                query_vector=query_vec,
                top_k=top_k,
            )

            context_blocks: list[str] = []
            context_chunks: list[dict[str, str]] = []
            sources: list[str] = []

            for idx, res in enumerate(search_results, start=1):
                payload = res.get("payload", {})
                content = payload.get("content", "")
                metadata = payload.get("metadata", {})
                title = (
                    metadata.get("atomic_note")
                    or metadata.get("daily_note")
                    or payload.get("atomic_note")
                    or payload.get("daily_note")
                    or f"Note #{idx}"
                )
                clean_title = str(title).replace(".md", "").strip()
                wikilink_title = f"[[{clean_title}]]"
                if wikilink_title not in sources and clean_title:
                    sources.append(wikilink_title)

                context_blocks.append(f"Source [{idx}] ({clean_title}):\n{content}")
                context_chunks.append({
                    "source": clean_title,
                    "content": content,
                    "score": str(res.get("score", 0.0)),
                })

            context_str = "\n\n".join(context_blocks) if context_blocks else "No relevant vault notes found."

            user_prompt = (
                f"User Question: {clean_query}\n\n"
                f"Retrieved Vault Context:\n{context_str}\n"
            )

            raw_answer = await self.llm.generate_text(
                prompt=user_prompt,
                system_prompt=QA_SYSTEM_PROMPT,
            )
            normalized_answer = normalize_obsidian_markdown(raw_answer.strip())

            return QueryResponse(
                query=clean_query,
                answer=normalized_answer,
                sources=sources,
                context_chunks=context_chunks,
            )

        except Exception as err:
            logger.exception("Failed to query knowledge base for '%s': %s", clean_query, err)
            return QueryResponse(
                query=clean_query,
                answer="⚠️ Sorry, an error occurred while processing your query.",
                sources=[],
                context_chunks=[],
            )


async def query_knowledge_base(
    query_text: str,
    top_k: int = 5,
    filters: dict[str, str] | None = None,
    embedder: TextEmbedder | None = None,
    vector_store: QdrantVectorStore | None = None,
    llm: AntigravityLLM | None = None,
) -> QueryResponse:
    """Convenience function to query knowledge base using KnowledgeBaseQAAgent."""
    agent = KnowledgeBaseQAAgent(embedder=embedder, vector_store=vector_store, llm=llm)
    return await agent.query(query_text=query_text, top_k=top_k, filters=filters)


__all__ = [
    "KnowledgeBaseQAAgent",
    "query_knowledge_base",
]
