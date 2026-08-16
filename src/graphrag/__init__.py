"""GraphRAG module exports for PKM AI Agent."""

from src.graphrag.chunker import MarkdownChunk, StructureAwareChunker, chunk_markdown_file_structured
from src.graphrag.embedder import TextEmbedder, resolve_torch_device
from src.graphrag.graph import NoteNode, VaultKnowledgeGraph
from src.graphrag.reindexer import VaultReindexer, reindex_vault
from src.graphrag.reranker import CrossEncoderReranker
from src.graphrag.resolver import WikiLinkResolver
from src.graphrag.retriever import HybridRetriever, extract_temporal_filters
from src.graphrag.sparse import BM25Index, tokenize_for_bm25
from src.graphrag.vector_db import QdrantVectorStore, get_vector_store
from src.graphrag.versioning import FileIndexRecord, IndexVersionRegistry, compute_file_hash
from src.graphrag.watcher import VaultWatcher

__all__ = [
    "MarkdownChunk",
    "StructureAwareChunker",
    "chunk_markdown_file_structured",
    "TextEmbedder",
    "resolve_torch_device",
    "NoteNode",
    "VaultKnowledgeGraph",
    "CrossEncoderReranker",
    "WikiLinkResolver",
    "HybridRetriever",
    "extract_temporal_filters",
    "BM25Index",
    "tokenize_for_bm25",
    "QdrantVectorStore",
    "get_vector_store",
    "FileIndexRecord",
    "IndexVersionRegistry",
    "compute_file_hash",
    "VaultReindexer",
    "reindex_vault",
    "VaultWatcher",
]
