"""Sparse BM25 lexical retriever for hybrid PKM search with incremental updates."""

import logging
import re
from typing import Any
from rank_bm25 import BM25Plus

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "did", "do", "does", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor",
    "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "with", "would", "you", "your", "yours", "yourself", "yourselves",
}


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize text for BM25 search, preserving WikiLink terms, code identifiers, and words."""
    if not text:
        return []

    # Extract WikiLink concepts as combined tokens
    wikilink_tokens = []
    wikilinks = re.findall(r"\[\[([^\]\|]+)(?:\|[^\]]+)?\]\]", text)
    for wl in wikilinks:
        clean = wl.strip().lower()
        if clean:
            wikilink_tokens.append(clean)
            wikilink_tokens.extend(re.findall(r"\w+", clean))

    # General word tokens
    words = re.findall(r"[a-zA-Z0-9_\-\.\#]+", text.lower())
    filtered_tokens = [w for w in words if w not in STOPWORDS and len(w) > 1]

    return filtered_tokens + wikilink_tokens


class BM25Index:
    """In-memory BM25 index for fast lexical retrieval across vault chunks with cached tokens."""

    def __init__(self) -> None:
        """Initialize empty BM25Index."""
        self.doc_ids: list[str] = []
        self.documents: list[dict[str, Any]] = []
        self.tokenized_corpus: list[list[str]] = []
        self._doc_store: dict[str, dict[str, Any]] = {}
        self._file_to_doc_ids: dict[str, set[str]] = {}
        self._bm25: BM25Plus | None = None

    def _sync_index_structures(self) -> None:
        """Re-sync lists and instantiate BM25Plus from precomputed tokens in _doc_store."""
        docs = list(self._doc_store.values())
        self.doc_ids = [d["id"] for d in docs]
        self.documents = docs
        self.tokenized_corpus = [d["_tokens"] for d in docs]
        if self.tokenized_corpus:
            self._bm25 = BM25Plus(self.tokenized_corpus)
            logger.info("Synchronized BM25 lexical index with %d documents.", len(self.documents))
        else:
            self._bm25 = None

    def build_index(self, documents: list[dict[str, Any]]) -> None:
        """Build or rebuild BM25 index from a list of document chunk payloads.

        Args:
            documents: List of dicts containing 'id', 'content', and optional 'metadata'.
        """
        self._doc_store.clear()
        self._file_to_doc_ids.clear()

        for doc in documents:
            doc_id = str(doc.get("id", ""))
            content = doc.get("content", "")
            meta = doc.get("metadata", doc.get("payload", {}))
            tokens = tokenize_for_bm25(content)
            if not tokens:
                tokens = ["empty"]
            
            entry = {
                "id": doc_id,
                "content": content,
                "metadata": meta,
                "_tokens": tokens,
            }
            self._doc_store[doc_id] = entry
            file_path = meta.get("file_path")
            if file_path:
                self._file_to_doc_ids.setdefault(file_path, set()).add(doc_id)

        self._sync_index_structures()

    def remove_file_chunks(self, file_rel_path: str) -> None:
        """Remove all chunks associated with a specific file path without re-tokenizing remaining docs."""
        target_ids = self._file_to_doc_ids.pop(file_rel_path, set())
        if not target_ids:
            # Fallback scan in case metadata was not indexed under file_to_doc_ids
            target_ids = {
                doc_id for doc_id, doc in self._doc_store.items()
                if doc.get("metadata", {}).get("file_path") == file_rel_path
            }

        if target_ids:
            for doc_id in target_ids:
                self._doc_store.pop(doc_id, None)
            self._sync_index_structures()

    def upsert_file_chunks(self, file_rel_path: str, new_chunks: list[dict[str, Any]]) -> None:
        """Update or insert chunks for a specific file incrementally, tokenizing only new chunks."""
        # 1. Remove old chunks for this file
        target_ids = self._file_to_doc_ids.pop(file_rel_path, set())
        if not target_ids:
            target_ids = {
                doc_id for doc_id, doc in self._doc_store.items()
                if doc.get("metadata", {}).get("file_path") == file_rel_path
            }
        for doc_id in target_ids:
            self._doc_store.pop(doc_id, None)

        # 2. Tokenize and insert only the new chunks
        new_ids: set[str] = set()
        for doc in new_chunks:
            doc_id = str(doc.get("id", ""))
            content = doc.get("content", "")
            meta = doc.get("metadata", doc.get("payload", {}))
            tokens = tokenize_for_bm25(content)
            if not tokens:
                tokens = ["empty"]

            self._doc_store[doc_id] = {
                "id": doc_id,
                "content": content,
                "metadata": meta,
                "_tokens": tokens,
            }
            new_ids.add(doc_id)

        self._file_to_doc_ids[file_rel_path] = new_ids
        self._sync_index_structures()

    def search(self, query: str, top_k: int = 30) -> list[dict[str, Any]]:
        """Search BM25 index for query string.

        Args:
            query: Query text.
            top_k: Number of top results to return.

        Returns:
            List of result dictionaries with 'id', 'score', 'payload', 'content'.
        """
        if not self._bm25 or not self.documents:
            return []

        tokens = tokenize_for_bm25(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        scored_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results: list[dict[str, Any]] = []
        for idx in scored_indices[:top_k]:
            score = float(scores[idx])
            if score <= 0.0001:
                # Check if tokens actually overlap with document tokens
                doc_tokens = set(self.tokenized_corpus[idx])
                if not any(t in doc_tokens for t in tokens):
                    break
            doc = self.documents[idx]
            results.append({
                "id": self.doc_ids[idx],
                "score": max(score, 0.01),
                "sparse_score": max(score, 0.01),
                "content": doc.get("content", ""),
                "payload": doc.get("metadata", doc.get("payload", {})),
            })

        return results


__all__ = ["BM25Index", "tokenize_for_bm25"]
