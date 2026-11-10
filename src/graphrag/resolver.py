"""Safe WikiLink Entity Resolver to prevent hallucinated links and duplicate notes."""

from difflib import SequenceMatcher
import logging
import re
from typing import Sequence
from src.config import settings
from src.graphrag.sparse import STOPWORDS

logger = logging.getLogger(__name__)

MIN_LINK_TERM_LENGTH = 3

# Markdown that must never gain a link: links, Dataview fields, callouts, code, URLs, dates, tags, block IDs.
_PROTECTED_MARKDOWN = re.compile(
    r"(\[\[[^\]\n]*\]\]"
    r"|\[[^\[\]\n]*::[^\[\]\n]*\]"
    r"|\[![A-Za-z]+\]"
    r"|`[^`\n]*`"
    r"|https?://\S+"
    r"|(?:➕|📅)\s*\d{4}-\d{2}-\d{2}"
    r"|(?<!\S)[#^][\w/-]+)"
)


def _mention_pattern(term: str) -> re.Pattern[str]:
    """Build a whole-word, case-insensitive pattern for a term and its simple plural or singular."""
    if len(term) > MIN_LINK_TERM_LENGTH and term.endswith("s") and not term.endswith("ss"):
        body = f"{re.escape(term[:-1])}s?"
    else:
        body = f"{re.escape(term)}(?:s|es)?"
    return re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)


def string_similarity(a: str, b: str) -> float:
    """Calculate SequenceMatcher string ratio between two lowercase strings."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def normalize_term_stem(term: str) -> str:
    """Basic stemming and normalization to match singular/plural and casing."""
    t = term.lower().strip()
    t = re.sub(r"\b(notes?|guide|docs?|concepts?)\b", "", t).strip()
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("es") and len(t) > 3:
        return t[:-2]
    if t.endswith("s") and len(t) > 3 and not t.endswith("ss"):
        return t[:-1]
    return t


class WikiLinkResolver:
    """Resolves proposed entity terms against existing vault notes and aliases with confidence scoring."""

    def __init__(
        self,
        existing_notes: Sequence[str] | None = None,
        alias_map: dict[str, str] | None = None,
        confidence_threshold: float | None = None,
    ) -> None:
        """Initialize WikiLinkResolver.

        Args:
            existing_notes: List of canonical note titles.
            alias_map: Dictionary mapping lowercase alias -> canonical note title.
            confidence_threshold: Minimum match confidence score (0.0 to 1.0).
        """
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else getattr(settings, "WIKILINKS_CONFIDENCE_THRESHOLD", 0.65)
        )
        self.existing_notes: list[str] = list(existing_notes or [])
        self.alias_map: dict[str, str] = dict(alias_map or {})
        self._note_title_set: set[str] = set(self.existing_notes)
        self._lower_to_canonical: dict[str, str] = {
            n.lower().strip(): n for n in self.existing_notes
        }

    def update_notes(self, notes: Sequence[str], alias_map: dict[str, str] | None = None) -> None:
        """Update existing note dictionary."""
        self.existing_notes = list(notes)
        self._note_title_set = set(self.existing_notes)
        self._lower_to_canonical = {n.lower().strip(): n for n in self.existing_notes}
        if alias_map is not None:
            self.alias_map = dict(alias_map)

    def score_candidate(self, proposed_term: str, candidate_title: str) -> float:
        """Score matching confidence between proposed term and a candidate note title."""
        p_clean = proposed_term.strip()
        c_clean = candidate_title.strip()

        # 1. Exact case-insensitive match
        if p_clean.lower() == c_clean.lower():
            return 1.0

        # 2. Known alias match
        if p_clean.lower() in self.alias_map and self.alias_map[p_clean.lower()] == candidate_title:
            return 0.95

        # 3. Stem / plural-singular match (e.g. Distributed System vs Distributed Systems)
        p_stem = normalize_term_stem(p_clean)
        c_stem = normalize_term_stem(c_clean)
        if p_stem and c_stem and p_stem == c_stem:
            return 0.90

        # 4. Word boundary / substring containment
        p_words = set(re.findall(r"\w+", p_clean.lower()))
        c_words = set(re.findall(r"\w+", c_clean.lower()))

        if p_words and c_words:
            if p_words == c_words:
                return 0.92
            if p_words.issubset(c_words) or c_words.issubset(p_words):
                intersection_len = len(p_words.intersection(c_words))
                union_len = len(p_words.union(c_words))
                return 0.70 + (0.20 * (intersection_len / union_len))

        # 5. String similarity ratio
        return string_similarity(p_clean, c_clean)

    def resolve(self, proposed_term: str) -> tuple[str | None, float]:
        """Resolve a proposed entity term to a canonical note title if confidence is high enough.

        Args:
            proposed_term: Candidate entity string (e.g., 'distributed systems').

        Returns:
            Tuple of (canonical_note_title_or_None, confidence_score).
        """
        clean_term = proposed_term.strip().replace("[[", "").replace("]]", "")
        if not clean_term:
            return None, 0.0

        # Direct exact match check
        if clean_term in self._note_title_set:
            return clean_term, 1.0

        clean_lower = clean_term.lower()
        if clean_lower in self._lower_to_canonical:
            return self._lower_to_canonical[clean_lower], 1.0

        if clean_lower in self.alias_map:
            return self.alias_map[clean_lower], 0.95

        # Search candidates
        best_candidate: str | None = None
        best_score = 0.0

        def _fast_filter(term: str) -> list[str]:
            term_words = set(re.findall(r"\w+", term.lower()))
            if not term_words:
                return self.existing_notes
            filtered = []
            for note in self.existing_notes:
                note_words = set(re.findall(r"\w+", note.lower()))
                if term_words.intersection(note_words):
                    filtered.append(note)
            return filtered if len(filtered) > 0 else self.existing_notes

        candidates = _fast_filter(clean_term)
        for note in candidates:
            score = self.score_candidate(clean_term, note)
            if score > best_score:
                best_score = score
                best_candidate = note

        if best_score >= self.confidence_threshold and best_candidate:
            return best_candidate, best_score

        return None, best_score

    def _linkable_terms(self) -> list[tuple[str, str]]:
        """Return (term, note title) pairs, longest first, skipping aliases of notes that no longer exist."""
        terms: dict[str, str] = {}
        for alias, title in self.alias_map.items():
            if title in self._note_title_set:
                terms[alias.lower().strip()] = title
        for title in self.existing_notes:
            terms[title.lower().strip()] = title
        usable = [
            (term, title)
            for term, title in terms.items()
            if len(term) >= MIN_LINK_TERM_LENGTH and term not in STOPWORDS
        ]
        return sorted(usable, key=lambda pair: len(pair[0]), reverse=True)

    def link_known_notes(self, text: str) -> str:
        """Link the first mention of each existing note, leaving every visible word unchanged.

        Only exact titles, known aliases, and simple plurals match. Fuzzy matching is
        deliberately not used here because a wrong link is worse than a missing one.
        """
        linked_text = text
        linked_titles: set[str] = set()
        for term, title in self._linkable_terms():
            if title in linked_titles:
                continue
            pattern = _mention_pattern(term)
            parts = _PROTECTED_MARKDOWN.split(linked_text)
            for index, part in enumerate(parts):
                # Odd positions hold protected markdown captured by the split.
                if index % 2 == 1:
                    continue
                match = pattern.search(part)
                if match:
                    mention = match.group(0)
                    link = f"[[{title}]]" if mention == title else f"[[{title}|{mention}]]"
                    parts[index] = f"{part[:match.start()]}{link}{part[match.end():]}"
                    linked_titles.add(title)
                    break
            linked_text = "".join(parts)
        return linked_text

    def format_wikilink_safe(self, proposed_term: str, display_text: str | None = None) -> str:
        """Format a WikiLink safely: if resolved to existing note, format as [[Canonical|Display]] or [[Canonical]].

        If not resolved, returns plain text or original proposed term based on confidence.
        """
        canonical, score = self.resolve(proposed_term)
        display = display_text or proposed_term

        if canonical:
            if canonical == display:
                return f"[[{canonical}]]"
            else:
                return f"[[{canonical}|{display}]]"

        # Low confidence -> do not wrap in WikiLink
        return display


__all__ = ["WikiLinkResolver", "string_similarity", "normalize_term_stem"]
