"""In-memory Knowledge Graph index extracted from Obsidian Markdown vault notes and WikiLinks."""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
from typing import Any
import networkx as nx

from src.config import settings
from src.graphrag.chunker import extract_frontmatter, extract_tags_from_text, extract_wikilinks

logger = logging.getLogger(__name__)


@dataclass
class NoteNode:
    """Metadata node in knowledge graph representing an Obsidian note."""

    title: str
    rel_path: str
    file_name: str
    aliases: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    headings: list[str] = field(default_factory=list)
    block_ids: list[str] = field(default_factory=list)
    outgoing_links: list[str] = field(default_factory=list)
    backlinks: list[str] = field(default_factory=list)
    created: str | None = None
    modified: str | None = None
    memory_type: str = "note"


class VaultKnowledgeGraph:
    """In-memory graph index of Obsidian vault notes, WikiLinks, and backlinks with incremental update support."""

    def __init__(self, vault_path: str | Path | None = None) -> None:
        """Initialize VaultKnowledgeGraph.

        Args:
            vault_path: Path to vault directory. Defaults to settings.VAULT_PATH.
        """
        self.vault_path = Path(vault_path or settings.VAULT_PATH).resolve()
        self.graph: nx.DiGraph = nx.DiGraph()
        self.nodes: dict[str, NoteNode] = {}
        self.alias_to_title: dict[str, str] = {}
        self.path_to_title: dict[str, str] = {}

    def _parse_note_node(self, file_path: Path) -> tuple[NoteNode, list[str]] | None:
        """Parse note metadata from file path."""
        try:
            rel_path = str(file_path.relative_to(self.vault_path))
            title = file_path.stem
            raw_text = file_path.read_text(encoding="utf-8")
            frontmatter, body = extract_frontmatter(raw_text)

            aliases: list[str] = []
            fm_aliases = frontmatter.get("aliases", [])
            if isinstance(fm_aliases, list):
                aliases.extend([str(a).strip() for a in fm_aliases if str(a).strip()])
            elif isinstance(fm_aliases, str):
                aliases.extend([a.strip() for a in fm_aliases.split(",") if a.strip()])

            tags = extract_tags_from_text(raw_text)
            fm_tags = frontmatter.get("tags", [])
            if isinstance(fm_tags, list):
                tags.extend([f"#{str(t).lstrip('#').lower()}" for t in fm_tags])
            elif isinstance(fm_tags, str):
                tags.extend([f"#{t.strip().lstrip('#').lower()}" for t in fm_tags.split(",") if t.strip()])
            tags = sorted(set(tags))

            headings = [
                m.group(1).strip()
                for m in re.finditer(r"^#{1,6}\s+(.*)$", body, re.MULTILINE)
            ]

            block_ids = [
                f"^{m.group(1)}"
                for m in re.finditer(r"\^([a-zA-Z0-9_-]+)\s*$", body, re.MULTILINE)
            ]

            outgoing = extract_wikilinks(body)

            node = NoteNode(
                title=title,
                rel_path=rel_path,
                file_name=file_path.name,
                aliases=aliases,
                tags=tags,
                headings=headings,
                block_ids=block_ids,
                outgoing_links=outgoing,
                backlinks=[],
                created=str(frontmatter.get("created", "")),
                memory_type=str(frontmatter.get("type", "note")),
            )
            return node, outgoing
        except Exception as err:
            logger.debug("Failed parsing note %s for graph: %s", file_path, err)
            return None

    def build_graph(self) -> None:
        """Scan vault Markdown files and construct graph nodes, WikiLink edges, and backlinks."""
        self.graph.clear()
        self.nodes.clear()
        self.alias_to_title.clear()
        self.path_to_title.clear()

        if not self.vault_path.exists():
            return

        ignored_parts = {".git", ".obsidian", ".trash", ".agent", ".venv", "__pycache__"}
        md_files: list[Path] = []
        for p in self.vault_path.rglob("*.md"):
            try:
                rel = p.relative_to(self.vault_path)
                if any(part in ignored_parts or part.startswith(".") for part in rel.parts):
                    continue
                if p.is_file():
                    md_files.append(p)
            except ValueError:
                continue

        raw_links_map: dict[str, list[str]] = {}

        # 1. First pass: extract note metadata
        for file_path in md_files:
            parsed = self._parse_note_node(file_path)
            if not parsed:
                continue
            node, outgoing = parsed
            title = node.title
            self.nodes[title] = node
            self.path_to_title[node.rel_path] = title
            self.alias_to_title[title.lower()] = title
            for alias in node.aliases:
                self.alias_to_title[alias.lower()] = title
            raw_links_map[title] = outgoing
            self.graph.add_node(title, **node.__dict__)

        # 2. Second pass: resolve edges and populate backlinks
        for src_title, out_links in raw_links_map.items():
            for target_name in out_links:
                canonical_target = self.resolve_canonical_title(target_name)
                if canonical_target:
                    self.graph.add_edge(src_title, canonical_target)
                    if canonical_target in self.nodes:
                        if src_title not in self.nodes[canonical_target].backlinks:
                            self.nodes[canonical_target].backlinks.append(src_title)

        logger.info(
            "Built Knowledge Graph: %d notes, %d WikiLink connections.",
            len(self.nodes),
            self.graph.number_of_edges(),
        )

    def update_file_note(self, file_path: Path) -> None:
        """Incrementally update or add a single note file in the graph."""
        parsed = self._parse_note_node(file_path)
        if not parsed:
            return
        node, outgoing = parsed
        title = node.title

        # Remove old edges from this node
        if title in self.graph:
            self.graph.remove_node(title)

        self.nodes[title] = node
        self.path_to_title[node.rel_path] = title
        self.alias_to_title[title.lower()] = title
        for alias in node.aliases:
            self.alias_to_title[alias.lower()] = title

        self.graph.add_node(title, **node.__dict__)

        # Reconnect outgoing links
        for target_name in outgoing:
            canonical_target = self.resolve_canonical_title(target_name)
            if canonical_target:
                self.graph.add_edge(title, canonical_target)
                if canonical_target in self.nodes:
                    if title not in self.nodes[canonical_target].backlinks:
                        self.nodes[canonical_target].backlinks.append(title)

    def remove_file_note(self, file_path: Path) -> None:
        """Remove a note file from the graph."""
        rel_path = str(file_path.relative_to(self.vault_path)) if file_path.is_absolute() else str(file_path)
        title = self.path_to_title.get(rel_path, file_path.stem)
        if title in self.graph:
            self.graph.remove_node(title)
        if title in self.nodes:
            del self.nodes[title]
        self.path_to_title.pop(rel_path, None)

    def resolve_canonical_title(self, raw_name: str) -> str | None:
        """Resolve note title or alias to canonical note title in vault."""
        clean = raw_name.replace("[[", "").replace("]]", "").strip()
        if clean in self.nodes:
            return clean
        clean_lower = clean.lower()
        if clean_lower in self.alias_to_title:
            return self.alias_to_title[clean_lower]
        return None

    def get_neighbors(
        self,
        note_title: str,
        max_hops: int = 1,
        max_neighbors: int = 5,
    ) -> list[str]:
        """Get 1-hop or n-hop neighboring note titles (both outgoing links and backlinks).

        Args:
            note_title: Target note title.
            max_hops: Maximum traversal depth. Defaults to 1.
            max_neighbors: Maximum neighbor notes to return. Defaults to 5.

        Returns:
            List of neighboring note titles.
        """
        canonical = self.resolve_canonical_title(note_title)
        if not canonical or canonical not in self.graph:
            return []

        neighbors: set[str] = set()

        for succ in self.graph.successors(canonical):
            if succ != canonical:
                neighbors.add(succ)

        for pred in self.graph.predecessors(canonical):
            if pred != canonical:
                neighbors.add(pred)

        if max_hops > 1:
            second_hop = set()
            for n in list(neighbors):
                for succ in self.graph.successors(n):
                    if succ != canonical:
                        second_hop.add(succ)
                for pred in self.graph.predecessors(n):
                    if pred != canonical:
                        second_hop.add(pred)
            neighbors.update(second_hop)

        return sorted(neighbors)[:max_neighbors]

    def expand_candidates(
        self,
        seed_note_titles: list[str],
        max_hops: int = 1,
        max_neighbors: int = 5,
    ) -> list[str]:
        """Expand a set of candidate note titles with their graph neighbors."""
        expanded = set(seed_note_titles)
        for title in seed_note_titles:
            nb = self.get_neighbors(title, max_hops=max_hops, max_neighbors=max_neighbors)
            expanded.update(nb)
        return sorted(expanded)


__all__ = ["VaultKnowledgeGraph", "NoteNode"]
