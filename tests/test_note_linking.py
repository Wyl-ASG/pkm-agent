"""Tests for adding links to existing notes without changing the user's words."""

import pytest

from src.agents.parser import EntryParserAgent
from src.graphrag.graph import VaultKnowledgeGraph
from src.graphrag.resolver import WikiLinkResolver
from src.llm.mock_llm import MockLLM
from src.vault.md_writer import ObsidianVaultWriter


def _link(text: str, notes: list[str], aliases: dict[str, str] | None = None) -> str:
    return WikiLinkResolver(existing_notes=notes, alias_map=aliases).link_known_notes(text)


# Matching -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "notes", "expected"),
    [
        ("Qdrant supports indexes", ["Qdrant"], "[[Qdrant]] supports indexes"),
        ("qdrant supports indexes", ["Qdrant"], "[[Qdrant|qdrant]] supports indexes"),
        ("Use Docker.", ["Docker"], "Use [[Docker]]."),
        ("many cgroups here", ["Cgroup"], "many [[Cgroup|cgroups]] here"),
        ("one distributed system", ["Distributed Systems"], "one [[Distributed Systems|distributed system]]"),
        ("AWS and AWS again", ["AWS"], "[[AWS]] and AWS again"),
        ("Machine Learning basics", ["Machine", "Machine Learning"], "[[Machine Learning]] basics"),
    ],
)
def test_existing_titles_are_linked(text, notes, expected):
    assert _link(text, notes) == expected


def test_aliases_link_to_their_note():
    assert _link("Runs on k8s now", ["Kubernetes"], {"k8s": "Kubernetes"}) == "Runs on [[Kubernetes|k8s]] now"


def test_alias_for_a_deleted_note_is_ignored():
    assert _link("Runs on k8s now", ["Docker"], {"k8s": "Kubernetes"}) == "Runs on k8s now"


@pytest.mark.parametrize(
    ("text", "notes"),
    [
        ("A dockerfile is not the same", ["Docker"]),
        ("Postgres is fast", ["Postgres Backup Strategy"]),
        ("Nothing matches at all", []),
        ("This is what we do", ["what"]),
        ("An AI model", ["AI"]),
    ],
)
def test_partial_words_similar_titles_and_trivial_titles_are_not_linked(text, notes):
    assert _link(text, notes) == text


@pytest.mark.parametrize(
    "text",
    [
        "See [[Docker]] first",
        "See [[Containers|Docker]] first",
        "[status:: Docker]",
        "Run `Docker` locally",
        "https://example.com/Docker/page",
        "tagged #Docker today",
        "> [!Docker] heading",
    ],
)
def test_existing_links_and_syntax_are_left_alone(text):
    assert _link(text, ["Docker"]) == text


# Parser ---------------------------------------------------------------------


async def _parse(raw_text: str, notes: list[str], aliases: dict[str, str] | None = None, **llm_fields):
    response = {"timestamp": "2026-08-30 20:00", "category": "discovery", "content": raw_text, **llm_fields}
    parser = EntryParserAgent(llm=MockLLM(default_json_response=response))
    return await parser.parse(raw_text, existing_notes=notes, note_aliases=aliases)


@pytest.mark.asyncio
async def test_parser_links_user_text_to_existing_notes():
    entry = await _parse("It was replaced by cgroups and seccomp", ["Cgroups", "Seccomp"])
    assert entry.content == "It was replaced by [[Cgroups|cgroups]] and [[Seccomp|seccomp]]"
    assert entry.extracted_wikilinks == ["Cgroups", "Seccomp"]


@pytest.mark.asyncio
async def test_parser_uses_aliases_from_the_vault():
    entry = await _parse("Runs on k8s", ["Kubernetes"], {"k8s": "Kubernetes"})
    assert entry.content == "Runs on [[Kubernetes|k8s]]"


@pytest.mark.asyncio
async def test_task_text_is_linked_without_touching_task_syntax():
    entry = await _parse("Migrate Postgres by 2026-09-01", ["Postgres"], category="task")
    assert entry.content == "- [ ] Migrate [[Postgres]] by 2026-09-01 ➕ 2026-08-30 📅 2026-09-01"


@pytest.mark.parametrize(
    "llm_content",
    [
        "It was replaced by [[Invented Note]]",
        "Summary of what the user said",
        "[[It was replaced by cgroups]]",
    ],
)
@pytest.mark.asyncio
async def test_model_markup_and_rewrites_never_reach_the_note(llm_content):
    entry = await _parse("It was replaced by cgroups", [], content=llm_content, extracted_wikilinks=["Invented Note"])
    assert entry.content == "It was replaced by cgroups"
    assert entry.extracted_wikilinks == []


@pytest.mark.asyncio
async def test_links_the_user_typed_are_kept():
    entry = await _parse("See [[Future Topic]] later", [])
    assert entry.content == "See [[Future Topic]] later"
    assert entry.extracted_wikilinks == ["Future Topic"]


@pytest.mark.asyncio
async def test_note_list_is_not_sent_to_the_model():
    llm = MockLLM(default_json_response={"timestamp": "2026-08-30 20:00", "content": "x"})
    await EntryParserAgent(llm=llm).parse("Plain thought", existing_notes=["Secret Project"])
    prompt, _system_prompt = llm.recorded_prompts[0]
    assert "Secret Project" not in prompt


@pytest.mark.asyncio
async def test_removing_all_notes_clears_old_link_targets():
    llm = MockLLM(default_json_response={"timestamp": "2026-08-30 20:00", "content": "x"})
    parser = EntryParserAgent(llm=llm)
    await parser.parse("Docker", existing_notes=["Docker"])
    entry = await parser.parse("Docker", existing_notes=[])
    assert entry.content == "Docker"


@pytest.mark.parametrize(
    ("raw_text", "category"),
    [
        ("use arr[i] in the loop", "discovery"),
        ("see note: foo:: bar here", "log"),
        ("fix arr[i] in the loop", "task"),
    ],
)
@pytest.mark.asyncio
async def test_user_text_with_brackets_reaches_the_daily_note_unchanged(tmp_path, raw_text, category):
    entry = await _parse(raw_text, [], category=category)
    assert raw_text in entry.content

    daily_path, _ = ObsidianVaultWriter(vault_path=tmp_path).append_interstitial_entry(entry, date_str="2026-08-30")
    assert raw_text in daily_path.read_text(encoding="utf-8")


# Vault integration ----------------------------------------------------------


def test_note_titles_exclude_daily_notes_and_dashboard(tmp_path):
    writer = ObsidianVaultWriter(vault_path=tmp_path)
    writer._ensure_directories()
    (writer.notes_dir / "Docker.md").write_text("x", encoding="utf-8")
    (writer.daily_notes_dir / "2026-08-30.md").write_text("x", encoding="utf-8")
    (tmp_path / "Dashboard.md").write_text("x", encoding="utf-8")

    assert writer.get_existing_note_titles() == ["Docker"]


@pytest.mark.asyncio
async def test_linked_capture_creates_a_graph_edge(tmp_path):
    writer = ObsidianVaultWriter(vault_path=tmp_path)
    writer._ensure_directories()
    (writer.notes_dir / "Chroot.md").write_text("# Chroot\n\nChanges the root.\n", encoding="utf-8")

    entry = await _parse("chroot does not isolate PIDs", writer.get_existing_note_titles())
    writer.append_interstitial_entry(entry, date_str="2026-08-30")

    graph = VaultKnowledgeGraph(tmp_path)
    graph.build_graph()
    assert graph.graph.has_edge("2026-08-30", "Chroot")
