"""Tree-sitter + PageRank repo map — Session 7 (v3.2).

Ported from Aider's ``aider/repomap.py`` (Apache 2.0,
github.com/Aider-AI/aider).  The approach:

1. Parse every .py in the project with tree-sitter.
2. Extract definitions (class / def / top-level assignment) and
   references (call / import / base-class).
3. Build a directed graph where edge ``file_A → file_B`` means file_A
   references a symbol defined in file_B.  Edge weight is the number
   of references.
4. Run personalised PageRank.  The personalisation vector lets us
   bias the ranking toward files the calling agent is actively
   editing (``chat_fnames``) or toward identifiers the agent has
   just mentioned (``mentioned_idents``).
5. Emit the top-K symbols as a token-budgeted text block, formatted
   as a grep-style file-then-symbols tree.

Why not just feed directory listings or grep?
    A directory dump is O(project size) with no signal; with 240
    files and 66 K LOC, that's multiple KB per prompt, with no hint
    about which files matter.  The repo-map is O(K) with *curated*
    signal — PageRank prioritises exactly the functions downstream
    code actually uses.  Same token cost, dramatically more signal.

Scope for session 7:
    Python (.py) only.  Aider also handles .go, .rs, .ts, etc. —
    follow-up work once Python is stable.  Non-Python files are
    skipped silently, not errored.

Attribution:
    Algorithm and personalisation-vector trick ported from Aider
    (github.com/Aider-AI/aider/blob/main/aider/repomap.py, Apache 2.0
    license).  This module is an independent re-implementation in
    Belief's style, not a vendored copy; Aider's code is the
    reference spec.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("belief.repomap")


_DEFAULT_CACHE_DIR = Path.home() / ".belief-engine" / "treesitter_cache"

# Rough token ratio for cl100k_base (tiktoken): 1 token ≈ 4 chars of
# English / code.  Used as a fallback when tiktoken isn't installed.
# Real Qwen tokenisation differs but is close enough for budgeting —
# we err toward smaller budget rather than larger.
_CHARS_PER_TOKEN_APPROX = 4


def _canonical_fname(path: Path, root: Path) -> str:
    """Relative-to-root fname, resilient to macOS /private/var vs /var
    symlink expansion.  Falls back to the absolute resolved path.
    """
    rp = os.path.realpath(str(path))
    rr = os.path.realpath(str(root))
    if rp.startswith(rr + os.sep):
        return rp[len(rr) + 1 :]
    if rp == rr:
        return ""
    return rp


# ---------------------------------------------------------------------------
# Parsed symbol record
# ---------------------------------------------------------------------------


@dataclass
class Tag:
    """One parsed symbol — a definition or a reference.

    ``kind`` is either ``"def"`` (definition) or ``"ref"`` (reference).
    For defs, ``line`` is the line of the ``def``/``class`` keyword; for
    refs, it's where the identifier appears.
    """

    fname: str
    name: str
    kind: str
    line: int
    signature: str = ""  # for defs: the line of source containing `def X(…):`


# ---------------------------------------------------------------------------
# RepoMap
# ---------------------------------------------------------------------------


class RepoMap:
    """Extract symbols, rank by PageRank, emit token-budgeted context."""

    def __init__(
        self,
        root: Path | str,
        *,
        cache_dir: Path | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._parser_cache: Any = None

    # ------------------------------------------------------------------
    # Parser setup
    # ------------------------------------------------------------------

    def _get_parser(self) -> Any:
        if self._parser_cache is not None:
            return self._parser_cache
        from tree_sitter import Language, Parser
        import tree_sitter_python

        lang = Language(tree_sitter_python.language())
        self._parser_cache = Parser(lang)
        return self._parser_cache

    # ------------------------------------------------------------------
    # Per-file tag extraction with disk cache
    # ------------------------------------------------------------------

    def tags_for_file(self, path: Path) -> list[Tag]:
        """Return tags for ``path`` — cached by mtime on disk."""
        try:
            st = path.stat()
        except FileNotFoundError:
            return []
        key = hashlib.sha1(str(path).encode()).hexdigest()
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                cached = pickle.loads(cache_file.read_bytes())
                if cached.get("mtime") == st.st_mtime_ns:
                    return [Tag(**d) for d in cached["tags"]]
            except Exception:
                pass  # fall through to re-parse

        try:
            source = path.read_bytes()
        except Exception:
            return []
        tags = list(self._parse_python(path, source))
        try:
            cache_file.write_bytes(
                pickle.dumps(
                    {
                        "mtime": st.st_mtime_ns,
                        "tags": [t.__dict__ for t in tags],
                    }
                )
            )
        except Exception as e:
            logger.debug("repomap cache write failed for %s: %s", path, e)
        return tags

    def _parse_python(self, path: Path, source: bytes) -> Iterable[Tag]:
        try:
            parser = self._get_parser()
            tree = parser.parse(source)
        except Exception as e:
            logger.debug("tree-sitter parse failed for %s: %s", path, e)
            return

        fname = _canonical_fname(path, self.root)

        def _line_text(byte_start: int) -> str:
            # Find the line boundaries around byte_start.
            start = source.rfind(b"\n", 0, byte_start) + 1
            end = source.find(b"\n", byte_start)
            if end == -1:
                end = len(source)
            return source[start:end].decode("utf-8", errors="replace").strip()

        # Walk AST.  For efficiency we iterate with a manual stack rather
        # than recursion (some Python projects have deep AST).
        stack = [tree.root_node]
        while stack:
            node = stack.pop()

            # DEFINITION: function_definition, class_definition
            if node.type in ("function_definition", "class_definition"):
                name_node = node.child_by_field_name("name")
                if name_node is not None:
                    yield Tag(
                        fname=fname,
                        name=name_node.text.decode("utf-8", errors="replace"),
                        kind="def",
                        line=node.start_point[0] + 1,
                        signature=_line_text(node.start_byte)[:200],
                    )

            # DEFINITION: module-level simple assignment (uppercase-named)
            if (
                node.type == "assignment"
                and node.parent is not None
                and node.parent.type == "module"
            ):
                target = node.children[0] if node.children else None
                if target is not None and target.type == "identifier":
                    name = target.text.decode("utf-8", errors="replace")
                    if name and name[0].isupper() or name.isupper():
                        yield Tag(
                            fname=fname,
                            name=name,
                            kind="def",
                            line=node.start_point[0] + 1,
                            signature=_line_text(node.start_byte)[:200],
                        )

            # REFERENCE: call — yield the callee's name if identifier.
            if node.type == "call":
                func_node = node.child_by_field_name("function")
                if func_node is not None:
                    name = None
                    if func_node.type == "identifier":
                        name = func_node.text.decode("utf-8", errors="replace")
                    elif func_node.type == "attribute":
                        attr = func_node.child_by_field_name("attribute")
                        if attr is not None:
                            name = attr.text.decode("utf-8", errors="replace")
                    if name:
                        yield Tag(
                            fname=fname,
                            name=name,
                            kind="ref",
                            line=node.start_point[0] + 1,
                        )

            # REFERENCE: import / import_from — the imported names are refs
            if node.type == "import_from_statement":
                # from X import A, B → refs to A, B (module X handled below)
                for child in node.children:
                    if child.type == "dotted_name" and child.start_byte > node.start_byte + 4:
                        # heuristic: skip the first dotted_name (the module)
                        continue
                    if child.type == "aliased_import" or child.type == "identifier":
                        name = child.text.decode("utf-8", errors="replace").split(" as ")[0]
                        yield Tag(
                            fname=fname,
                            name=name,
                            kind="ref",
                            line=node.start_point[0] + 1,
                        )

            # REFERENCE: base class list
            if node.type == "class_definition":
                supers = node.child_by_field_name("superclasses")
                if supers is not None:
                    for child in supers.children:
                        if child.type == "identifier":
                            yield Tag(
                                fname=fname,
                                name=child.text.decode("utf-8", errors="replace"),
                                kind="ref",
                                line=node.start_point[0] + 1,
                            )

            # Descend into children (right-to-left to preserve natural order).
            for child in reversed(node.children):
                stack.append(child)

    # ------------------------------------------------------------------
    # Graph + PageRank
    # ------------------------------------------------------------------

    def build_graph(self, files: Iterable[Path] | None = None) -> tuple[Any, dict[str, list[Tag]]]:
        """Build the reference graph for the project.

        Returns ``(graph, defs_by_name)`` where ``defs_by_name`` maps a
        symbol name to the list of tags that define it (used to
        resolve refs → def edges).
        """
        import networkx as nx

        if files is None:
            files = list(self._iter_python_files())
        else:
            files = [Path(f) for f in files]

        all_tags: list[Tag] = []
        for p in files:
            all_tags.extend(self.tags_for_file(p))

        defs_by_name: dict[str, list[Tag]] = {}
        for t in all_tags:
            if t.kind == "def":
                defs_by_name.setdefault(t.name, []).append(t)

        G: Any = nx.DiGraph()
        # Every file is a node.
        for p in files:
            fname = _canonical_fname(p, self.root)
            G.add_node(fname)
        # Every ref adds a weighted edge from the ref's file to each
        # file that defines a symbol of that name.
        for t in all_tags:
            if t.kind != "ref":
                continue
            defs = defs_by_name.get(t.name, [])
            for d in defs:
                if d.fname == t.fname:
                    continue  # ignore self-references
                if G.has_edge(t.fname, d.fname):
                    G[t.fname][d.fname]["weight"] += 1.0
                else:
                    G.add_edge(t.fname, d.fname, weight=1.0)
        return G, defs_by_name

    def rank(
        self,
        G: Any,
        *,
        chat_fnames: list[str] | None = None,
        mentioned_fnames: list[str] | None = None,
        mentioned_idents: list[str] | None = None,
        defs_by_name: dict[str, list[Tag]] | None = None,
    ) -> dict[str, float]:
        """Return a dict ``fname -> PageRank score`` with personalisation.

        Weights:
          - chat_fnames        → 1.0 (agent is actively editing these)
          - mentioned_fnames   → 0.5 (referenced in the current prompt)
          - mentioned_idents   → 0.3 (per file that defines any of them)
          - default            → 0.01
        """
        import networkx as nx

        personalization: dict[str, float] = {}
        default = 0.01
        chat = set(chat_fnames or [])
        mentioned = set(mentioned_fnames or [])
        idents = set(mentioned_idents or [])

        for n in G.nodes:
            w = default
            if n in chat:
                w = max(w, 1.0)
            if n in mentioned:
                w = max(w, 0.5)
            personalization[n] = w

        if idents and defs_by_name:
            for ident in idents:
                for d in defs_by_name.get(ident, []):
                    if d.fname in personalization:
                        personalization[d.fname] = max(personalization[d.fname], 0.3)

        try:
            return nx.pagerank(
                G,
                alpha=0.85,
                personalization=personalization,
                weight="weight",
            )
        except (nx.PowerIterationFailedConvergence, nx.NetworkXError) as e:
            logger.debug("PageRank failed to converge: %s; falling back to uniform", e)
            return {n: 1.0 / max(len(G.nodes), 1) for n in G.nodes}

    # ------------------------------------------------------------------
    # Token-budgeted tag map
    # ------------------------------------------------------------------

    def get_ranked_tags_map(
        self,
        *,
        chat_fnames: list[str] | None = None,
        mentioned_fnames: list[str] | None = None,
        mentioned_idents: list[str] | None = None,
        max_tokens: int = 2000,
        files: Iterable[Path] | None = None,
    ) -> str:
        """Return a PageRank-ranked, token-budgeted tag map.

        Output format (grep-like)::

            belief/agents/planner.py:
              class PlannerAgent(BaseAgent):
                async def run(self, state): ...
            belief/llm.py:
              class AsyncOllamaClient: ...
        """
        G, defs_by_name = self.build_graph(files=files)
        ranks = self.rank(
            G,
            chat_fnames=chat_fnames,
            mentioned_fnames=mentioned_fnames,
            mentioned_idents=mentioned_idents,
            defs_by_name=defs_by_name,
        )
        # Order files by descending rank.
        ordered_files = sorted(ranks.items(), key=lambda kv: kv[1], reverse=True)

        char_budget = max_tokens * _CHARS_PER_TOKEN_APPROX
        lines: list[str] = []
        used = 0
        for fname, _score in ordered_files:
            defs = [t for t in defs_by_name.values() for t in t if t.fname == fname]
            if not defs:
                continue
            chunk_lines = [f"{fname}:"]
            for d in defs:
                chunk_lines.append(f"  {d.signature}")
            chunk = "\n".join(chunk_lines) + "\n"
            if used + len(chunk) > char_budget and used > 0:
                break
            lines.append(chunk)
            used += len(chunk)
        return "".join(lines)

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def _iter_python_files(self) -> Iterable[Path]:
        """Yield every .py file under ``self.root`` (skipping common
        non-source dirs)."""
        skip = {
            ".git",
            ".venv",
            "venv",
            "node_modules",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            "build",
            "dist",
            "output",
        }
        for p in self.root.rglob("*.py"):
            if any(part in skip for part in p.parts):
                continue
            yield p


__all__ = ["RepoMap", "Tag"]
