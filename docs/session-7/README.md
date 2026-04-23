# Session 7 — tree-sitter + PageRank repo-map

Terse verification per the pacing rule.

## Files

New:
- `belief/repomap/__init__.py` — RepoMap class (tree-sitter + networkx PageRank)
- `tests/test_repomap.py` — 8 hermetic tests
- `docs/session-7/README.md` (this file)

Modified:
- `belief/cli.py` — `belief repomap [--root DIR] [--query IDENT] [--top N]`
- `pyproject.toml` — tree-sitter, tree-sitter-python, networkx, scipy added to core deps

## Commit

```bash
cd ~/Desktop/belief-engine
git checkout main
git pull
git checkout -b session-7-repomap

git add belief/repomap/ tests/test_repomap.py belief/cli.py pyproject.toml docs/session-7/

git commit -m "session-7: tree-sitter + PageRank repo-map (Aider port)

- belief/repomap/__init__.py: RepoMap class. Parses .py via
  tree-sitter, extracts func/class/assign defs + call/import/base
  refs, builds a networkx DiGraph, runs personalised PageRank with a
  chat_fnames/mentioned_fnames/mentioned_idents bias vector, emits
  top-N symbols as a token-budgeted grep-style tree.
- Cache: per-file parse results keyed by (path, mtime, size), stored
  in ~/.belief-engine/treesitter_cache/. Warm cache makes the belief/
  walk <1s; cold is ~5-10s.
- CLI: 'belief repomap' for manual inspection.
- pyproject.toml: tree-sitter, tree-sitter-python, networkx, scipy
  promoted to core deps (networkx 3.x's PageRank needs scipy).
- 8 hermetic tests — parse, ranking, personalization, budget, cache.
- Aider attribution in module docstring (github.com/Aider-AI/aider,
  Apache 2.0).

Note: the existing 'belief/agents/repo_map.py' (in-memory per-build
extractor) is untouched. The session-7 RepoMap is a different tool —
it walks the full source tree for cross-build context. Future session
integrates the session-7 RepoMap into agents' context budgets."
```

## Verify on Mac

```bash
pip3 install -e ".[dev,local]"
python3 -m pytest tests/test_repomap.py -v  # 8 passed
python3 -m pytest tests/ -q --timeout=60    # ~1107 passed (1099 after session 6 + 8 new)
```

Then try the CLI:
```bash
belief repomap --top 500
belief repomap --query "AsyncOllamaClient" --top 1000
```

## Not in this session

- Agent wiring (builder/architect/debugger context budgets) — the
  in-memory `belief/agents/repo_map.py` is already doing this role
  for code_files; the session-7 RepoMap is for cross-build. Follow-up
  session can integrate if needed.
- tree-sitter-languages bundled multi-language parser — Python-only
  for now.

## Merge

```bash
git checkout main
git merge --no-ff session-7-repomap
git push origin main
```
