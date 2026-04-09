# Contributing to Belief Engine

## Architecture Overview

The Belief Engine is a 74-file Python system with 11+ LangGraph agents that collaborate in a convergence loop. Before contributing, read `CLAUDE.md` for the full technical reference.

## Key Principles

1. **Deterministic over probabilistic.** If something can be enforced via AST checks instead of prompt injection, do it that way. See `belief/validators/` for examples.

2. **Zero LLM tokens when possible.** The skeleton generator, covenant enforcer, import fixer, and validator all run without LLM calls. Keep it that way.

3. **The soil remembers.** Every build deposits nutrients. If you change agent behavior, the soil's patterns may become stale. Consider whether existing nutrients need archiving.

4. **Covenants are immutable.** Self-learned covenants in ChromaDB reflect real failure clusters. Don't override them in prompts — enforce them structurally.

## Development Setup

```bash
git clone https://github.com/metafiopy-tech/belief-engine.git
cd belief-engine
pip install -e ".[dev]"
cp .env.example .env  # Add ANTHROPIC_API_KEY
```

## Running Tests

```bash
# Unit tests
pytest tests/ -v

# Verify the pipeline compiles
python3 -c "
from belief.graph import build_pipeline
from belief.config import ModelRouter
build_pipeline(ModelRouter())
print('Pipeline OK')
"

# Run a build
python3 -m belief.cli --goal "Build a hello world FastAPI server"
```

## File Structure

```
belief/
  agents/       — One file per agent. Each has a run() method.
  validators/   — Deterministic AST checks (Move 2)
  memory/       — ChromaDB soil, nutrients, decomposer, recomposer
  refinement/   — Water cycle (analyze → fix → revalidate)
  models/       — Pydantic models for state and artifacts
  config/       — Model routing (Sonnet vs Haiku per agent)
```

## Adding a New Agent

1. Create `belief/agents/my_agent.py` inheriting from `BaseAgent`
2. Add a `ModelRole` entry in `belief/config/models.py`
3. Wire the node in `belief/graph.py`
4. Add to the pipeline edge chain

## Adding a New Covenant Enforcer

1. Add a function in `belief/validators/__init__.py` following the pattern:
   ```python
   def _enforce_my_rule(fname, code, uses_sqlalchemy) -> list[Violation]:
   ```
2. Add it to the `enforcers` list in `enforce_all()`
3. Test it deterministically — no LLM needed

## Pull Request Checklist

- [ ] Pipeline compiles: `python3 -c "from belief.graph import build_pipeline; ..."`
- [ ] No new LLM calls in deterministic paths
- [ ] Existing tests pass
- [ ] CLAUDE.md updated if architecture changed
