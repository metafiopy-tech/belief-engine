"""Controlled A/B experiment harness for the Belief Engine.

Compares three conditions:
  engine_cloud — Engine + Claude (Anthropic)
  engine_local — Engine + Ollama (local)
  raw_local    — Raw Ollama call, no engine (control)

Results are stored in ~/.belief-engine/experiments.db for longitudinal
analysis. The key question: does the engine add measurable value over
a bare model call on the same hardware?
"""
