"""Tests for the atomizer (SE Session 5)."""

from __future__ import annotations

import pytest

from belief.photosynthesis.synthesis.atomizer import (
    AXES,
    AXIS_TEMPLATES,
    ResearchPrompt,
    atomize_word,
    atomize_words,
    prompts_per_word,
)


# ---------------------------------------------------------------------------
# Axes + templates -- pinned shape
# ---------------------------------------------------------------------------


class TestAxes:
    def test_five_axes_pinned(self) -> None:
        assert AXES == (
            "mechanism",
            "constraint",
            "tradeoff",
            "cross_domain",
            "counterexample",
        )

    def test_three_templates_per_axis(self) -> None:
        for axis in AXES:
            assert axis in AXIS_TEMPLATES
            assert len(AXIS_TEMPLATES[axis]) == 3

    def test_every_template_uses_word_placeholder(self) -> None:
        for axis, templates in AXIS_TEMPLATES.items():
            for tpl in templates:
                assert "{word}" in tpl, f"axis {axis} has a template without {{word}}: {tpl[:60]}"

    def test_counterexample_axis_seeks_failure_modes(self) -> None:
        """SE plan acceptance: 'Counterexample axis produces prompts
        that explicitly seek failure modes.'"""
        templates = AXIS_TEMPLATES["counterexample"]
        joined = " ".join(templates).lower()
        assert "fail" in joined
        # At least one template should explicitly contemplate "where
        # does X NOT appear" -- the negation framing the SE plan
        # called out.
        assert any(
            "fail" in t.lower() or "not appear" in t.lower() or "break" in t.lower()
            for t in templates
        )


# ---------------------------------------------------------------------------
# Single-word atomization
# ---------------------------------------------------------------------------


class TestAtomizeWord:
    def test_returns_fifteen_prompts(self) -> None:
        prompts = atomize_word("mantis_shrimp")
        assert len(prompts) == 15
        assert prompts_per_word() == 15

    def test_all_axes_represented(self) -> None:
        prompts = atomize_word("mantis_shrimp")
        axes_seen = {p.axis for p in prompts}
        assert axes_seen == set(AXES)

    def test_three_prompts_per_axis(self) -> None:
        prompts = atomize_word("mantis_shrimp")
        from collections import Counter

        counts = Counter(p.axis for p in prompts)
        for axis in AXES:
            assert counts[axis] == 3

    def test_word_substituted_into_query(self) -> None:
        prompts = atomize_word("mantis_shrimp")
        for p in prompts:
            assert "mantis_shrimp" in p.query
            assert "{word}" not in p.query

    def test_each_prompt_carries_word_field(self) -> None:
        prompts = atomize_word("mantis_shrimp")
        assert all(p.word == "mantis_shrimp" for p in prompts)

    def test_axes_in_canonical_order(self) -> None:
        """The first 3 prompts should all be the 'mechanism' axis,
        next 3 'constraint', etc."""
        prompts = atomize_word("camera")
        for i, axis in enumerate(AXES):
            for j in range(3):
                assert prompts[i * 3 + j].axis == axis

    def test_empty_word_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            atomize_word("")
        with pytest.raises(ValueError, match="non-empty"):
            atomize_word("   ")

    def test_n_per_axis_clamps(self) -> None:
        # n_per_axis above the template count silently caps.
        prompts = atomize_word("camera", n_per_axis=99)
        assert len(prompts) == 15
        # Below 1 clamps up to 1.
        prompts = atomize_word("camera", n_per_axis=0)
        assert len(prompts) == 5  # 1 per axis

    def test_source_hints_applied_per_axis(self) -> None:
        prompts = atomize_word("mantis_shrimp")
        # Each axis has its own configured hint set; check a few.
        cross_domain = [p for p in prompts if p.axis == "cross_domain"]
        assert all("arxiv" in p.source_hints for p in cross_domain)
        mechanism = [p for p in prompts if p.axis == "mechanism"]
        assert all("arxiv" in p.source_hints for p in mechanism)
        assert all("github" in p.source_hints for p in mechanism)


# ---------------------------------------------------------------------------
# Multi-word atomization
# ---------------------------------------------------------------------------


class TestAtomizeWords:
    def test_two_words_thirty_prompts(self) -> None:
        """SE plan acceptance: '2 input words produce 20-30 research
        prompts (10-15 per word across five axes).'"""
        prompts = atomize_words(["mantis_shrimp", "camera"])
        assert len(prompts) == 30

    def test_words_appear_in_input_order(self) -> None:
        """First word's prompts come before second word's."""
        prompts = atomize_words(["alpha", "beta"])
        first_words = [p.word for p in prompts[:15]]
        second_words = [p.word for p in prompts[15:]]
        assert first_words == ["alpha"] * 15
        assert second_words == ["beta"] * 15

    def test_three_words_forty_five_prompts(self) -> None:
        prompts = atomize_words(["a", "b", "c"])
        assert len(prompts) == 45

    def test_empty_words_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one word"):
            atomize_words([])


# ---------------------------------------------------------------------------
# ResearchPrompt validation
# ---------------------------------------------------------------------------


class TestResearchPrompt:
    def test_unknown_axis_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown axis"):
            ResearchPrompt(word="x", axis="nonsense", query="q")

    def test_valid_construction(self) -> None:
        p = ResearchPrompt(word="x", axis="mechanism", query="how does x work?")
        assert p.word == "x"
        assert p.axis == "mechanism"
        assert p.source_hints == ()
