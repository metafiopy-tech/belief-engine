"""Tests for the bounded priority heap + saturation detector."""

from __future__ import annotations

from pathlib import Path

import pytest

from belief.photosynthesis.state import PhotosynthesisState
from belief.photosynthesis.synthesis.heap import (
    BoundedPriorityHeap,
    DEFAULT_CAPACITY,
    NoveltySaturation,
    SATURATION_CYCLES_REQUIRED,
)


@pytest.fixture()
def state(tmp_path: Path) -> PhotosynthesisState:
    return PhotosynthesisState(str(tmp_path / "signals.sqlite"))


def test_push_into_empty_heap_returns_true(state: PhotosynthesisState) -> None:
    heap = BoundedPriorityHeap(state, capacity=4)
    assert heap.push({"title": "a"}, 0.5) is True
    assert heap.size() == 1


def test_heap_respects_capacity(state: PhotosynthesisState) -> None:
    heap = BoundedPriorityHeap(state, capacity=3)
    # Fill
    for v in [0.2, 0.4, 0.6]:
        heap.push({"v": v}, v)
    assert heap.size() == 3

    # A new push below the current min is dropped
    assert heap.push({"v": 0.1}, 0.1) is False
    assert heap.size() == 3

    # A new push above the current min evicts the min
    assert heap.push({"v": 0.7}, 0.7) is True
    assert heap.size() == 3

    # The 0.2 entry must have been evicted
    top = heap.peek_top()
    assert top is not None and top.value == pytest.approx(0.7)
    minimum = heap.peek_min()
    assert minimum is not None and minimum.value >= 0.4


def test_pop_top_returns_highest_and_removes(state: PhotosynthesisState) -> None:
    heap = BoundedPriorityHeap(state, capacity=5)
    for v in [0.3, 0.9, 0.5, 0.8]:
        heap.push({"v": v}, v)

    top = heap.pop_top()
    assert top is not None and top.value == pytest.approx(0.9)
    assert heap.size() == 3
    # Next pop is 0.8
    top2 = heap.pop_top()
    assert top2 is not None and top2.value == pytest.approx(0.8)


def test_pop_from_empty_returns_none(state: PhotosynthesisState) -> None:
    heap = BoundedPriorityHeap(state, capacity=4)
    assert heap.pop_top() is None


def test_default_capacity_matches_spec() -> None:
    assert DEFAULT_CAPACITY == 64


def test_saturation_after_three_cycles(state: PhotosynthesisState) -> None:
    """If every cycle's min stays above 0.70, saturation triggers after 3 cycles."""
    heap = BoundedPriorityHeap(state, capacity=4)
    # Stock the heap with values all above 0.70 so min is high
    for v in [0.80, 0.85, 0.90, 0.95]:
        heap.push({"v": v}, v)

    # Record 3 cycles — each should mark saturation
    for _ in range(SATURATION_CYCLES_REQUIRED):
        heap.record_cycle(pushed_count=1)

    assert heap.check_saturation() is True
    with pytest.raises(NoveltySaturation):
        heap.raise_if_saturated()


def test_saturation_resets_on_low_cycle(state: PhotosynthesisState) -> None:
    heap = BoundedPriorityHeap(state, capacity=4)
    # First two cycles: high min (saturated)
    for v in [0.80, 0.85, 0.90, 0.95]:
        heap.push({"v": v}, v)
    heap.record_cycle()
    heap.record_cycle()

    # Third cycle: insert a low-value that drops the min below 0.70
    heap.pop_top()
    heap.push({"v": 0.10}, 0.10)
    heap.record_cycle()

    # 3 consecutive saturated cycles not met (the third was not saturated)
    assert heap.check_saturation() is False
