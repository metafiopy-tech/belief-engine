"""Tests for the FSRS-4.5 implementation (belief/memory/fsrs.py).

Covers:
  - retrievability at t=0 and t=stability
  - stability growth on success (grade 3, 4)
  - stability collapse on failure (grade 1)
  - difficulty clamping to [1.0, 10.0]
  - schedule_next_review returns positive days
  - review() transitions decay_state correctly
  - Full lifecycle: 5 reviews with mixed grades
"""

from datetime import datetime, timedelta, timezone


from belief.memory.fsrs import (
    FSRSState,
    retrievability,
    review,
    schedule_next_review,
    update_difficulty,
    update_stability_on_failure,
    update_stability_on_success,
)


# ── retrievability ──────────────────────────────────────────────────────────


class TestRetrievability:
    def test_at_t0_is_one(self):
        """R(0, S) = 1.0 for any positive stability."""
        assert retrievability(1.0, 0.0) == 1.0
        assert retrievability(10.0, 0.0) == 1.0
        assert retrievability(100.0, 0.0) == 1.0

    def test_at_stability_is_about_0_9(self):
        """R(S, S) should be approximately 0.9 (the defining FSRS property)."""
        for s in (1.0, 5.0, 30.0, 365.0):
            r = retrievability(s, s)
            assert abs(r - 0.9) < 0.01, f"R({s}, {s}) = {r}, expected ~0.9"

    def test_monotonically_decreasing(self):
        """Retrievability should decrease as elapsed time increases."""
        s = 10.0
        prev = 1.0
        for t in (1.0, 5.0, 10.0, 50.0, 100.0):
            r = retrievability(s, t)
            assert r < prev, f"R not decreasing: R({s},{t})={r} >= {prev}"
            prev = r

    def test_zero_stability_returns_zero(self):
        assert retrievability(0.0, 1.0) == 0.0
        assert retrievability(-1.0, 1.0) == 0.0

    def test_negative_elapsed_returns_one(self):
        assert retrievability(5.0, -1.0) == 1.0


# ── update_stability_on_success ─────────────────────────────────────────────


class TestStabilitySuccess:
    def test_stability_grows(self):
        """Stability should increase on a successful review."""
        s = 5.0
        d = 5.0
        r = 0.9
        new_s = update_stability_on_success(s, d, r)
        assert new_s > s, f"Expected growth: {new_s} > {s}"

    def test_lower_difficulty_grows_faster(self):
        """Easy items (low D) should gain more stability than hard items."""
        s, r = 5.0, 0.7
        easy = update_stability_on_success(s, 2.0, r)
        hard = update_stability_on_success(s, 9.0, r)
        assert easy > hard, f"Easy ({easy}) should grow more than hard ({hard})"

    def test_spaced_review_grows_more(self):
        """Lower retrievability (longer gap) should give bigger stability boost."""
        s, d = 5.0, 5.0
        spaced = update_stability_on_success(s, d, 0.5)   # Longer gap
        massed = update_stability_on_success(s, d, 0.95)   # Just reviewed
        assert spaced > massed, f"Spaced ({spaced}) should exceed massed ({massed})"


# ── update_stability_on_failure ─────────────────────────────────────────────


class TestStabilityFailure:
    def test_stability_drops(self):
        """Stability should decrease on failure."""
        s = 20.0
        d = 5.0
        r = 0.7
        new_s = update_stability_on_failure(s, d, r)
        assert new_s < s, f"Expected drop: {new_s} < {s}"

    def test_floor_at_0_1(self):
        """Stability should never drop below 0.1."""
        new_s = update_stability_on_failure(0.2, 10.0, 0.1)
        assert new_s >= 0.1


# ── update_difficulty ───────────────────────────────────────────────────────


class TestDifficulty:
    def test_grade_3_no_change(self):
        """Grade 3 (good) should leave difficulty unchanged."""
        assert update_difficulty(5.0, 3) == 5.0

    def test_grade_1_decreases(self):
        """Grade 1 (again) pushes difficulty down: D + 0.1*(1-3) = D - 0.2."""
        d = update_difficulty(5.0, 1)
        assert d < 5.0

    def test_grade_4_increases(self):
        """Grade 4 (easy) pushes difficulty up: D + 0.1*(4-3) = D + 0.1."""
        d = update_difficulty(5.0, 4)
        assert d > 5.0

    def test_clamped_low(self):
        """Difficulty should never drop below 1.0."""
        d = update_difficulty(1.0, 4)
        assert d >= 1.0

    def test_clamped_high(self):
        """Difficulty should never exceed 10.0."""
        d = update_difficulty(10.0, 1)
        assert d <= 10.0


# ── schedule_next_review ────────────────────────────────────────────────────


class TestScheduleNextReview:
    def test_positive_days(self):
        """Should always return a positive interval."""
        days = schedule_next_review(5.0)
        assert days > 0

    def test_higher_stability_longer_interval(self):
        short = schedule_next_review(1.0)
        long = schedule_next_review(100.0)
        assert long > short

    def test_zero_stability(self):
        assert schedule_next_review(0.0) == 0.0

    def test_default_retention_0_9(self):
        """At 0.9 retention the interval should equal the stability (approximately)."""
        s = 10.0
        interval = schedule_next_review(s, desired_retention=0.9)
        # R(interval, s) should be ~0.9
        r = retrievability(s, interval)
        assert abs(r - 0.9) < 0.01, f"R at scheduled interval: {r}"


# ── review() ────────────────────────────────────────────────────────────────


class TestReview:
    def test_new_to_learning(self):
        """First successful review should transition new -> learning."""
        state = FSRSState()
        assert state.decay_state == "new"
        new_state = review(state, grade=3)
        assert new_state.decay_state == "learning"
        assert new_state.reps == 1

    def test_learning_to_stable_after_3(self):
        """After 3 successful reviews, should transition learning -> stable."""
        state = FSRSState()
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        for i in range(3):
            state = review(state, grade=3, now=now + timedelta(days=i))
        assert state.decay_state == "stable"
        assert state.reps == 3

    def test_stable_to_lapsed_on_failure(self):
        """A failure (grade 1) after becoming stable should transition to lapsed."""
        state = FSRSState()
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        # Become stable
        for i in range(3):
            state = review(state, grade=3, now=now + timedelta(days=i))
        assert state.decay_state == "stable"
        # Fail
        state = review(state, grade=1, now=now + timedelta(days=5))
        assert state.decay_state == "lapsed"
        assert state.lapses == 1

    def test_lapsed_recovers_on_success(self):
        """Successful review after lapse should transition to learning."""
        state = FSRSState(decay_state="lapsed", lapses=1, reps=3,
                          last_review=datetime(2025, 1, 1, tzinfo=timezone.utc))
        state = review(state, grade=3)
        assert state.decay_state == "learning"

    def test_stability_grows_on_success(self):
        state = FSRSState(stability=5.0,
                          last_review=datetime(2025, 1, 1, tzinfo=timezone.utc))
        new_state = review(state, grade=3, now=datetime(2025, 1, 5, tzinfo=timezone.utc))
        assert new_state.stability > 5.0

    def test_stability_drops_on_failure(self):
        state = FSRSState(stability=20.0,
                          last_review=datetime(2025, 1, 1, tzinfo=timezone.utc))
        new_state = review(state, grade=1, now=datetime(2025, 1, 5, tzinfo=timezone.utc))
        assert new_state.stability < 20.0

    def test_next_review_scheduled(self):
        state = FSRSState()
        new_state = review(state, grade=3)
        assert new_state.next_review is not None
        assert new_state.last_review is not None
        assert new_state.next_review > new_state.last_review

    def test_grade_clamped(self):
        """Grades outside 1-4 should be clamped."""
        state = FSRSState()
        s1 = review(state, grade=0)   # Clamped to 1
        s2 = review(state, grade=10)  # Clamped to 4
        assert s1.lapses == 1   # grade 1 = failure
        assert s2.reps == 1     # grade 4 = success


# ── Full lifecycle ──────────────────────────────────────────────────────────


class TestFullLifecycle:
    def test_five_reviews_mixed_grades(self):
        """5 reviews with mixed grades should produce a reasonable final state."""
        state = FSRSState()
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        grades = [3, 4, 3, 1, 3]  # good, easy, good, again, good

        for i, grade in enumerate(grades):
            state = review(state, grade=grade, now=now + timedelta(days=i * 3))

        # After 3 successes, 1 failure, 1 recovery:
        assert state.reps == 4        # 4 successes total
        assert state.lapses == 1      # 1 failure
        assert state.stability > 0    # Still has some stability
        assert 1.0 <= state.difficulty <= 10.0
        assert state.last_review is not None
        assert state.next_review is not None
        # Should be in learning state (recovered from lapse)
        assert state.decay_state == "learning"

    def test_all_easy_reviews(self):
        """5 consecutive easy reviews should produce high stability."""
        state = FSRSState()
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)

        for i in range(5):
            state = review(state, grade=4, now=now + timedelta(days=i * 7))

        assert state.stability > 10.0  # Should have grown significantly
        assert state.difficulty > 5.0  # Grade 4 pushes difficulty up per formula
        assert state.decay_state == "stable"
        assert state.reps == 5
        assert state.lapses == 0
