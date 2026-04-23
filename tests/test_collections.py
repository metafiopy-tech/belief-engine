"""Tests for the 5-collection architecture (belief/memory/collections.py + soil.py).

Covers:
  - All 5 collections created with cosine space
  - Deposit into correct collection based on nutrient type
  - Retrieve across collections
  - FSRS metadata present on all records
  - Soil health metrics compute correctly
  - Migration from legacy collection works
"""

import chromadb
import pytest

from belief.memory.collections import (
    COLLECTION_CONFIGS,
    collection_for_nutrient_type,
    get_or_create_collections,
    migrate_from_legacy,
    _add_fsrs_defaults,
)
from belief.memory.nutrients import Nutrient, NutrientType
from belief.memory.soil import Soil, _HashEmbeddingFunction


@pytest.fixture
def ephemeral_client():
    """Create an ephemeral ChromaDB client for testing.

    Clears all collections to ensure test isolation since ChromaDB's
    ephemeral client may reuse in-memory state within the process.
    """
    client = chromadb.Client()
    # Clear any leftover collections from previous tests
    for col in client.list_collections():
        client.delete_collection(col.name)
    return client


@pytest.fixture
def ef():
    """Hash embedding function for tests."""
    return _HashEmbeddingFunction()


@pytest.fixture
def soil(tmp_path):
    """Create a Soil instance backed by a temp directory."""
    return Soil(persist_dir=tmp_path / "soil")


# ── Collection creation ────────────────────────────────────────────────────


class TestCollectionCreation:
    def test_all_five_collections_created(self, ephemeral_client, ef):
        """get_or_create_collections should create all 5 collections."""
        collections = get_or_create_collections(ephemeral_client, ef)
        assert len(collections) == 5
        for name in COLLECTION_CONFIGS:
            assert name in collections

    def test_cosine_space(self, ephemeral_client, ef):
        """All collections should use cosine distance."""
        collections = get_or_create_collections(ephemeral_client, ef)
        for name, col in collections.items():
            meta = col.metadata
            assert meta.get("hnsw:space") == "cosine", f"{name} should use cosine space"

    def test_idempotent(self, ephemeral_client, ef):
        """Calling get_or_create_collections twice should be safe."""
        c1 = get_or_create_collections(ephemeral_client, ef)
        c2 = get_or_create_collections(ephemeral_client, ef)
        assert set(c1.keys()) == set(c2.keys())


class TestCollectionRouting:
    def test_pattern_routes_to_principles(self):
        assert collection_for_nutrient_type("pattern") == "belief_principles"

    def test_antipattern_routes_to_failures(self):
        assert collection_for_nutrient_type("antipattern") == "belief_failures"

    def test_skeleton_routes_to_tools(self):
        assert collection_for_nutrient_type("skeleton") == "belief_tools"

    def test_covenant_routes_to_covenants(self):
        assert collection_for_nutrient_type("covenant") == "belief_covenants"

    def test_unknown_routes_to_episodes(self):
        assert collection_for_nutrient_type("something_else") == "belief_episodes"


# ── Deposit routing ─────────────────────────────────────────────────────────


class TestDepositRouting:
    def test_pattern_deposited_in_principles(self, soil):
        """A PATTERN nutrient should end up in belief_principles."""
        n = Nutrient(
            nutrient_type=NutrientType.PATTERN,
            content="Use dependency injection for testability",
            embedding_text="dependency injection pattern for testable code",
        )
        soil.deposit(n)
        col = soil._collections["belief_principles"]
        assert col.count() == 1

    def test_antipattern_deposited_in_failures(self, soil):
        n = Nutrient(
            nutrient_type=NutrientType.ANTIPATTERN,
            content="Don't catch bare Exception",
            embedding_text="catching bare exception hides bugs",
        )
        soil.deposit(n)
        col = soil._collections["belief_failures"]
        assert col.count() == 1

    def test_skeleton_deposited_in_tools(self, soil):
        n = Nutrient(
            nutrient_type=NutrientType.SKELETON,
            content="FastAPI scaffold with routers",
            embedding_text="fastapi project skeleton with router pattern",
        )
        soil.deposit(n)
        col = soil._collections["belief_tools"]
        assert col.count() == 1

    def test_covenant_deposited_in_covenants(self, soil):
        n = Nutrient(
            nutrient_type=NutrientType.COVENANT,
            content="Always pin dependency versions",
            embedding_text="pin dependency versions in requirements",
        )
        soil.deposit(n)
        col = soil._collections["belief_covenants"]
        assert col.count() == 1


# ── Cross-collection retrieval ──────────────────────────────────────────────


class TestCrossCollectionRetrieval:
    def test_retrieve_across_collections(self, soil):
        """retrieve() without a type filter should search all collections."""
        soil.deposit(
            Nutrient(
                nutrient_type=NutrientType.PATTERN,
                content="Use async/await for I/O-bound operations",
                embedding_text="async await pattern for io bound code",
            )
        )
        soil.deposit(
            Nutrient(
                nutrient_type=NutrientType.COVENANT,
                content="Always validate user input at API boundary",
                embedding_text="input validation at api boundary rule",
            )
        )

        results = soil.retrieve("api code patterns", n=10)
        assert len(results) == 2

    def test_retrieve_with_type_filter(self, soil):
        """retrieve() with a type should only search that collection."""
        soil.deposit(
            Nutrient(
                nutrient_type=NutrientType.PATTERN,
                content="Use pydantic models for validation",
                embedding_text="pydantic model validation pattern",
            )
        )
        soil.deposit(
            Nutrient(
                nutrient_type=NutrientType.ANTIPATTERN,
                content="Don't use dict for API responses",
                embedding_text="dict instead of pydantic for api response antipattern",
            )
        )

        patterns = soil.retrieve("validation", nutrient_type=NutrientType.PATTERN)
        assert all(n.nutrient_type == NutrientType.PATTERN for n in patterns)

    def test_retrieve_profile_populates_all_categories(self, soil):
        """retrieve_profile should populate covenants, antipatterns, patterns, skeletons."""
        soil.deposit(
            Nutrient(
                nutrient_type=NutrientType.PATTERN,
                content="REST API with proper status codes",
                embedding_text="rest api status code pattern",
            )
        )
        soil.deposit(
            Nutrient(
                nutrient_type=NutrientType.ANTIPATTERN,
                content="Returning 200 for errors",
                embedding_text="returning 200 for error responses antipattern",
            )
        )
        soil.deposit(
            Nutrient(
                nutrient_type=NutrientType.COVENANT,
                content="Always return proper HTTP status codes",
                embedding_text="http status code covenant rule",
            )
        )

        profile = soil.retrieve_profile("build a REST API")
        assert len(profile.patterns) >= 1
        assert len(profile.antipatterns) >= 1
        assert len(profile.covenants) >= 1


# ── FSRS metadata ──────────────────────────────────────────────────────────


class TestFSRSMetadata:
    def test_fsrs_defaults_added(self):
        """_add_fsrs_defaults should add all FSRS fields."""
        meta = {"nutrient_type": "pattern"}
        enriched = _add_fsrs_defaults(meta)
        assert "fsrs_stability" in enriched
        assert "fsrs_difficulty" in enriched
        assert "fsrs_reps" in enriched
        assert "fsrs_lapses" in enriched
        assert "fsrs_decay_state" in enriched
        assert enriched["fsrs_stability"] == 1.0
        assert enriched["fsrs_decay_state"] == "new"

    def test_fsrs_defaults_dont_overwrite(self):
        """Existing FSRS fields should not be overwritten."""
        meta = {"fsrs_stability": 99.0, "fsrs_decay_state": "stable"}
        enriched = _add_fsrs_defaults(meta)
        assert enriched["fsrs_stability"] == 99.0
        assert enriched["fsrs_decay_state"] == "stable"

    def test_deposited_nutrient_has_fsrs_metadata(self, soil):
        """After deposit, the record should have FSRS fields in its metadata."""
        n = Nutrient(
            nutrient_type=NutrientType.PATTERN,
            content="Test pattern",
            embedding_text="test pattern for metadata check",
        )
        nid = soil.deposit(n)

        col = soil._collections["belief_principles"]
        result = col.get(ids=[nid], include=["metadatas"])
        meta = result["metadatas"][0]
        assert "fsrs_stability" in meta
        assert "fsrs_difficulty" in meta
        assert "fsrs_decay_state" in meta
        assert meta["fsrs_decay_state"] == "new"


# ── Review and due reviews ──────────────────────────────────────────────────


class TestReviewNutrient:
    def test_review_updates_fsrs_state(self, soil):
        """review_nutrient should update FSRS fields in the collection."""
        n = Nutrient(
            nutrient_type=NutrientType.PATTERN,
            content="Test review update",
            embedding_text="test review update pattern",
        )
        nid = soil.deposit(n)

        soil.review_nutrient(nid, "belief_principles", grade=3)

        col = soil._collections["belief_principles"]
        result = col.get(ids=[nid], include=["metadatas"])
        meta = result["metadatas"][0]
        assert meta["fsrs_reps"] == 1
        assert meta["fsrs_decay_state"] == "learning"
        assert meta["fsrs_last_review"] > 0

    def test_get_due_reviews(self, soil):
        """get_due_reviews should return records with next_review <= now."""
        n = Nutrient(
            nutrient_type=NutrientType.PATTERN,
            content="Due for review",
            embedding_text="due for review test pattern",
        )
        nid = soil.deposit(n)

        # Default fsrs_next_review is 0.0 which is <= any current time
        from datetime import datetime, timezone

        due = soil.get_due_reviews("belief_principles", now=datetime.now(timezone.utc))
        assert any(d["id"] == nid for d in due)


# ── Soil health ─────────────────────────────────────────────────────────────


class TestSoilHealth:
    def test_empty_soil_health(self, soil):
        """Empty soil should return all zeros."""
        health = soil.get_soil_health()
        assert health["duplicate_rate"] == 0.0
        assert health["staleness"] == 0.0
        assert health["lapse_rate"] == 0.0
        assert all(v == 0 for v in health["total_count"].values())

    def test_soil_health_with_data(self, soil):
        """Health metrics should compute with deposited data."""
        soil.deposit(
            Nutrient(
                nutrient_type=NutrientType.PATTERN,
                content="Pattern A for health check",
                embedding_text="health check pattern a",
            )
        )
        soil.deposit(
            Nutrient(
                nutrient_type=NutrientType.ANTIPATTERN,
                content="Antipattern B for health check",
                embedding_text="health check antipattern b",
            )
        )

        health = soil.get_soil_health()
        assert health["total_count"]["belief_principles"] == 1
        assert health["total_count"]["belief_failures"] == 1
        assert health["duplicate_rate"] == 0.0  # Different content
        assert health["staleness"] == 0.0  # Just created

    def test_soil_health_counts_all_collections(self, soil):
        """total_count should have entries for all 5 collections."""
        health = soil.get_soil_health()
        assert len(health["total_count"]) == 5
        for name in COLLECTION_CONFIGS:
            assert name in health["total_count"]


# ── Migration ───────────────────────────────────────────────────────────────


class TestMigration:
    def test_migrate_from_legacy(self, ephemeral_client, ef):
        """Migration should move legacy records into the correct new collections."""
        # Create fake legacy data
        legacy = ephemeral_client.get_or_create_collection(
            name="nutrients",
            metadata={"hnsw:space": "cosine"},
            embedding_function=ef,
        )

        legacy.upsert(
            ids=["n-001", "n-002", "n-003"],
            documents=[
                "fastapi dependency injection pattern",
                "bare exception catching antipattern",
                "always validate input covenant",
            ],
            metadatas=[
                {
                    "nutrient_type": "pattern",
                    "content": "DI pattern",
                    "stability": 5.0,
                    "difficulty": 3.0,
                },
                {
                    "nutrient_type": "antipattern",
                    "content": "Bare except",
                    "stability": 2.0,
                    "difficulty": 7.0,
                },
                {
                    "nutrient_type": "covenant",
                    "content": "Validate input",
                    "stability": 10.0,
                    "difficulty": 4.0,
                },
            ],
        )

        assert legacy.count() == 3

        # Run migration
        counts = migrate_from_legacy(ephemeral_client, "nutrients", ef)

        assert counts["belief_principles"] == 1  # pattern
        assert counts["belief_failures"] == 1  # antipattern
        assert counts["belief_covenants"] == 1  # covenant

        # Verify records exist in new collections
        collections = get_or_create_collections(ephemeral_client, ef)
        principles = collections["belief_principles"]
        result = principles.get(ids=["n-001"], include=["metadatas"])
        assert result["ids"] == ["n-001"]
        meta = result["metadatas"][0]
        # FSRS defaults should be added
        assert "fsrs_stability" in meta
        assert "fsrs_decay_state" in meta
        # Original metadata preserved
        assert meta["content"] == "DI pattern"

    def test_legacy_collection_not_deleted(self, ephemeral_client, ef):
        """Migration should NOT delete the legacy collection."""
        legacy = ephemeral_client.get_or_create_collection(
            name="nutrients",
            metadata={"hnsw:space": "cosine"},
            embedding_function=ef,
        )
        legacy.upsert(
            ids=["n-001"],
            documents=["test doc"],
            metadatas=[{"nutrient_type": "pattern", "content": "test"}],
        )

        migrate_from_legacy(ephemeral_client, "nutrients", ef)

        # Legacy should still exist with original data
        legacy_after = ephemeral_client.get_collection(name="nutrients", embedding_function=ef)
        assert legacy_after.count() == 1

    def test_migrate_empty_legacy(self, ephemeral_client, ef):
        """Migration of an empty collection should be a no-op."""
        ephemeral_client.get_or_create_collection(
            name="nutrients",
            metadata={"hnsw:space": "cosine"},
            embedding_function=ef,
        )
        counts = migrate_from_legacy(ephemeral_client, "nutrients", ef)
        assert all(v == 0 for v in counts.values())

    def test_migrate_nonexistent_legacy(self, ephemeral_client, ef):
        """Migration should handle missing legacy collection gracefully."""
        counts = migrate_from_legacy(ephemeral_client, "nonexistent", ef)
        assert all(v == 0 for v in counts.values())


# ── Backward compatibility ──────────────────────────────────────────────────


class TestBackwardCompat:
    def test_soil_count_sums_all_collections(self, soil):
        """soil.count() should return total across all collections."""
        soil.deposit(
            Nutrient(
                nutrient_type=NutrientType.PATTERN,
                content="P1",
                embedding_text="pattern one",
            )
        )
        soil.deposit(
            Nutrient(
                nutrient_type=NutrientType.COVENANT,
                content="C1",
                embedding_text="covenant one",
            )
        )
        assert soil.count() == 2

    def test_soil_get_finds_across_collections(self, soil):
        """soil.get() should find a nutrient regardless of which collection it's in."""
        n = Nutrient(
            nutrient_type=NutrientType.COVENANT,
            content="Find me",
            embedding_text="findable covenant",
        )
        nid = soil.deposit(n)
        found = soil.get(nid)
        assert found is not None
        assert found.content == "Find me"

    def test_deposit_retrieve_roundtrip(self, soil):
        """deposit -> retrieve should return the same nutrient."""
        n = Nutrient(
            nutrient_type=NutrientType.PATTERN,
            content="Roundtrip test pattern",
            embedding_text="roundtrip test pattern for deposit retrieve",
        )
        nid = soil.deposit(n)
        results = soil.retrieve("roundtrip test", nutrient_type=NutrientType.PATTERN)
        assert len(results) >= 1
        assert any(r.nutrient_id == nid for r in results)
