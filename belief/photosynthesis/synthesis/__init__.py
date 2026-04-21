"""Goal synthesis: convert filter survivors into Belief Engine session specs.

The pipeline:

    survivor seed
        |
        v
    novelty.score_novelty()   -- ChromaDB neighbor cosine + Haiku judge
        |
        v    (novelty < 0 -> reject)
    difficulty.estimate_difficulty()  -- ZPD fit, POET minimal criterion
        |
        v    (zpd out of band -> reject)
    ranker.combined_value()   -- weighted score + Bittensor bias
        |
        v    (value < 0.45 -> reject)
    heap.BoundedPriorityHeap.push()
        |
        v
    generator.synthesize()    -- Sonnet K=4 + post-dup check
        |
        v
    renderer.write_session()  -- pending_sessions/{goal_id}.{md,json}

The research doc's §4.2 generator prompt template isn't shipped in the
uploaded material; GENERATOR_PROMPT in prompts.py is a spec-faithful
stand-in that covers every required output JSON field. Session 5 or a
later edit can swap it out without touching any caller.
"""

__all__: list[str] = []
