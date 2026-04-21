"""Shared cross-cutting primitives: HTTP, config, etc.

Modules here are deliberately narrow — common infrastructure that more
than one subsystem needs (e.g. the Photosynthesis daemon and any future
webhook consumer both want a circuit-broken httpx client). Keep it
dependency-light; anything that needs heavy ML / scheduler deps belongs
in its own subsystem package.
"""
