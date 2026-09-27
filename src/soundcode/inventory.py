"""Which instrument is a stem really? Two sources vote.

The separator's stem name is only a prior. tsumugi's refinement model says
what the stem sounds like *within* that prior; the mix-level transcription
(no stem prior at all) says which family each of the stem's notes belongs to.
Agreement is confidence; strong disagreement reassigns the stem.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import gm
from .tsumugi_sc import Track

REASSIGN_SHARE = 0.60
MIN_VOTES = 5
LOW_CONF = 0.5
STEM_FAMILY = {"piano": "keys", "guitar": "gtr", "bass": "bass", "other": "other",
               "drums": "drums", "lead_vocals": "voice", "backing_vocals": "voice",
               "vocals": "voice"}
# the separator is reliable on these; a vote never moves them
NEVER_REASSIGN = frozenset({"voice", "drums"})
FAMILY_REFINE_STEM = {"keys": "piano", "gtr": "guitar", "bass": "bass", "voice": "vocals"}


@dataclass
class Decision:
    family: str
    reassigned: bool
    conf: float
    votes: dict[str, int] = field(default_factory=dict)
    warn: str | None = None


def _family(klass: str) -> str:
    return gm.TSUMUGI[klass]["family"]


def mix_votes(stem_tracks: list[Track], mix_tracks: list[Track],
              prefer: str | None = None) -> dict[str, int]:
    """Family votes for the stem's notes. When a matching mix note of the
    `prefer`red family exists (the stem's own), it wins: a sung line doubled
    by a piano is still a voice."""
    mix = [(s, p, _family(t.klass)) for t in mix_tracks for s, _, p, _ in t.notes]
    votes: dict[str, int] = {}
    for t in stem_tracks:
        for s, _, p, _ in t.notes:
            hits = [f for ms, mp, f in mix if abs(ms - s) <= 0.05 and abs(mp - p) <= 1]
            if not hits:
                continue
            hit = prefer if prefer in hits else hits[0]
            votes[hit] = votes.get(hit, 0) + 1
    return votes


def decide(stem: str, prior_family: str, refine_top: list[tuple[str, float]],
           votes: dict[str, int]) -> Decision:
    p_refine = refine_top[0][1] if refine_top else 0.0
    total = sum(votes.values())
    if total < MIN_VOTES:
        return Decision(prior_family, False, p_refine, votes,
                        f"few mix votes ({total}); refinement only")
    top_family, top_n = max(votes.items(), key=lambda kv: kv[1])
    if prior_family in NEVER_REASSIGN:
        agree = votes.get(prior_family, 0) / total
        return Decision(prior_family, False, max(p_refine * agree, p_refine * 0.5), votes, None)
    share = top_n / total
    if top_family != prior_family and share >= REASSIGN_SHARE:
        return Decision(top_family, True, share, votes,
                        f"{stem} stem reassigned to {top_family} by mix-level vote {share:.2f}")
    agree = votes.get(prior_family, 0) / total
    return Decision(prior_family, False, p_refine * agree, votes, None)
