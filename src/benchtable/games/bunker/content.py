"""Authored neutral dossier pools for the bunker game.

Original content authored for benchtable.  Deliberately avoids sensitive
real-world attributes (religion, ethnicity, politics, protected
characteristics) and any commercial game's deck text.
"""

from __future__ import annotations

DOSSIER_CATEGORIES: tuple[str, ...] = ("profession", "health", "skill", "trait")

DOSSIER_VALUES: dict[str, tuple[str, ...]] = {
    "profession": (
        "structural engineer",
        "primary-school teacher",
        "paramedic",
        "organic farmer",
        "electrician",
        "librarian",
        "veterinarian",
        "carpenter",
        "radio technician",
        "line cook",
    ),
    "health": (
        "fully healthy",
        "mild pollen allergy",
        "wearing a leg brace",
        "recovering from a broken arm",
        "chronic back pain",
        "near-sighted",
        "hard of hearing in one ear",
        "managed asthma",
        "recovering from the flu",
        "color-blind",
    ),
    "skill": (
        "advanced first aid",
        "water purification",
        "basic electronics repair",
        "foraging",
        "public speaking",
        "weather forecasting",
        "woodworking",
        "amateur radio operation",
        "improvised machinery repair",
        "cooking for a crowd",
    ),
    "trait": (
        "calm under pressure",
        "habitual pessimist",
        "natural negotiator",
        "restless tinkerer",
        "quiet observer",
        "gifted storyteller",
        "stubborn planner",
        "optimistic organizer",
        "dry sense of humor",
        "methodical note-taker",
    ),
}
