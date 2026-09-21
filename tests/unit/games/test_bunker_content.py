"""Content sanity tests for the authored bunker dossier pools."""

from __future__ import annotations

import re

from benchtable.games.bunker.content import DOSSIER_CATEGORIES, DOSSIER_VALUES

_SENSITIVE_WORDS = ("religious", "muslim", "christian", "jewish", "white", "black")


def test_content_defines_exactly_the_initial_categories() -> None:
    assert DOSSIER_CATEGORIES == ("profession", "health", "skill", "trait")
    assert set(DOSSIER_VALUES) == set(DOSSIER_CATEGORIES)


def test_every_category_has_a_large_unique_nonblank_pool() -> None:
    for category in DOSSIER_CATEGORIES:
        values = DOSSIER_VALUES[category]
        assert len(values) >= 10
        assert len(set(values)) == len(values)
        assert all(type(value) is str and value.strip() for value in values)


def test_values_avoid_sensitive_descriptors() -> None:
    for category, values in DOSSIER_VALUES.items():
        for value in values:
            folded = value.casefold()
            for word in _SENSITIVE_WORDS:
                assert re.search(rf"\b{word}\b", folded) is None, (
                    f"{category} value {value!r} must not contain {word!r}"
                )
