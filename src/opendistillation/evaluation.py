"""Held-out comparison rows for the v0 sample-notes learning signal."""

from __future__ import annotations

from pathlib import Path

from .dataset import validate_dataset


SAMPLE_NOTES_FILENAME = "sample-notes.md"
SAMPLE_FACT_MARKERS: tuple[str, ...] = (
    "Project codename: Glass Harbor",
    "Notebook signal phrase: copper-lantern-47",
    "Local runner label: llama-harbor-alpha",
    "Review ritual time: 4:17 PM",
    "Review ritual color: ultramarine",
)

SAMPLE_FACT_COMPARISON_ROWS: tuple[dict[str, str], ...] = (
    {
        "instruction": "In the demo notes, what phrase is listed as the project codename?",
        "response": "The project codename listed in the demo notes is Glass Harbor.",
        "source_chunk_id": "chunk-0001",
    },
    {
        "instruction": "Which checkpoint phrase should verify that the sample notes were remembered?",
        "response": "The checkpoint phrase for the sample notes is copper-lantern-47.",
        "source_chunk_id": "chunk-0002",
    },
    {
        "instruction": "Which local runner label does the sample say to remember?",
        "response": "The local runner label to remember is llama-harbor-alpha.",
        "source_chunk_id": "chunk-0003",
    },
    {
        "instruction": "What time and color are paired in the review ritual notes?",
        "response": "The review ritual pairs 4:17 PM with the color ultramarine.",
        "source_chunk_id": "chunk-0004",
    },
)


def build_sample_fact_comparison_rows(filename: str, *, text: str | None = None) -> list[dict[str, str]]:
    """Return held-out comparison rows for the committed sample notes only."""

    if Path(filename).name != SAMPLE_NOTES_FILENAME:
        return []
    if text is None or any(marker not in text for marker in SAMPLE_FACT_MARKERS):
        return []
    return validate_dataset(SAMPLE_FACT_COMPARISON_ROWS)


__all__ = [
    "SAMPLE_FACT_COMPARISON_ROWS",
    "SAMPLE_FACT_MARKERS",
    "SAMPLE_NOTES_FILENAME",
    "build_sample_fact_comparison_rows",
]
