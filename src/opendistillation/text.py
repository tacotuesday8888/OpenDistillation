"""Plain-text loading and chunking helpers for the v0 notebook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


SUPPORTED_EXTENSIONS = {".txt", ".md"}
SHORT_DOCUMENT_WARNING = "Document is short; the demo may generate only a few examples."


class TextValidationError(ValueError):
    """Raised when a user-provided text file cannot be used by the demo."""


@dataclass(frozen=True)
class LoadedTextDocument:
    filename: str
    extension: str
    text: str
    char_count: int
    word_count: int
    preview: str
    warnings: list[str]


@dataclass(frozen=True)
class TextChunk:
    id: str
    index: int
    text: str
    char_count: int
    word_count: int


def load_text_document(
    filename: str,
    content: bytes | str,
    *,
    preview_chars: int = 1000,
    min_words: int = 80,
) -> LoadedTextDocument:
    """Validate and normalize one uploaded TXT/MD document."""

    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise TextValidationError("Only .txt and .md files are supported in the v0 prototype.")

    raw_text = _decode_content(content)
    text = normalize_text(raw_text)
    if not text:
        raise TextValidationError("Uploaded file is empty after whitespace cleanup.")

    word_count = count_words(text)
    warnings = []
    if word_count < min_words:
        warnings.append(SHORT_DOCUMENT_WARNING)

    return LoadedTextDocument(
        filename=Path(filename).name,
        extension=extension,
        text=text,
        char_count=len(text),
        word_count=word_count,
        preview=text[:preview_chars],
        warnings=warnings,
    )


def chunk_text(text: str, *, max_chars: int = 700) -> list[TextChunk]:
    """Split text into ordered chunks, preferring paragraph boundaries."""

    normalized = normalize_text(text)
    if not normalized:
        raise TextValidationError("Cannot chunk empty text.")
    if max_chars < 40:
        raise ValueError("max_chars must be at least 40 so chunks remain readable.")

    paragraph_chunks: list[str] = []
    current = ""
    for paragraph in _paragraphs(normalized):
        for piece in _split_paragraph(paragraph, max_chars=max_chars):
            candidate = piece if not current else f"{current}\n\n{piece}"
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    paragraph_chunks.append(current)
                current = piece
    if current:
        paragraph_chunks.append(current)

    return [
        TextChunk(
            id=f"chunk-{index + 1:04d}",
            index=index,
            text=chunk,
            char_count=len(chunk),
            word_count=count_words(chunk),
        )
        for index, chunk in enumerate(paragraph_chunks)
    ]


def normalize_text(text: str) -> str:
    """Normalize newlines and trailing whitespace without flattening Markdown."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    cleaned = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", cleaned)


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def _decode_content(content: bytes | str) -> str:
    if isinstance(content, bytes):
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TextValidationError("Uploaded file must be UTF-8 text.") from exc
    return str(content)


def _paragraphs(text: str) -> list[str]:
    return [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]


def _split_paragraph(paragraph: str, *, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]

    chunks: list[str] = []
    current_words: list[str] = []
    for word in paragraph.split():
        candidate_words = [*current_words, word]
        candidate = " ".join(candidate_words)
        if len(candidate) <= max_chars:
            current_words = candidate_words
            continue

        if current_words:
            chunks.append(" ".join(current_words))
            current_words = []

        if len(word) > max_chars:
            chunks.extend(word[start : start + max_chars] for start in range(0, len(word), max_chars))
        else:
            current_words = [word]

    if current_words:
        chunks.append(" ".join(current_words))
    return chunks
