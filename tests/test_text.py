import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opendistillation.text import TextValidationError, chunk_text, load_text_document


class TextLoadingTests(unittest.TestCase):
    def test_load_text_document_accepts_markdown_bytes_and_builds_preview(self):
        content = b"# Notes\n\nOpenDistillation turns notes into examples.\n"

        document = load_text_document("sample-notes.md", content, min_words=20)

        self.assertEqual(document.filename, "sample-notes.md")
        self.assertEqual(document.extension, ".md")
        self.assertIn("OpenDistillation turns notes", document.preview)
        self.assertEqual(document.char_count, len(document.text))
        self.assertGreater(document.word_count, 0)
        self.assertEqual(document.warnings, ["Document is short; the demo may generate only a few examples."])

    def test_load_text_document_rejects_unsupported_file_types(self):
        with self.assertRaisesRegex(TextValidationError, "Only .txt and .md files are supported"):
            load_text_document("notes.pdf", b"not supported")

    def test_load_text_document_rejects_empty_text(self):
        with self.assertRaisesRegex(TextValidationError, "Uploaded file is empty"):
            load_text_document("empty.txt", "   \n\n ")

    def test_chunk_text_preserves_order_and_stable_ids(self):
        text = "\n\n".join(
            [
                "First paragraph introduces OpenDistillation and the demo goal.",
                "Second paragraph explains that teacher examples are generated locally in the skeleton.",
                "Third paragraph points toward later training without doing it yet.",
            ]
        )

        chunks = chunk_text(text, max_chars=95)

        self.assertEqual([chunk.id for chunk in chunks], ["chunk-0001", "chunk-0002", "chunk-0003"])
        self.assertEqual([chunk.index for chunk in chunks], [0, 1, 2])
        self.assertIn("First paragraph", chunks[0].text)
        self.assertIn("Second paragraph", chunks[1].text)
        self.assertIn("Third paragraph", chunks[2].text)
        self.assertTrue(all(chunk.char_count == len(chunk.text) for chunk in chunks))

    def test_chunk_text_splits_long_paragraph_by_words(self):
        text = " ".join(f"word{i}" for i in range(40))

        chunks = chunk_text(text, max_chars=80)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.char_count <= 80 for chunk in chunks))
        self.assertEqual(" ".join(chunk.text for chunk in chunks), text)


if __name__ == "__main__":
    unittest.main()
