import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opendistillation.dataset import validate_dataset
from opendistillation.teacher import MockTeacherEngine, TeacherRequest, build_teacher_prompt, generate_mock_qa_pairs
from opendistillation.text import TextChunk


class MockTeacherTests(unittest.TestCase):
    def test_build_teacher_prompt_includes_chunk_id_and_local_schema(self):
        chunk = TextChunk(
            id="chunk-0001",
            index=0,
            text="OpenDistillation creates question-answer examples from user notes.",
            char_count=66,
            word_count=7,
        )

        prompt = build_teacher_prompt(chunk, examples_per_chunk=2)

        self.assertIn("chunk-0001", prompt)
        self.assertIn("instruction", prompt)
        self.assertIn("response", prompt)
        self.assertIn("source_chunk_id", prompt)
        self.assertIn("2 question-answer pairs", prompt)

    def test_generate_mock_qa_pairs_is_deterministic_and_valid(self):
        chunks = [
            TextChunk(
                id="chunk-0001",
                index=0,
                text="OpenDistillation turns notes into small local model training examples.",
                char_count=74,
                word_count=9,
            ),
            TextChunk(
                id="chunk-0002",
                index=1,
                text="The v0 notebook uses a safe mock teacher before real model calls.",
                char_count=69,
                word_count=12,
            ),
        ]

        first = generate_mock_qa_pairs(chunks, examples_per_chunk=2)
        second = generate_mock_qa_pairs(chunks, examples_per_chunk=2)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual({row["source_chunk_id"] for row in first}, {"chunk-0001", "chunk-0002"})
        validate_dataset(first)

    def test_mock_teacher_engine_exposes_future_engine_interface_metadata(self):
        chunk = TextChunk(
            id="chunk-0001",
            index=0,
            text="Future real teacher engines should fit behind the same generate interface.",
            char_count=74,
            word_count=11,
        )
        engine = MockTeacherEngine()
        request = TeacherRequest(chunks=[chunk], examples_per_chunk=1)

        rows = engine.generate(request)

        self.assertEqual(engine.name, "mock-local-teacher")
        self.assertFalse(engine.sends_data_remote)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_chunk_id"], "chunk-0001")
        validate_dataset(rows)


if __name__ == "__main__":
    unittest.main()
