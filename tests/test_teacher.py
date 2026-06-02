import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opendistillation.dataset import validate_dataset
from opendistillation.teacher import (
    DEFAULT_REAL_TEACHER_MODEL,
    HuggingFaceLocalTeacherEngine,
    MockTeacherEngine,
    RealTeacherDependencyError,
    RealTeacherGenerationError,
    RealTeacherModelLoadError,
    RealTeacherOutputError,
    TeacherRequest,
    build_teacher_prompt,
    explain_teacher_failure,
    generate_mock_qa_pairs,
    parse_teacher_jsonl_output,
)
from opendistillation.text import TextChunk


class FakePipeline:
    def __init__(self, generated_text):
        self.generated_text = generated_text
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return [{"generated_text": [{"role": "assistant", "content": self.generated_text}]}]


class FakePipelineFactory:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.pipeline


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
            text="Real teacher engines fit behind the same generate interface.",
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


class HuggingFaceLocalTeacherTests(unittest.TestCase):
    def test_real_teacher_engine_generates_valid_rows_with_fake_pipeline(self):
        chunk = TextChunk(
            id="chunk-0001",
            index=0,
            text="Photosynthesis turns light energy into chemical energy stored as sugar.",
            char_count=72,
            word_count=10,
        )
        generated = (
            '{"instruction":"What does photosynthesis turn light into?",'
            '"response":"Photosynthesis turns light energy into chemical energy stored as sugar.",'
            '"source_chunk_id":"chunk-0001"}'
        )
        fake_pipeline = FakePipeline(generated)
        fake_factory = FakePipelineFactory(fake_pipeline)
        engine = HuggingFaceLocalTeacherEngine(pipeline_factory=fake_factory)

        rows = engine.generate(TeacherRequest(chunks=[chunk], examples_per_chunk=1))

        self.assertEqual(engine.model_name, DEFAULT_REAL_TEACHER_MODEL)
        self.assertEqual(engine.name, "huggingface-local-teacher")
        self.assertFalse(engine.sends_data_remote)
        self.assertEqual(rows[0]["source_chunk_id"], "chunk-0001")
        self.assertIn("photosynthesis", rows[0]["instruction"].lower())
        validate_dataset(rows)
        self.assertEqual(fake_factory.calls[0]["model"], DEFAULT_REAL_TEACHER_MODEL)
        self.assertEqual(fake_factory.calls[0]["task"], "text-generation")
        self.assertEqual(fake_factory.calls[0]["dtype"], "auto")
        self.assertEqual(fake_factory.calls[0]["device_map"], "auto")
        self.assertFalse(fake_pipeline.calls[0]["do_sample"])
        self.assertEqual(fake_pipeline.calls[0]["max_new_tokens"], 512)
        self.assertIn("Return only JSONL", fake_pipeline.calls[0]["messages"][0]["content"])

    def test_parse_teacher_jsonl_output_accepts_fenced_jsonl(self):
        generated = """```jsonl
{"instruction":"What is remembered?","response":"The notes say to keep v0 narrow.","source_chunk_id":"chunk-0001"}
```"""

        rows = parse_teacher_jsonl_output(generated, expected_chunk_id="chunk-0001")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["instruction"], "What is remembered?")
        validate_dataset(rows)

    def test_parse_teacher_jsonl_output_rejects_wrong_chunk_id(self):
        generated = (
            '{"instruction":"What is remembered?",'
            '"response":"The notes say to keep v0 narrow.",'
            '"source_chunk_id":"chunk-9999"}'
        )

        with self.assertRaises(RealTeacherOutputError):
            parse_teacher_jsonl_output(generated, expected_chunk_id="chunk-0001")

    def test_real_teacher_engine_wraps_model_load_failure(self):
        def failing_factory(**kwargs):
            raise OSError("model files were not found")

        engine = HuggingFaceLocalTeacherEngine(pipeline_factory=failing_factory)

        with self.assertRaises(RealTeacherModelLoadError) as context:
            engine.generate(TeacherRequest(chunks=[_teacher_chunk()], examples_per_chunk=1))

        lines = explain_teacher_failure(context.exception)
        self.assertIn("The real teacher model could not be downloaded or loaded.", lines)

    def test_real_teacher_engine_preserves_dependency_failure_from_factory(self):
        def failing_factory(**kwargs):
            raise RealTeacherDependencyError("Transformers is not importable")

        engine = HuggingFaceLocalTeacherEngine(pipeline_factory=failing_factory)

        with self.assertRaises(RealTeacherDependencyError) as context:
            engine.generate(TeacherRequest(chunks=[_teacher_chunk()], examples_per_chunk=1))

        lines = explain_teacher_failure(context.exception)
        self.assertIn("A required optional Hugging Face package is missing or not importable.", lines)

    def test_real_teacher_engine_wraps_generation_memory_failure(self):
        class FailingPipeline:
            def __call__(self, messages, **kwargs):
                raise RuntimeError("CUDA out of memory")

        engine = HuggingFaceLocalTeacherEngine(pipeline_factory=lambda **kwargs: FailingPipeline())

        with self.assertRaises(RealTeacherGenerationError) as context:
            engine.generate(TeacherRequest(chunks=[_teacher_chunk()], examples_per_chunk=1))

        lines = explain_teacher_failure(context.exception)
        self.assertIn("The GPU ran out of memory while running the real teacher.", lines)

    def test_real_teacher_engine_wraps_bad_generated_rows(self):
        engine = HuggingFaceLocalTeacherEngine(
            pipeline_factory=lambda **kwargs: FakePipeline("not jsonl")
        )

        with self.assertRaises(RealTeacherOutputError) as context:
            engine.generate(TeacherRequest(chunks=[_teacher_chunk()], examples_per_chunk=1))

        lines = explain_teacher_failure(context.exception)
        self.assertIn("The real teacher output did not match the v0 JSONL schema.", lines)

    def test_real_teacher_dependency_error_explains_install_step(self):
        error = RealTeacherDependencyError("Transformers is not importable")

        lines = explain_teacher_failure(error)

        self.assertIn("A required optional Hugging Face package is missing or not importable.", lines)
        self.assertTrue(any("python -m pip install" in line for line in lines))


def _teacher_chunk():
    return TextChunk(
        id="chunk-0001",
        index=0,
        text="OpenDistillation should create useful QA rows from notes.",
        char_count=58,
        word_count=8,
    )


if __name__ == "__main__":
    unittest.main()
