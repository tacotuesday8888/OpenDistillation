# Real Teacher Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one opt-in local Hugging Face teacher engine that can generate v0 QA rows from TXT/MD notes while keeping `MockTeacherEngine` as the safe default.

**Architecture:** Extend `src/opendistillation/teacher.py` behind the existing `TeacherEngine.generate()` protocol. The real path uses a local Transformers text-generation pipeline for `Qwen/Qwen2.5-1.5B-Instruct`, parses JSONL from the generated text, validates rows with the existing dataset schema, and wraps dependency/model/generation/schema failures in beginner-readable exceptions. The notebook adds `RUN_REAL_TEACHER = False` so the default path still uses the mock teacher and requires no model download.

**Tech Stack:** Python standard library, Hugging Face Transformers pipeline API, existing `DatasetValidationError` and `validate_dataset`, unittest with fake pipeline dependencies.

---

## Files

- Modify `src/opendistillation/teacher.py`: add default model/config, real teacher engine, JSONL parser, and teacher-failure explanations.
- Modify `src/opendistillation/__init__.py`: export real teacher symbols.
- Modify `tests/test_teacher.py`: no-download fake dependency tests for success and failure modes.
- Modify `tests/test_notebook.py`: assert notebook has `RUN_REAL_TEACHER = False`, default mock behavior, and real teacher opt-in symbols.
- Modify `notebooks/opendistillation_v0_demo.ipynb`: branch teacher generation between mock and real engine.
- Modify docs: `README.md`, `docs/current-decisions.md`, `docs/first-demo-flow.md`, `docs/engine-integration-points.md`, `docs/colab-smoke-test-checklist.md`, `docs/next-goal-prompt.md`, and stale status docs found by search.

## Hugging Face Decision Evidence

- Hugging Face model search and repo details showed `Qwen/Qwen2.5-1.5B-Instruct` is a Transformers `text-generation` chat model, Apache-2.0 licensed, 1.54B parameters, and uses `AutoModelForCausalLM`.
- Hugging Face Transformers chat docs recommend chat messages with `role`/`content`, `TextGenerationPipeline`, `dtype="auto"`, and `device_map="auto"` for local model loading.
- The selected v0 teacher path is local/open-source: model weights download from Hugging Face, but note text is not sent to a proprietary or paid API.

## Task 1: Teacher Engine Tests

- [x] **Step 1: Add failing success test**

Add to `tests/test_teacher.py` a fake pipeline test that creates `HuggingFaceLocalTeacherEngine(pipeline_factory=fake_factory)`, generates one JSONL row for a fake chunk, and asserts:

```python
self.assertEqual(engine.model_name, "Qwen/Qwen2.5-1.5B-Instruct")
self.assertFalse(engine.sends_data_remote)
self.assertEqual(rows[0]["source_chunk_id"], "chunk-0001")
validate_dataset(rows)
self.assertEqual(fake_factory.calls[0]["model"], "Qwen/Qwen2.5-1.5B-Instruct")
self.assertFalse(fake_pipeline.calls[0]["do_sample"])
```

Run:

```bash
python3 -m unittest tests.test_teacher
```

Expected: fails because `HuggingFaceLocalTeacherEngine` is not defined.

- [x] **Step 2: Add failing parser/error tests**

Add tests for:

```python
parse_teacher_jsonl_output('```jsonl\n{"instruction":"Q","response":"A","source_chunk_id":"chunk-0001"}\n```')
```

and invalid output:

```python
with self.assertRaises(RealTeacherOutputError):
    parse_teacher_jsonl_output("not json", expected_chunk_id="chunk-0001")
```

Run:

```bash
python3 -m unittest tests.test_teacher
```

Expected: fails because parser/error symbols are not defined.

## Task 2: Teacher Engine Implementation

- [x] **Step 1: Implement minimal config and exceptions**

Add to `src/opendistillation/teacher.py`:

```python
DEFAULT_REAL_TEACHER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

@dataclass(frozen=True)
class RealTeacherConfig:
    model_name: str = DEFAULT_REAL_TEACHER_MODEL
    max_new_tokens: int = 512

class RealTeacherError(RuntimeError): ...
class RealTeacherDependencyError(RealTeacherError): ...
class RealTeacherModelLoadError(RealTeacherError): ...
class RealTeacherGenerationError(RealTeacherError): ...
class RealTeacherOutputError(RealTeacherError): ...
```

- [x] **Step 2: Implement JSONL parsing and validation**

Add `parse_teacher_jsonl_output(text, expected_chunk_id)` that accepts raw JSONL or fenced JSONL, validates with `validate_dataset`, and raises `RealTeacherOutputError` when rows are empty, malformed, or use the wrong `source_chunk_id`.

- [x] **Step 3: Implement `HuggingFaceLocalTeacherEngine`**

Use an injectable `pipeline_factory`. The default factory imports `transformers.pipeline` lazily and calls:

```python
pipeline(
    task="text-generation",
    model=config.model_name,
    dtype="auto",
    device_map="auto",
)
```

Generate with chat messages and:

```python
pipeline(messages, max_new_tokens=config.max_new_tokens, do_sample=False, return_full_text=False)
```

- [x] **Step 4: Add `explain_teacher_failure(exc)`**

Return plain-language lines for missing dependencies, model download/load failures, CUDA or out-of-memory failures, and invalid generated rows.

- [x] **Step 5: Export symbols in `src/opendistillation/__init__.py`**

Export `DEFAULT_REAL_TEACHER_MODEL`, `RealTeacherConfig`, `HuggingFaceLocalTeacherEngine`, real teacher errors, `parse_teacher_jsonl_output`, and `explain_teacher_failure`.

- [x] **Step 6: Run teacher tests**

```bash
python3 -m unittest tests.test_teacher
```

Expected: pass.

## Task 3: Notebook Opt-In Path

- [x] **Step 1: Add failing notebook assertions**

In `tests/test_notebook.py`, assert the notebook contains:

```python
self.assertIn("RUN_REAL_TEACHER = False", sources)
self.assertIn("HuggingFaceLocalTeacherEngine", sources)
self.assertIn("explain_teacher_failure", sources)
self.assertIn("DEFAULT_REAL_TEACHER_MODEL", sources)
```

Run:

```bash
python3 -m unittest tests.test_notebook
```

Expected: fails until the notebook is patched.

- [x] **Step 2: Patch notebook setup imports**

Import `DEFAULT_REAL_TEACHER_MODEL`, `HuggingFaceLocalTeacherEngine`, and `explain_teacher_failure` from `opendistillation`.

- [x] **Step 3: Patch teacher generation cell**

Replace the mock-only cell with:

```python
RUN_REAL_TEACHER = False
teacher_request = TeacherRequest(chunks=chunks, examples_per_chunk=2)

if RUN_REAL_TEACHER:
    teacher_engine = HuggingFaceLocalTeacherEngine()
    print(f"Real teacher model: {DEFAULT_REAL_TEACHER_MODEL}")
    try:
        rows = teacher_engine.generate(teacher_request)
    except Exception as exc:
        print("Real teacher generation failed with a recoverable setup/runtime issue.")
        for line in explain_teacher_failure(exc):
            print(f"- {line}")
        raise
else:
    teacher_engine = MockTeacherEngine()
    rows = teacher_engine.generate(teacher_request)

dataset_jsonl = rows_to_jsonl(rows)
```

- [x] **Step 4: Run notebook test and JSON validation**

```bash
python3 -m unittest tests.test_notebook
python3 -m json.tool notebooks/opendistillation_v0_demo.ipynb
```

Expected: both pass.

## Task 4: Docs Alignment

- [x] **Step 1: Update public/docs status**

Update README and docs to state:

- Mock teacher remains default.
- `Qwen/Qwen2.5-1.5B-Instruct` is the first optional local open-source teacher path.
- `RUN_REAL_TEACHER = False` keeps default local/Colab path safe.
- Real teacher execution is implemented but not yet clean-Colab-smoke-tested unless evidence is added later.

- [x] **Step 2: Update checklist**

Add a real teacher section to `docs/colab-smoke-test-checklist.md` with dependency/model-load/schema/CUDA failure fields.

- [x] **Step 3: Update next goal**

Make `docs/next-goal-prompt.md` point to clean Colab smoke testing the real teacher path, not selecting it.

## Task 5: Verification And Release

- [x] **Step 1: Run tests**

```bash
python3 -m unittest discover -s tests
```

- [x] **Step 2: Run default notebook path smoke test**

Execute notebook code cells locally with defaults and assert the mock teacher path completes without optional model downloads.

- [x] **Step 3: Run notebook JSON validation**

```bash
python3 -m json.tool notebooks/opendistillation_v0_demo.ipynb
```

- [x] **Step 4: Run whitespace/diff check**

```bash
git diff --check
```

- [x] **Step 5: Run secret and artifact scans**

```bash
rg -n "(sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{20,}|BEGIN (RSA |OPENSSH |EC |DSA )?PRIVATE KEY|OPENAI_API_KEY|HF_TOKEN|api[_-]?key|access[_-]?token|secret)" README.md START_HERE.md docs notebooks src tests examples AGENTS.md pyproject.toml LICENSE
find . -maxdepth 4 -type f \( -name "*.gguf" -o -name "*.safetensors" -o -name "*.bin" -o -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" -o -name "*.jsonl" -o -name ".env" \) -print
```

- [x] **Step 6: Commit and push**

```bash
git status --short
git add ...
git commit -m "feat: add optional real teacher path"
git push
```

## Self-Review

- Spec coverage: the plan covers one open-source local teacher path, opt-in notebook safety, schema validation, no-download tests, docs, verification, and artifact/secrets guardrails.
- Placeholder scan: no TODO/TBD implementation placeholders remain.
- Type consistency: the planned symbols are consistently named across tests, implementation, notebook, and docs.
