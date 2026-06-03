# First Quality Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first honest quality loop for the notes-model Colab demo without broadening v0 beyond TXT/MD notes.

**Architecture:** Keep the current open-source stack: PyTorch, Hugging Face Transformers, TRL `SFTTrainer`, PEFT LoRA, and Qwen models. Add deterministic local dataset-quality helpers that require no model downloads, improve teacher row variety through prompt/templates, and extend the existing comparison helper to handle a bounded list of generated questions. The notebook should show dataset quality separately from model quality.

**Tech Stack:** Python standard library, unittest, existing notebook JSON, Hugging Face Transformers/TRL/PEFT docs for API alignment. No new runtime dependency.

---

### Task 1: Dataset Quality Helpers

**Files:**
- Create: `src/opendistillation/quality.py`
- Modify: `src/opendistillation/__init__.py`
- Test: `tests/test_quality.py`

- [ ] **Step 1: Write failing tests**

Create tests that call `analyze_dataset_quality(rows, expected_chunk_ids=[...])` and assert it reports row count, chunk coverage, missing fields, exact duplicate questions, near-duplicate questions, short answers, extra source chunk IDs, and beginner-readable lines from `format_dataset_quality_report(report)`.

- [ ] **Step 2: Run tests to verify RED**

Run: `python3 -m unittest tests.test_quality`

Expected: import failure for `opendistillation.quality` or missing helper functions.

- [ ] **Step 3: Implement minimal helper**

Create `DatasetQualityReport`, `DatasetQualityIssue`, `analyze_dataset_quality`, and `format_dataset_quality_report`. Use only standard-library code: field checks, word counts, normalized question strings, and `difflib.SequenceMatcher` for near-duplicate detection.

- [ ] **Step 4: Run tests to verify GREEN**

Run: `python3 -m unittest tests.test_quality`

Expected: all quality tests pass.

### Task 2: Better Teacher Row Variety

**Files:**
- Modify: `src/opendistillation/teacher.py`
- Test: `tests/test_teacher.py`

- [ ] **Step 1: Write failing tests**

Add tests that verify `build_teacher_prompt()` asks for factual recall, explanation, flashcard, and misconception-check examples while still requiring exactly `instruction`, `response`, and `source_chunk_id`. Add a mock-teacher test with `examples_per_chunk=4` that verifies four varied question styles for one chunk.

- [ ] **Step 2: Run tests to verify RED**

Run: `python3 -m unittest tests.test_teacher`

Expected: new style assertions fail against the old two-template prompt/mock rows.

- [ ] **Step 3: Implement teacher prompt/template update**

Add a small ordered question-style list. Update the real-teacher prompt to require grounded JSONL rows across those styles. Update the mock generator to cycle through the same styles deterministically without adding fields to the public schema.

- [ ] **Step 4: Run tests to verify GREEN**

Run: `python3 -m unittest tests.test_teacher`

Expected: all teacher tests pass.

### Task 3: Multi-Question Comparison Quality Report

**Files:**
- Modify: `src/opendistillation/comparison.py`
- Test: `tests/test_comparison.py`

- [ ] **Step 1: Write failing tests**

Add tests that call `build_comparison_request(rows, training_result, max_examples=3)` and assert it selects up to three validated examples. Add fake-dependency tests that verify `BeforeAfterComparisonEngine.compare()` returns per-question base/trained answers and deterministic reference-overlap scores.

- [ ] **Step 2: Run tests to verify RED**

Run: `python3 -m unittest tests.test_comparison`

Expected: `max_examples` and per-item result assertions fail.

- [ ] **Step 3: Implement comparison update**

Add small `ComparisonExample` and `BeforeAfterComparisonItem` dataclasses. Keep compatibility properties for `question`, `reference_response`, `base_answer`, `trained_answer`, and `source_chunk_id` so existing notebook code remains readable. Load the base model and adapter once, then loop over bounded questions.

- [ ] **Step 4: Run tests to verify GREEN**

Run: `python3 -m unittest tests.test_comparison`

Expected: all comparison tests pass.

### Task 4: Notebook Quality Loop

**Files:**
- Modify: `notebooks/opendistillation_v0_demo.ipynb`
- Test: `tests/test_notebook.py`

- [ ] **Step 1: Update imports and cells**

Import `analyze_dataset_quality` and `format_dataset_quality_report`. Set the notebook teacher request to four examples per chunk. In the dataset preview cell, print a beginner-readable dataset quality report and record an `OD_STATUS` marker. In the comparison cell, request up to three examples and print a model quality report that labels overlap as a crude smoke signal.

- [ ] **Step 2: Validate notebook JSON and default path**

Run: `jq empty notebooks/opendistillation_v0_demo.ipynb`

Run the local default notebook smoke path from the existing plan command.

Expected: JSON is valid and default path completes with training/comparison skipped while still printing dataset quality.

### Task 5: Docs

**Files:**
- Modify: `README.md`
- Modify: `notebooks/README.md`
- Modify: `docs/current-decisions.md`
- Modify: `docs/first-demo-flow.md`
- Modify: `docs/dataset-schema.md`
- Modify: `docs/engine-integration-points.md`
- Modify: `docs/colab-smoke-test-results.md`
- Modify: `docs/next-goal-prompt.md`

- [ ] **Step 1: Update public status**

State that the first quality loop is deterministic locally for dataset quality and bounded for model comparison, but real model-quality evidence still needs a Colab quality smoke run.

- [ ] **Step 2: Document dependency decision**

Record that Hugging Face Evaluate/LightEval were considered and not added because the first loop does not yet have a stable held-out benchmark. Record that Unsloth remains a later speed/memory optimization after quality is measurable.

### Task 6: Final Verification And Publish

**Files:**
- Review all changed files.

- [ ] **Step 1: Run required checks**

Run: `python3 -m unittest discover -s tests`
Run: `jq empty notebooks/opendistillation_v0_demo.ipynb`
Run: `jq '[.cells[] | select(.outputs and (.outputs | length > 0))] | length' notebooks/opendistillation_v0_demo.ipynb`
Run: `git diff --check`
Run the secret scan from AGENTS.md practice.
Run the generated model/data artifact scan.

- [ ] **Step 2: Review, commit, push**

Review `git diff`, stage only intended files, commit, push `origin main`, and confirm `git status --short --branch --untracked-files=all` is clean.
