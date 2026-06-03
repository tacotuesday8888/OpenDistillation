# Note-Grounded Learning Signal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one bounded v0 experiment that tests whether a tiny LoRA adapter can learn concrete unusual facts from notes.

**Architecture:** Keep TXT/MD notes, MockTeacherEngine, TRL SFTTrainer, PEFT LoRA, and the Colab-first notebook. Strengthen the sample notes and mock rows, then add held-out comparison rows that ask about the same facts with different wording from the training rows.

**Tech Stack:** Python standard library helpers, unittest, Colab, Hugging Face Transformers, TRL SFTTrainer, PEFT LoRA, Qwen2.5-0.5B-Instruct.

---

### Task 1: Fact-Rich Sample Notes

**Files:**
- Modify: `examples/sample-notes.md`
- Test: `tests/test_evaluation.py`

- [ ] Replace generic sample notes with four short fact cards. Each chunk must include concrete unlikely facts such as `Glass Harbor`, `copper-lantern-47`, `4:17 PM`, `ultramarine`, and `llama-harbor-alpha`.
- [ ] Add a test that loads `examples/sample-notes.md`, chunks it at `max_chars=300`, and proves it produces four chunks with those facts present.

### Task 2: Held-Out Comparison Rows

**Files:**
- Create: `src/opendistillation/evaluation.py`
- Modify: `src/opendistillation/__init__.py`
- Test: `tests/test_evaluation.py`

- [ ] Add `build_sample_fact_comparison_rows(filename)` that returns four fixed v0-schema rows for `sample-notes.md` and an empty list for uploaded/user files.
- [ ] Make the held-out questions ask about sample-note facts with wording different from the mock teacher training rows.
- [ ] Export the helper from `opendistillation`.

### Task 3: Clearer Mock Rows

**Files:**
- Modify: `src/opendistillation/teacher.py`
- Test: `tests/test_teacher.py`

- [ ] Add tests showing that colon-style facts in a note chunk generate direct grounded QA rows.
- [ ] Implement lightweight fact extraction for `Label: value` lines without changing the JSONL schema.
- [ ] Keep the current generic excerpt fallback for normal uploaded notes.

### Task 4: Notebook Experiment Wiring

**Files:**
- Modify: `notebooks/opendistillation_v0_demo.ipynb`
- Test: `jq empty notebooks/opendistillation_v0_demo.ipynb`

- [ ] Import `build_sample_fact_comparison_rows`.
- [ ] Use six mock examples per chunk in the notebook sample path.
- [ ] Use `SFTLoRAConfig(max_steps=30)` in the optional notebook training plan while keeping `RUN_TRAINING = False`.
- [ ] Print held-out comparison questions when the sample file is used.
- [ ] Use held-out rows for comparison after training; keep generated training rows for uploaded notes.

### Task 5: Evidence, Docs, Verification

**Files:**
- Modify docs named by the goal after implementation and after any Colab run.

- [ ] Run unit tests, notebook JSON validation, default local notebook smoke, `git diff --check`, secret scan, artifact scan, and git status.
- [ ] If feasible, run a bounded Colab T4 smoke from pushed main and record package versions, runtime, dataset quality, training steps, adapter path, held-out questions, base/trained answers, overlap deltas, and better/unchanged/worse judgment.
- [ ] Commit and push only source, tests, notebook, docs, and sample-note changes. Do not commit generated datasets, adapters, checkpoints, model files, secrets, or local config.
