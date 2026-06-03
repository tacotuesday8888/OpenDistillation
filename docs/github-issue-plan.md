# GitHub Issue Plan

This file is the starter backlog to create or sync on GitHub after pushing local changes.

## Labels To Create

- `documentation` - docs, README, examples, and contributor-facing text.
- `prototype` - concrete first-demo implementation work.
- `research` - bounded decisions that require checking current model or package docs.
- `scope` - guardrails and product decisions.
- `triage` - newly filed issues that need review.
- `good first issue` - small tasks suitable for new contributors.

## Milestones

### v0.1 Notes-Model Colab Prototype

Goal: a notebook that opens in Colab, accepts `.txt` or `.md` notes, previews text, and shows the planned notes-model flow with training skipped by default.

### v0.2 Notes Dataset Generation

Goal: uploaded notes become validated question-answer JSONL examples through the safe mock teacher by default and one opt-in local real teacher.

### v0.3 Short Training Demo

Goal: one small student model can run a short fine-tuning job from the generated dataset.

### v0.4 Local Run Path

Goal: the trained output can be saved and either exported to GGUF or accompanied by an exact verified export path.

### Public Launch

Goal: README, notebook, license metadata, and docs are ready for public attention.

## Starter Issues

### 1. Create GitHub labels and milestones

Labels: `scope`, `documentation`

Milestone: Public Launch

Acceptance criteria:

- Labels in this file exist on GitHub.
- Milestones in this file exist on GitHub.
- Issue templates show the intended labels once issues are created.

### 2. Verify the v0 prototype from a fresh Colab runtime

Labels: `prototype`

Milestone: v0.1 Colab Skeleton

Acceptance criteria:

- `notebooks/opendistillation_v0_demo.ipynb` opens from GitHub in Colab after the remote exists.
- Notebook runs top to bottom on CPU with training skipped.
- The sample notes path works.
- The upload path works with one `.txt` and one `.md` file.
- Notebook clearly labels optional training and export status.

### 3. Harden TXT/MD upload validation messages

Labels: `prototype`, `good first issue`

Milestone: v0.1 Colab Skeleton

Acceptance criteria:

- `.txt` and `.md` are accepted.
- Unsupported extensions are rejected with a readable message.
- Empty files are rejected.
- Uploaded content preview shows file name, character count, word count, and first 1,000 characters.

### 4. Review simple notes chunking defaults

Labels: `prototype`

Milestone: v0.2 Dataset Generation

Acceptance criteria:

- Chunks preserve source order.
- Empty chunks are dropped.
- Chunk IDs are stable.
- The first 3 chunks are previewed in the notebook.
- `examples/sample-notes.md` produces multiple notes chunks with the notebook default.

### 5. Expand dataset schema documentation after real teacher selection

Labels: `prototype`, `documentation`

Milestone: v0.2 Dataset Generation

Acceptance criteria:

- `docs/dataset-schema.md` documents required fields.
- Each row includes `instruction`, `response`, and `source_chunk_id`.
- Dataset validation fails clearly on missing fields.
- Any new optional fields are documented before they are used.

Status: implemented for the current v0 schema. Both `MockTeacherEngine` and the optional `HuggingFaceLocalTeacherEngine` must return `instruction`, `response`, and `source_chunk_id`.

### 6. Choose the default notes teacher path

Labels: `research`, `scope`

Milestone: v0.2 Dataset Generation

Acceptance criteria:

- Current official docs for the chosen teacher path are checked.
- Choice is recorded in `docs/current-decisions.md`.
- Notebook states whether uploaded notes are sent to a remote endpoint.
- Fallback behavior is documented for teacher failures.

Status: implemented and smoke-tested once on Colab T4. The default remains `MockTeacherEngine`. The opt-in real teacher is `HuggingFaceLocalTeacherEngine` using `Qwen/Qwen2.5-1.5B-Instruct`; it downloads model weights from Hugging Face but does not send notes text to a paid or proprietary remote API.

### 7. Generate question-answer examples from chunks

Labels: `prototype`

Milestone: v0.2 Dataset Generation

Acceptance criteria:

- At least one valid dataset row can be generated from `examples/sample-notes.md`.
- The notebook previews the first 5 examples.
- Dataset is saved to an ignored generated-data path.
- Dataset can be downloaded from Colab.

Status: implemented for the default mock teacher path and smoke-tested once for the optional real teacher path on Colab T4 with one sample-note chunk and one generated row. More output-quality hardening remains.

### 8. Choose the default notes student model and training backend

Labels: `research`, `scope`

Milestone: v0.3 Short Training Demo

Acceptance criteria:

- Current official model and library docs are checked.
- Student model target remains around 0.5B-1.5B parameters.
- Choice is recorded in `docs/current-decisions.md`.
- Hardware expectations are documented in the notebook.

Status: implemented locally with `Qwen/Qwen2.5-0.5B-Instruct`, TRL `SFTTrainer`, and PEFT LoRA.

### 9. Add the short fine-tuning path

Labels: `prototype`

Milestone: v0.3 Short Training Demo

Acceptance criteria:

- Training can start from the generated JSONL dataset after `RUN_TRAINING = True`.
- Default run is intentionally short.
- Output artifacts stay in ignored runtime paths.
- Missing GPU and out-of-memory failures are explained plainly.

Status: bounded local engine and notebook entry point exist. A clean GitHub-opened Colab T4 runtime completed the optional training path and created `/content/OpenDistillation/outputs/notes-lora/adapter`. A later real-teacher T4 verification confirmed a 1-step adapter at `/content/OpenDistillation/outputs/notes-lora-real-teacher-smoke/adapter`. Remaining work is quality and demo hardening, not broadening the training system.

### 10. Add before/after comparison

Labels: `prototype`

Milestone: v0.3 Short Training Demo

Acceptance criteria:

- Notebook asks the base model and trained model the same question.
- Both answers are shown side by side.
- The comparison is labeled as a sanity check, not a benchmark.

Status: implemented locally as an optional notebook section using the first generated dataset question. A clean GitHub-opened Colab T4 runtime printed one real base-vs-adapter comparison. The output is a qualitative sanity check, not a benchmark.

### 11. Run manual Colab GPU smoke test for optional training and comparison

Labels: `prototype`

Milestone: v0.3 Short Training Demo

Acceptance criteria:

- `docs/colab-smoke-test-checklist.md` is completed from a fresh Colab GPU runtime.
- Optional Hugging Face package install result is recorded.
- Runtime check prints a GPU name.
- `Qwen/Qwen2.5-0.5B-Instruct` download status is recorded.
- Short training starts and either creates `outputs/notes-lora/adapter` or records the exact failure.
- Before/after comparison either prints base and adapter answers or records the exact failure.
- Docs are updated to mark the verified and unverified parts honestly.

Status: completed on 2026-06-02 from a clean GitHub-opened Colab T4 runtime. Results are recorded in `docs/colab-smoke-test-results.md`. The optional real teacher path also passed one T4 end-to-end wiring check on 2026-06-03. GGUF export and local runtime instructions remain separate follow-up work.

### 12. Verify GGUF or local-run export path

Labels: `prototype`, `research`

Milestone: v0.4 Local Run Path

Acceptance criteria:

- GGUF export is either tested or explicitly deferred.
- Local run instructions are documented in `docs/local-run.md`.
- README does not claim local running works until verified.

### 13. Public launch pass

Labels: `documentation`, `scope`

Milestone: Public Launch

Acceptance criteria:

- README explains the project in under one minute.
- Launch checklist is complete.
- No generated datasets, model artifacts, checkpoints, `.env` files, or local machine config are committed.
- Repository license metadata shows Apache-2.0 before public announcement.
