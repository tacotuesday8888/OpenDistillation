# GitHub Issue Plan

This file is the starter backlog to create after the repository has a GitHub remote. It exists because this local repo does not have a remote configured yet.

## Labels To Create

- `documentation` - docs, README, examples, and contributor-facing text.
- `prototype` - concrete first-demo implementation work.
- `research` - bounded decisions that require checking current model or package docs.
- `scope` - guardrails and product decisions.
- `triage` - newly filed issues that need review.
- `good first issue` - small tasks suitable for new contributors.

## Milestones

### v0.1 Colab Skeleton

Goal: a notebook that opens in Colab, accepts `.txt` or `.md`, previews text, and shows the planned flow without real training.

### v0.2 Dataset Generation

Goal: uploaded text becomes validated question-answer JSONL examples.

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

### 2. Create the v0 Colab notebook skeleton

Labels: `prototype`

Milestone: v0.1 Colab Skeleton

Acceptance criteria:

- `notebooks/opendistillation_v0_demo.ipynb` exists.
- Notebook opens in Colab.
- Notebook runs top to bottom on CPU without real training.
- Notebook clearly labels planned-but-unimplemented cells.

### 3. Implement TXT/MD upload validation

Labels: `prototype`, `good first issue`

Milestone: v0.1 Colab Skeleton

Acceptance criteria:

- `.txt` and `.md` are accepted.
- Unsupported extensions are rejected with a readable message.
- Empty files are rejected.
- Uploaded content preview shows file name, character count, word count, and first 1,000 characters.

### 4. Implement simple document chunking

Labels: `prototype`

Milestone: v0.2 Dataset Generation

Acceptance criteria:

- Chunks preserve source order.
- Empty chunks are dropped.
- Chunk IDs are stable.
- The first 3 chunks are previewed in the notebook.

### 5. Define the JSONL dataset schema

Labels: `prototype`, `documentation`

Milestone: v0.2 Dataset Generation

Acceptance criteria:

- `docs/dataset-schema.md` documents required fields.
- Each row includes `instruction`, `response`, and `source_chunk_id`.
- Dataset validation fails clearly on missing fields.
- README links to the schema after it exists.

### 6. Choose the default teacher path

Labels: `research`, `scope`

Milestone: v0.2 Dataset Generation

Acceptance criteria:

- Current official docs for the chosen teacher path are checked.
- Choice is recorded in `docs/current-decisions.md`.
- Notebook states whether uploaded text is sent to a remote endpoint.
- Fallback behavior is documented for teacher failures.

### 7. Generate question-answer examples from chunks

Labels: `prototype`

Milestone: v0.2 Dataset Generation

Acceptance criteria:

- At least one valid dataset row can be generated from `examples/sample-notes.md`.
- The notebook previews the first 5 examples.
- Dataset is saved to an ignored generated-data path.
- Dataset can be downloaded from Colab.

### 8. Choose the default student model and training backend

Labels: `research`, `scope`

Milestone: v0.3 Short Training Demo

Acceptance criteria:

- Current official model and library docs are checked.
- Student model target remains around 0.5B-1.5B parameters.
- Choice is recorded in `docs/current-decisions.md`.
- Hardware expectations are documented in the notebook.

### 9. Add the short fine-tuning path

Labels: `prototype`

Milestone: v0.3 Short Training Demo

Acceptance criteria:

- Training starts from the generated JSONL dataset.
- Default run is intentionally short.
- Output artifacts stay in ignored runtime paths.
- Missing GPU and out-of-memory failures are explained plainly.

### 10. Add before/after comparison

Labels: `prototype`

Milestone: v0.3 Short Training Demo

Acceptance criteria:

- Notebook asks the base model and trained model the same question.
- Both answers are shown side by side.
- The comparison is labeled as a sanity check, not a benchmark.

### 11. Verify GGUF or local-run export path

Labels: `prototype`, `research`

Milestone: v0.4 Local Run Path

Acceptance criteria:

- GGUF export is either tested or explicitly deferred.
- Local run instructions are documented in `docs/local-run.md`.
- README does not claim local running works until verified.

### 12. Public launch pass

Labels: `documentation`, `scope`

Milestone: Public Launch

Acceptance criteria:

- README explains the project in under one minute.
- Launch checklist is complete.
- No generated datasets, model artifacts, checkpoints, `.env` files, or local machine config are committed.
- Repository license metadata shows Apache-2.0 before public announcement.
