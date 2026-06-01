# First Demo Implementation Plan

This plan turns the v0 Colab flow into small implementation milestones. It is intentionally narrow and does not build a SaaS, Mac app, account system, cloud backend, or broad training platform.

## Demo Target

The first demo should prove this path:

> Upload one text file, generate a small training dataset, fine-tune one small student model, compare before/after answers, and show the local run path.

The demo is successful if the user understands the whole workflow, even if the trained model is small and imperfect.

## Build Principles

- Keep the notebook readable for beginners.
- Use one default path before adding choices.
- Prefer a reliable small model over an impressive model that fails in Colab.
- Show intermediate outputs after every major step.
- Save generated datasets and model artifacts only in ignored local/runtime paths.
- Label unimplemented export or training pieces honestly.

## Milestone 1: Colab Skeleton

Purpose:

> Make the notebook runnable from top to bottom before real model training exists.

Files:

- Create `notebooks/opendistillation_v0_demo.ipynb`.
- Create `src/opendistillation/__init__.py` only if shared helpers are needed.
- Update `notebooks/README.md`.

Tasks:

1. Add a short notebook introduction with the exact v0 promise.
2. Add runtime setup cells with visible dependency placeholders.
3. Add upload cell for `.txt` and `.md`.
4. Add text preview with file name, character count, word count, and first 1,000 characters.
5. Add a static walkthrough cell showing the planned output sections.
6. Run the notebook on CPU to confirm it finishes without training.

Acceptance criteria:

- Notebook opens in Colab.
- Notebook can run without a GPU.
- Unsupported file types produce a readable message.
- No generated files are committed.

Status: implemented as `notebooks/opendistillation_v0_demo.ipynb`.

## Milestone 2: TXT/MD Loading And Chunking

Purpose:

> Turn uploaded text into ordered chunks that can be used for teacher generation.

Files:

- Add `src/opendistillation/text.py`.
- Add tests under `tests/` once a Python test setup exists.
- Update `docs/first-demo-flow.md` if behavior changes.

Tasks:

1. Implement file extension validation for `.txt` and `.md`.
2. Normalize whitespace without destroying Markdown structure.
3. Split text by paragraph first, then by character limit when needed.
4. Give each chunk a stable ID such as `chunk-0001`.
5. Show the first 3 chunks in the notebook.

Acceptance criteria:

- Empty files are rejected.
- Tiny files show a warning.
- Chunk order is stable.
- The sample notes produce multiple readable chunks.

Status: implemented in `src/opendistillation/text.py`.

## Milestone 3: Dataset Schema And Teacher Generation

Purpose:

> Generate a small, inspectable JSONL dataset from chunks.

Files:

- Add `src/opendistillation/dataset.py`.
- Add `src/opendistillation/teacher.py`.
- Add `docs/dataset-schema.md`.
- Update `notebooks/opendistillation_v0_demo.ipynb`.

Initial JSONL row:

```json
{"instruction":"Question about the user's document","response":"Answer grounded in the document","source_chunk_id":"chunk-0001"}
```

Tasks:

1. Define the dataset row fields in `docs/dataset-schema.md`.
2. Add deterministic validation for required fields.
3. Add teacher prompt construction from one chunk.
4. Choose one teacher path during implementation after checking current official model/provider docs.
5. Generate a small number of examples per chunk.
6. Save the dataset to a runtime path outside the repository.
7. Preview the first 5 examples in the notebook.

Acceptance criteria:

- Every dataset row has `instruction`, `response`, and `source_chunk_id`.
- Invalid rows are shown before training.
- Dataset output is downloadable from Colab.
- The notebook clearly says if text is sent to a remote teacher endpoint.

Status: implemented for the skeleton with `MockTeacherEngine`; real teacher-model selection remains a later milestone.

## Milestone 4: Short Student Fine-Tuning

Purpose:

> Fine-tune one small student model with the generated dataset.

Files:

- Add `src/opendistillation/train.py`.
- Update `notebooks/opendistillation_v0_demo.ipynb`.
- Update `docs/current-decisions.md` once model/backend choices are verified.

Tasks:

1. Choose one student model around 0.5B-1.5B parameters after checking current official model and library docs.
2. Choose one fine-tuning backend after a small Colab smoke test.
3. Add a training cell with small defaults.
4. Save adapter or model output to an ignored runtime path.
5. Add a beginner-readable failure message for missing GPU or out-of-memory errors.

Acceptance criteria:

- Training starts with the generated dataset.
- The default run is short enough for a demo.
- Output artifacts are not committed.
- Failure cases explain the likely next step.

## Milestone 5: Before/After Comparison

Purpose:

> Show that the trained model learned something from the uploaded document.

Files:

- Add `src/opendistillation/evaluate.py` if helper code is needed.
- Update `notebooks/opendistillation_v0_demo.ipynb`.

Tasks:

1. Pick one comparison question from the generated dataset.
2. Ask the base model and trained model the same prompt.
3. Display both answers side by side.
4. Label the comparison as a sanity check, not a benchmark.

Acceptance criteria:

- The notebook shows a concrete before/after answer.
- The comparison uses the user's uploaded content.
- The output does not claim broad benchmark improvement.

## Milestone 6: Export And Local Run Path

Purpose:

> Make the end of the demo point toward local ownership.

Files:

- Add `docs/local-run.md`.
- Update `notebooks/opendistillation_v0_demo.ipynb`.
- Update `README.md`.

Tasks:

1. Test whether GGUF export is practical in the v0 Colab path.
2. If practical, add the export cell and local command.
3. If not practical, show the exact export command as a documented follow-up and state what remains unverified.
4. Add llama.cpp and/or Ollama-style local run instructions.

Acceptance criteria:

- The notebook has a clear final output.
- The local-run instructions are honest about what has been tested.
- The README does not imply export works until it has been verified.

## Milestone 7: Public Release Polish

Purpose:

> Make the repo ready for public attention after the demo works.

Files:

- Update `README.md`.
- Update `docs/github-launch-checklist.md`.
- Confirm GitHub recognizes the Apache-2.0 license.
- Add contribution docs only when outside contributors are expected.

Tasks:

1. Confirm the repository displays Apache-2.0 license metadata on GitHub.
2. Run the notebook from a clean runtime.
3. Confirm `.gitignore` catches generated datasets and model outputs.
4. Review docs for overclaims.
5. Create GitHub issues from `docs/github-issue-plan.md`.

Acceptance criteria:

- A new visitor can understand the project in under one minute.
- The notebook has a tested happy path.
- No secrets, datasets, checkpoints, or model artifacts are committed.
