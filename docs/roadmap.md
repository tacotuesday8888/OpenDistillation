# Roadmap

OpenDistillation should move in small phases. Each phase must produce something a user can understand or run.

The long-term product is a personal model factory for the AI PC and AI phone era. The first working path is only a notes / school model from TXT/MD input.

## Phase 0: GitHub-Ready Foundation

Goal:

> A public visitor can understand the personal model factory vision and see a believable first prototype plan.

Deliverables:

- Public README with honest status.
- Product vision and roadmap aligned with the README.
- Exact v0 Colab flow.
- First-demo implementation plan.
- GitHub issue forms and initial issue plan.
- Guardrails against committing secrets, generated datasets, checkpoints, and model artifacts.
- Apache-2.0 license.

Exit criteria:

- Docs agree on the same narrow v0.
- The long-term multi-model vision is clearly labeled as future direction.
- No training pipeline or multi-profile implementation is implied to exist.
- Repo is safe to push once a GitHub remote is ready.

## Phase 1: Notes-Model Colab Skeleton

Goal:

> A user can open the notebook, load a `.txt` or `.md` notes file, and see the planned notes-model workflow shape with training skipped by default.

Deliverables:

- `notebooks/opendistillation_v0_demo.ipynb`.
- Upload and text validation cells.
- Text preview and chunk preview.
- Deterministic mock QA dataset preview.
- Clear labels showing optional training and export status.

Exit criteria:

- The notebook runs top to bottom without GPU.
- A beginner can see how notes become training examples.
- No real teacher model, paid API, model download, GPU, or remote text transfer is required for the default path.

## Phase 2: Real Teacher Path For Notes

Goal:

> Uploaded notes become a small inspectable JSONL dataset through one real teacher-generation path.

Deliverables:

- One real open-source teacher path.
- Clear local-versus-remote labeling.
- TXT/MD loader and chunker reused from the prototype.
- JSONL dataset schema.
- Dataset preview, quality report, and download step.

Exit criteria:

- A sample notes file creates a valid dataset.
- The user can inspect examples and dataset quality before training.
- The mock teacher remains available as a safe deterministic fallback.

Current status: implemented as an opt-in `HuggingFaceLocalTeacherEngine` using `Qwen/Qwen2.5-1.5B-Instruct`, with `RUN_REAL_TEACHER = False` keeping the default notebook path safe. Local tests cover request construction, JSONL parsing, schema validation, dataset quality reporting, dependency failures, model-load failures, generation/CUDA-memory failures, and invalid-output failures without downloading a model. A Colab T4 smoke test passed once with one sample-note chunk, one valid generated QA row, a 1-step LoRA adapter, and before/after comparison; broader teacher-output quality remains unproven.

## Phase 3: Short Notes-Model Training Path

Goal:

> A small student model can be fine-tuned from the generated notes dataset in Colab.

Deliverables:

- One recommended student model: `Qwen/Qwen2.5-0.5B-Instruct`.
- One efficient supervised fine-tuning path: TRL `SFTTrainer` with PEFT LoRA.
- Small default training run, skipped unless the notebook user opts in.
- Bounded before/after quality report, skipped unless training creates an adapter. The sample-notes path uses held-out fact questions; uploaded notes use chunk-diverse generated questions.
- Deterministic fact-ledger quality gate so train/eval leakage and exact expected-term coverage are checked before optional training.
- Clear warnings about runtime, GPU, and quality limits.
- Manual Colab GPU smoke-test checklist.

Exit criteria:

- The notebook demonstrates a visible, useful behavior change on the sample notes.
- The run stays small enough for a beginner Colab demo.

Current status: the bounded training engine, sample-fact/chunk-diverse comparison engine, runtime readiness helpers, manual smoke-test checklist, notebook entry points, and fact-ledger quality gate exist. Local tests cover configuration, dataset formatting, runtime messages, comparison request behavior, reference-overlap scoring, fake base-vs-adapter generation, source-chunk diversity, held-out sample questions, adapter-disabled base generation, fact extraction, fact train/eval row building, train/eval leakage checks, expected-term checks, and exact fact-hit scoring. A clean GitHub-opened Colab T4 runtime completed the optional dependency install, sample-notes flow, mock-generated dataset, LoRA adapter training, and real before/after output once before the multi-question report existed. A later Tesla T4 quality smoke ran the 16-row dataset quality report, 3-step LoRA training, and 3-question before/after report; all trained-adapter answers were identical to the base answers. Follow-up diagnosis found the comparison path could compare against the adapter-enabled model on both sides; the local helper now uses PEFT's adapter-disabled inference path for base answers. A second T4 smoke after that fix made all three trained answers change, but they were generic or hallucinated rather than useful notes answers. The current sample-fact experiment produces 4 chunks, 24 schema-valid mock rows, 4/4 chunk coverage, zero duplicate or near-duplicate questions, zero answer-length warnings, and 4 held-out fact questions. A 2026-06-06 Colab CLI T4 run trained the 30-step adapter and compared all four held-out questions. It changed all four answers, but both base and trained answers hit 0/4 expected facts, and the trained answers were still wrong or hallucinated. The safe notebook path now also reports a fact-ledger split from the same sample notes: 8 facts, 24 fact train rows, 8 held-out eval rows, zero exact train/eval leaks, zero near-duplicate leaks, and zero missing expected terms. Phase 3's useful behavior-change exit criterion is not met yet; the next bounded Colab smoke should train/evaluate against this fact-ledger split before changing training knobs.

## Phase 4: Local Notes-Model Run Path

Goal:

> The user can save the trained notes-model output and follow an explicit path toward local usage.

Deliverables:

- Saved adapter or merged model output.
- GGUF export if practical.
- If GGUF is not practical in v0, exact follow-up command and limitation.
- llama.cpp and/or Ollama-style run instructions.

Exit criteria:

- The notebook does not end at training only.
- The local-running story is honest and testable.

## Phase 5: Thin CLI

Goal:

> The notes-model Colab flow can be repeated locally with a small command-line wrapper.

Potential command:

```bash
opendistill run config.yaml
```

The CLI should stay thin. Shared logic should live in the Python package so Colab and local runs do not drift.

## Phase 6: Future Personal Model Types

Goal:

> Add new personal model profiles only after the notes model works end to end.

Possible future profiles:

- Notes / school model.
- Coding model.
- Writing model.
- Work model.
- Phone model.

These profiles should reuse the same boundaries where possible: input preparation, example generation, dataset validation, training, evaluation, and export. They should not be built as a broad multi-profile system before the first notes model is proven.

## Not In V0

- Multiple model profiles.
- Coding model implementation.
- Writing model implementation.
- Work model implementation.
- Phone model implementation.
- SaaS.
- Mac app.
- Account system.
- Billing.
- PDF parsing.
- Arbitrary web crawling.
- Large-scale distributed training.
- Full benchmark suite.
- Claims of novel distillation algorithms.
