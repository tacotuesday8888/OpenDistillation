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
- TXT/MD loader and chunker reused from the skeleton.
- JSONL dataset schema.
- Dataset preview and download step.

Exit criteria:

- A sample notes file creates a valid dataset.
- The user can inspect examples before training.
- The mock teacher remains available as a safe deterministic fallback.

## Phase 3: Short Notes-Model Training Path

Goal:

> A small student model can be fine-tuned from the generated notes dataset in Colab.

Deliverables:

- One recommended student model: `Qwen/Qwen2.5-0.5B-Instruct`.
- One efficient supervised fine-tuning path: TRL `SFTTrainer` with PEFT LoRA.
- Small default training run, skipped unless the notebook user opts in.
- Before/after prompt comparison, skipped unless training creates an adapter.
- Clear warnings about runtime, GPU, and quality limits.
- Manual Colab GPU smoke-test checklist.

Exit criteria:

- The notebook demonstrates a visible behavior change on the sample notes.
- The run stays small enough for a beginner Colab demo.

Current status: the bounded training engine, comparison engine, runtime readiness helpers, manual smoke-test checklist, and notebook entry points exist. Local tests cover configuration, dataset formatting, runtime messages, comparison request behavior, and fake base-vs-adapter generation. A clean GitHub-opened Colab T4 runtime completed the optional dependency install, sample-notes flow, LoRA adapter training, and real before/after output once. The output is a qualitative wiring check, not a benchmark.

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
