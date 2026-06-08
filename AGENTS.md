# AGENTS.md

## Project Mission

OpenDistillation is an open-source personal model factory.

The product direction is:

> user TXT/MD material -> high-quality teacher-generated train/eval data -> smaller personal model -> honest quality report.

The long-term vision can include coding, writing, work, and phone-local models, but v0 stays narrow:

> one notes / school model from TXT/MD notes.

Model quality comes before GitHub polish, product packaging, or broad platform features.

## Current Stage

The repo has a Colab-first prototype. The wiring works: notes loading, chunking, dataset validation, mock teacher rows, optional Hugging Face teacher path, optional TRL/PEFT LoRA training, and before/after comparison have all been exercised.

The latest meaningful GPU quality result is still not good enough: a bounded Colab T4 smoke on the revised value-first fact-ledger rows trained a 30-step adapter on 24 fact-ledger rows, but both base and trained answers hit `0/8` held-out facts. The adapter changed every answer without learning the checked facts. The previous local diagnosis ruled out an obvious TRL/PEFT format bug and improved the row wording, but that was not enough. The current local direction is to diagnose the remaining learning signal and evaluation/reporting path before spending more GPU.

The current product-core design lives under `docs/superpowers/specs/`.

## Product Defaults

Use these defaults unless the user explicitly changes direction:

- Public positioning: personal model factory for the AI PC and AI phone era.
- First model type: notes / school model.
- First-run experience: Colab notebook.
- Input: `.txt` and `.md` notes only.
- Default teacher: deterministic `MockTeacherEngine`.
- Optional private/cheap teacher: small local Hugging Face Qwen model.
- First stronger teacher target: Qwen 7B/8B-class instruct model where feasible.
- Future teacher/critic candidates: larger Qwen or DeepSeek-style open-weight models, depending on cost, privacy, and hardware.
- Student target: small open-source instruct model, currently Qwen2.5-0.5B-class for Colab smoke tests.
- Training stack: PyTorch, Hugging Face Transformers, TRL `SFTTrainer`, PEFT LoRA.
- Acceleration path: Unsloth only after the standard TRL/PEFT path proves quality or hits memory/speed limits.
- Colab automation: prefer `google-colab-cli`; use Chrome or Computer Use only when the CLI cannot do the job.

## Scope Boundaries

Do not broaden v0 into:

- SaaS.
- Mac app.
- Phone app.
- Account system.
- Backend.
- GGUF export.
- Local runtime packaging.
- Multi-profile system.
- Broad benchmark suite.
- Coding, writing, work, or phone model flows.
- Arbitrary PDF/document ingestion.
- Dashboards, billing, or cloud training platform.

Those can be future product directions only after the notes-model quality loop works.

## Technical Direction

Prefer proven open-source ML tools over custom algorithms.

- Use PyTorch and Hugging Face libraries for model loading, generation, tokenization, training, adapters, and datasets.
- Use TRL/PEFT for SFT and LoRA instead of writing training loops by hand.
- Use current official docs before changing version-sensitive ML packages or model APIs.
- Treat Unsloth as a speed/memory optimization layer, not the core quality solution.
- Do not claim novel ML algorithms unless the repo actually implements and verifies them.

## RAG Role

RAG means searching the user's notes at answer time and giving the model relevant passages.

Use RAG as:

- exact memory for facts;
- citation/source support;
- a baseline for what can be answered without training;
- a helper for teacher data generation and evaluation.

Do not treat RAG as proof that the small model learned. Evaluate retrieval-assisted answers separately from adapter-only answers.

## Quality Rules

Do not claim model quality without held-out evidence.

The near-term quality loop should prioritize:

1. stable note chunks;
2. a fact ledger of facts the model should learn;
3. teacher-generated training rows;
4. separate held-out eval rows;
5. leakage checks so eval questions are not copied from training;
6. exact fact-hit scoring before subjective answer quality;
7. honest reporting of better, unchanged, or worse.

If a trained model changes its answers but still misses the facts, call that a failure.

## Privacy Rules

Real personal data is sensitive.

- Hosted teacher models are acceptable for synthetic/demo data.
- For real user data, require a local/private path or explicit opt-in before sending data to any hosted service.
- Never commit or print API keys, tokens, `.env` files, private data, local config, generated datasets, adapters, checkpoints, or model files.

## Communication Style

The user is still learning the ML details.

- Explain choices in plain language.
- Translate terms like LoRA, RAG, GGUF, teacher model, and eval into product meaning.
- Recommend the safest default when asking a question.
- Be direct when a path is too broad or weak.
- Prefer concrete next steps over abstract strategy.
- Do not keep goal prompts running if the project needs vision realignment first.

## Development Workflow

- Inspect the current repo before making changes.
- Plan before large changes; implement directly for small obvious fixes.
- Keep changes small, intentional, and aligned with the existing project.
- Before committing, run `git status`, inspect the diff, and check for secrets.
- Commit only intended files.
- Push verified work.
- If using plugins/tools for ML planning, prefer `@superpowers` for planning/debugging/quality work and `@Hugging Face` plus official docs for model/package choices.

## Verification

Before saying work is done, run the best relevant checks:

- unit tests when code changes;
- notebook JSON validation when notebooks change;
- default local notebook smoke when notebook flow changes;
- `git diff --check`;
- secret scan;
- generated artifact/model/data scan;
- final `git status`.

If verification cannot run, state exactly what was not verified and why.

## Near-Term Priority

The next implementation work should diagnose why the revised value-first fact-ledger T4 smoke still hit `0/8` exact facts before running another GPU job. Do not change training knobs, add export, or build product surfaces before the notes-model quality loop shows exact held-out fact improvement.
