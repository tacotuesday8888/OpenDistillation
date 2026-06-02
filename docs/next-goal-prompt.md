# Recommended Next Goal Prompt

Use this as the next `/goal` after the personal model factory positioning update is committed:

```text
/goal Choose and implement the first real OpenDistillation teacher-generation path for the v0 notes model. Work in /Users/langqi/Developer/Projects/OpenDistillation on the current branch. Keep the existing Colab notebook flow and helper interfaces. Research current official docs for the chosen open-source teacher option, add one bounded teacher engine that produces the existing JSONL schema from TXT/MD notes chunks, clearly label whether uploaded notes leave the notebook runtime, keep the deterministic mock teacher as the safe default or fallback, add tests for prompt construction and dataset validation, avoid real model training and large model artifacts, verify tests and notebook behavior, review the diff for secrets/generated files, and commit locally. Do not build coding, writing, work, phone, SaaS, Mac app, or multi-profile features in this goal.
```

## Why This Goal

The project now has the right long-term direction: a personal model factory for many future small models. The next useful implementation step is still narrow: replace the mock QA generator with one bounded real teacher path for the notes model only.

## Done Means

- A real teacher path is chosen based on current official docs.
- The mock teacher remains available as a safe deterministic path.
- Generated rows still validate against `docs/dataset-schema.md`.
- The notebook clearly states whether notes are processed locally or remotely.
- Training and export remain placeholders.
- No coding, writing, work, phone, or multi-profile implementation is added.
- No generated datasets, model artifacts, checkpoints, secrets, or local config are committed.

## Do Not Use This Goal For

- Real fine-tuning.
- GGUF export implementation.
- Coding model implementation.
- Writing model implementation.
- Work model implementation.
- Phone model implementation.
- SaaS, Mac app, account system, billing, or cloud backend.
