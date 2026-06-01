# Recommended Next Goal Prompt

Use this as the next `/goal` after the v0 skeleton is committed:

```text
/goal Choose and implement the first real OpenDistillation teacher-generation path. Work in /Users/langqi/Developer/Projects/OpenDistillation on the current branch. Keep the existing notebook flow and helper interfaces. Research current official docs for the chosen open-source teacher option, add a bounded teacher engine that produces the existing JSONL schema from TXT/MD chunks, clearly label whether user text leaves the notebook runtime, keep the deterministic mock teacher as the safe default or fallback, add tests for prompt construction and dataset validation, avoid real model training and large model artifacts, verify tests and notebook behavior, review the diff for secrets/generated files, and commit locally.
```

## Why This Goal

This is the right next step because the skeleton already demonstrates the flow. The next useful proof is replacing the mock QA generator with one bounded real teacher path while preserving the same dataset schema.

## Done Means

- A real teacher path is chosen based on current official docs.
- The mock teacher remains available as a safe deterministic path.
- Generated rows still validate against `docs/dataset-schema.md`.
- The notebook clearly states whether text is processed locally or remotely.
- Training and export remain placeholders.
- No generated datasets, model artifacts, checkpoints, secrets, or local config are committed.

## Do Not Use This Goal For

- Real fine-tuning.
- GGUF export implementation.
- SaaS, Mac app, account system, billing, or cloud backend.
