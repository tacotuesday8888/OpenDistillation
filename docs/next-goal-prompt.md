# Recommended Next Goal Prompt

Use this as the next `/goal` after the real-teacher Colab smoke result is committed:

```text
/goal Harden the first public OpenDistillation Colab demo without broadening v0. Work in /Users/langqi/Developer/Projects/OpenDistillation on the current branch. Keep scope narrow: TXT/MD notes only, one notes/school model only, Colab-first, deterministic MockTeacherEngine as the safe default, optional local Qwen real teacher, optional short TRL/PEFT LoRA training, and optional before/after comparison. Use Chrome as the first Colab control path and Computer as the fallback if Chrome cannot read or operate Colab. Do not ask the user for screenshots when those plugins can inspect the page. Do not build a SaaS, Mac app, phone app, account system, cloud backend, multi-profile system, GGUF export, local runtime, or real large training pipeline. Improve the first-demo experience around the verified path: clearer Colab run order, stronger log markers for long optional cells, a safer way to capture output if Colab output frames fail, and beginner-readable notes that the tiny smoke test proves wiring not model quality. Keep generated datasets/model artifacts/secrets out of git, update README/docs/current-decisions.md/docs/first-demo-flow.md/docs/colab-smoke-test-results.md/docs/github-issue-plan.md if status changes, run local verification, review the diff for secrets/artifacts, commit locally, and push.
```

## Why This Goal

The first optional real teacher path now has one Colab T4 wiring pass:

- `Qwen/Qwen2.5-1.5B-Instruct` loaded and generated one valid QA row from `examples/sample-notes.md`.
- Dataset validation passed.
- A 1-step `Qwen/Qwen2.5-0.5B-Instruct` TRL/PEFT LoRA adapter existed at `/content/OpenDistillation/outputs/notes-lora-real-teacher-smoke/adapter`.
- Before/after comparison ran.

That proves the path can work. It does not prove the model is useful yet. The next risk is making the first public demo reliable and understandable for a beginner without turning the project into a larger platform.

## Done Means

- The default notebook path still runs without GPU, package installs, model downloads, paid APIs, remote APIs, or training.
- Optional real teacher, training, and comparison cells have clear run order and clear skip/run markers.
- Long optional cells write concise status markers so a Colab output-frame failure does not erase evidence.
- Docs state that the real-teacher result is a tiny wiring smoke test, not a quality benchmark.
- `MockTeacherEngine` remains the safe fallback.
- No generated datasets, model artifacts, checkpoints, secrets, or local config are committed.

## Do Not Use This Goal For

- Expanding beyond TXT/MD notes.
- Adding coding, writing, work, phone, or multi-profile flows.
- Building account systems, cloud services, or paid API infrastructure.
- Implementing GGUF export or local runtime.
- Claiming model quality from a tiny smoke run.
