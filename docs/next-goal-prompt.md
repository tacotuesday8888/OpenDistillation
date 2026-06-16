# Recommended Next Goal Prompt

Use this as the next `/goal` after the local label/value disambiguation branch lands:

```text
/goal Run one bounded OpenDistillation fact-ledger T4 quality smoke for the local label/value disambiguation signal.

For this task, write yourself a new goal and spawn agents in parallel — as many as needed to do it better and faster. Split the work into independent pieces, dispatch them concurrently, and synthesize the results as they return. Give each agent its own dedicated /goal.

Context:
Work in /Users/langqi/Developer/Projects/OpenDistillation. Keep v0 narrow: TXT/MD notes only, one notes/school model only, Colab-first, MockTeacherEngine as the safe default, optional local/open-source Qwen teacher, optional short TRL/PEFT LoRA training, and honest before/after quality reporting. Do not build SaaS, Mac app, phone app, accounts, backend, GGUF export, local runtime packaging, multiple profiles, coding model, writing model, work model, phone model, broad benchmark suite, Unsloth migration, bitsandbytes migration, or a larger training platform.

Evidence:
The 2026-06-15 bounded Colab T4 smoke used commit d918f5d1845a4651ea00427fa91d139c3862d935, Tesla T4, 8 extracted facts, 48 fact-ledger train rows, 8 held-out eval rows, zero leakage, zero missing expected terms, a six-row SFT preview, Qwen/Qwen2.5-0.5B-Instruct, and 30 TRL/PEFT LoRA steps. The adapter changed all 8 answers and improved exact held-out fact hits from 0/8 to 1/8. This is a bounded positive learning signal, not proof of general model quality. It still missed 7/8 facts and confused labels/values, such as Project codename -> Mira Vale and Review ritual time -> 14:07 PM.

Latest local change:
The fact-ledger row builder now adds same-chunk label/value disambiguation rows without changing the public JSONL schema. For each fact that has another fact in the same chunk, the fifth and sixth rows teach the target value against a nearby contrast value and correct an explicit swapped `Label: wrong value` prompt. The sidecar manifest records contrast label/value metadata. The readiness report now shows disambiguation coverage; sample notes produce 8 facts, 48 train rows, 8 eval rows, 8/8 contrastable facts covered, and 16 disambiguation rows.

Task:
Run exactly one bounded Colab T4 quality smoke for the current disambiguation signal if local checks pass and non-interactive `google-colab-cli` is usable. Do not change training knobs, broaden scope, or edit product surfaces. Do not change the public JSONL schema: it remains `instruction`, `response`, `source_chunk_id`.

Exact GPU smoke:
- 8 extracted facts.
- 48 fact-ledger train rows.
- 8 held-out eval rows.
- 8/8 contrastable facts covered by 16 disambiguation rows.
- Zero exact train/eval leaks and zero near-duplicate/token-overlap leaks before training.
- 30-step `Qwen/Qwen2.5-0.5B-Instruct` TRL/PEFT LoRA on Colab T4.
- Exact fact-hit scoring attached by fact/question identity, not row position.

Quality rule:
Treat the smoke as useful only if trained exact held-out fact hits improve beyond 1/8 without leakage or unscored answers. Changed wording is not useful learning unless exact held-out fact hits improve over the previous best result.

Preserve:
Notebook defaults stay `INSTALL_TRAINING_DEPS = False`, `RUN_REAL_TEACHER = False`, and `RUN_TRAINING = False`. Fact metadata stays internal/sidecar. Do not commit generated datasets, adapters, checkpoints, model files, secrets, `.env` files, API keys, or local config.

Verification:
Before the T4 run, run focused unit tests, the full unit suite, notebook JSON validation, safe notebook CPU execution, `git diff --check`, secret scan, generated artifact/model/data scan, and git status. After the T4 run, update only evidence docs with the exact result, including raw base/trained fact-hit counts, changed-answer count, and whether the adapter still swaps labels/values.

Finish:
Report what changed, what was verified, what remains next, the GPU runtime used or blocker, exact base/trained fact-hit counts, and whether the result passed or failed by the exact fact-hit rule.
```

## Why This Goal

The local rows now directly target the failure seen in the latest T4 run: label/value swaps and invented values. The next evidence should be one bounded GPU check, not more training-knob changes or product-surface work.

## Done Means

- Local checks still show 8 facts, 48 train rows, 8 held-out eval rows, zero leakage, and 16 disambiguation rows for the sample notes.
- The T4 smoke either improves exact held-out fact hits beyond 1/8 or honestly records failure.
- Generated datasets, adapters, model artifacts, checkpoints, secrets, and local config stay out of git.

## Do Not Use This Goal For

- Expanding beyond TXT/MD notes.
- Adding product surfaces, accounts, SaaS, backend, Mac, phone, export, or multi-profile work.
- Migrating to Unsloth or bitsandbytes.
- Running broad benchmarks.
- Claiming model quality from changed answers alone.
