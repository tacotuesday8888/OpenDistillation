# Recommended Next Goal Prompt

Status: completed by the local fact-miss diagnostics work. Keep this file as historical context until the next task is chosen; do not treat it as the current next-goal prompt.

Use this as the next `/goal` after the failed same-chunk disambiguation T4 smoke:

```text
/goal Diagnose OpenDistillation's failed fact-ledger disambiguation learning signal locally before another GPU run.

For this task, write yourself a new goal and spawn agents in parallel — as many as needed to do it better and faster. Split the work into independent pieces, dispatch them concurrently, and synthesize the results as they return. Give each agent its own dedicated /goal.

Context:
Work in /Users/langqi/Developer/Projects/OpenDistillation. Keep v0 narrow: TXT/MD notes only, one notes/school model only, Colab-first, MockTeacherEngine as the safe default, optional local/open-source Qwen teacher, optional short TRL/PEFT LoRA training, and honest before/after quality reporting. Do not build SaaS, Mac app, phone app, accounts, backend, GGUF export, local runtime packaging, multiple profiles, coding model, writing model, work model, phone model, broad benchmark suite, Unsloth migration, bitsandbytes migration, or a larger training platform.

Evidence:
The 2026-06-16 bounded Colab T4 smoke used commit f8090fcb533159a9e22668c6b42f9f5f9a61b0b3, Tesla T4, 8 extracted facts, 48 fact-ledger train rows, 16 same-chunk disambiguation rows, 8 held-out eval rows, zero leakage, zero missing expected terms, a six-row SFT preview, Qwen/Qwen2.5-0.5B-Instruct, and 30 TRL/PEFT LoRA steps. The adapter changed all 8 answers but scored base 0/8 and trained 0/8 exact held-out fact hits. This failed the exact-hit rule because it did not improve beyond the previous best trained score of 1/8 and missed every checked fact.

Failure pattern:
The adapter mostly learned answer shape, not exact note values. Trained answers invented simple values such as `10`, `104`, `1047`, `10:45 AM`, `Alpha Zero`, and `version 2` while missing `Glass Harbor`, `copper-lantern-47`, `llama-harbor-alpha`, `4:17 PM`, `Mira Vale`, `notes-only-v0`, `basalt-arc-29`, and `ultramarine`.

Task:
Build a local, deterministic diagnosis and data-signal improvement that explains and targets this failure before any new GPU run. Keep the public JSONL schema unchanged as `instruction`, `response`, and `source_chunk_id`. Preserve train/eval separation, SFT preview visibility, fact/question-identity scoring, and plain-language readiness reporting.

Likely local direction:
- Compare the six-row 1/8 signal against the disambiguation 0/8 signal at the exact prompt/completion level.
- Identify whether the disambiguation rows over-teach generic correction patterns or numeric placeholder answers.
- Add local checks or previews that make invented-value risk visible before training.
- Strengthen rows only if the change directly teaches exact target values and avoids ambiguous contrast wording.
- Add tests for the 2026-06-16 failure pattern: answer-shape learning, invented numeric values, and missed expected terms despite changed answers.

Do not:
- Run another GPU smoke until local diagnostics show a concrete reason the data should beat 1/8.
- Change training knobs as the main intervention.
- Claim model quality from adapter creation or changed answers.
- Broaden beyond TXT/MD notes or the notes/school model.

Verification:
Run focused unit tests for changed behavior, full unit tests, notebook JSON validation if notebooks change, safe notebook CPU execution if notebook flow changes, `git diff --check`, secret scan, generated artifact/model/data scan, and final git status. If docs change, update docs/colab-smoke-test-results.md, docs/current-decisions.md, START_HERE.md, and this file only where the evidence requires it.

Finish:
Report what changed, what was verified, what remains next, and the exact criteria that must pass locally before another bounded T4 smoke is justified.
```

## Why This Goal

The same-chunk disambiguation rows were locally valid and GPU-runnable, but they regressed the exact-hit result to 0/8. This goal added local exact-miss diagnostics that classify the 2026-06-16 trained-answer failure as invented numeric/time/identifier values. The next useful work should use those diagnostics for a targeted row-signal change, not spend another T4 run on the same row shape or change training knobs.

## Done Means

- The failure mode from the 2026-06-16 T4 smoke is explained in local, inspectable terms.
- Any changed train rows make exact target values clearer without weakening train/eval separation.
- Readiness and SFT preview output expose the new signal clearly.
- Tests protect against changed-but-wrong answers being counted as progress.
- Safe notebook defaults stay off for installs, real teacher, and training.
- Generated datasets, adapters, model artifacts, checkpoints, secrets, and local config stay out of git.

## Do Not Use This Goal For

- Expanding beyond TXT/MD notes.
- Adding product surfaces, accounts, SaaS, backend, Mac, phone, export, or multi-profile work.
- Migrating to Unsloth or bitsandbytes.
- Running another GPU smoke without a new local reason.
- Claiming model quality from changed answers alone.
