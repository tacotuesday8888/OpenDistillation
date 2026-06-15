# Recommended Next Goal Prompt

Use this as the next `/goal` after the six-row label/value T4 evidence lands:

```text
/goal Strengthen OpenDistillation's local fact-ledger label/value disambiguation signal before another GPU run.

For this task, write yourself a new goal and spawn agents in parallel — as many as needed to do it better and faster. Split the work into independent pieces, dispatch them concurrently, and synthesize the results as they return. Give each agent its own dedicated /goal.

Context:
Work in /Users/langqi/Developer/Projects/OpenDistillation. Keep v0 narrow: TXT/MD notes only, one notes/school model only, Colab-first, MockTeacherEngine as the safe default, optional local/open-source Qwen teacher, optional short TRL/PEFT LoRA training, and honest before/after quality reporting. Do not build SaaS, Mac app, phone app, accounts, backend, GGUF export, local runtime packaging, multiple profiles, coding model, writing model, work model, phone model, broad benchmark suite, Unsloth migration, bitsandbytes migration, or a larger training platform.

Evidence:
The 2026-06-15 bounded Colab T4 smoke used commit d918f5d1845a4651ea00427fa91d139c3862d935, Tesla T4, 8 extracted facts, 48 fact-ledger train rows, 8 held-out eval rows, zero leakage, zero missing expected terms, a six-row SFT preview, Qwen/Qwen2.5-0.5B-Instruct, and 30 TRL/PEFT LoRA steps. The adapter changed all 8 answers and improved exact held-out fact hits from 0/8 to 1/8. This is a bounded positive learning signal, not proof of general model quality. It still missed 7/8 facts and confused labels/values, such as Project codename -> Mira Vale and Review ritual time -> 14:07 PM.

Task:
Build a local, deterministic fact-ledger training-signal upgrade that specifically reduces label/value confusion before spending more GPU. Prefer simple maintainable Python changes in the existing package and notebook flow. Keep the public JSONL schema unchanged as instruction/response/source_chunk_id unless there is a proven reason to change it. Preserve exact train/eval separation, SFT preview visibility, readiness checks, and exact fact-hit scoring by fact/question identity.

Likely local direction:
- Add train row styles that teach label discrimination, not only value recall.
- Add negative or contrastive rows only if they are simple, deterministic, and still produce clear SFT prompt/completion text.
- Add validation that every expected label/value pair remains attached to the right fact after row generation and comparison selection.
- Add previews that make label/value swaps easy to inspect.
- Add tests for the exact failure pattern seen in the T4 run: swapped values, invented numeric values, and copied labels instead of values.

Do not:
- Change training knobs as the main intervention.
- Run another GPU smoke until the local report explains why the data should reduce label/value swaps.
- Claim model quality from adapter creation or changed answers.
- Broaden beyond TXT/MD notes or the notes/school model.

Exact next GPU experiment after local changes pass:
Run one bounded Colab T4 smoke with 8 facts, the new fact-ledger train rows, 8 held-out eval rows, 30-step Qwen/Qwen2.5-0.5B-Instruct LoRA, and exact fact-hit scoring. Treat it as useful only if trained exact held-out fact hits improve beyond 1/8 without increasing leakage or unscored answers.

Verification:
Run focused unit tests for changed behavior, full unit tests, notebook JSON validation if notebooks change, safe notebook CPU execution, git diff checks, secret scan, generated artifact/model/data scan, and final git status. If docs change, update docs/colab-smoke-test-results.md, docs/current-decisions.md, START_HERE.md, and this file only where the evidence requires it.

Finish:
Report what changed, what was verified, what remains next, and the exact next GPU experiment. Do not commit generated datasets, adapters, checkpoints, model files, secrets, .env files, API keys, or local config.
```

## Why This Goal

The six-row label/value signal finally produced one exact held-out fact hit, which means the direction is not completely dead. But the run still failed most facts and exposed the next local problem: the adapter often learned answer shape without binding the right label to the right value.

## Done Means

- Local train rows more directly teach each label's own value.
- The readiness report and SFT preview make that signal inspectable.
- Train/eval leakage remains zero.
- Exact fact-hit scoring remains attached by fact/question identity.
- Tests protect the failure pattern from the 2026-06-15 T4 run.
- Safe notebook defaults stay off for installs, real teacher, and training.
- Generated datasets, adapters, model artifacts, checkpoints, secrets, and local config stay out of git.

## Do Not Use This Goal For

- Expanding beyond TXT/MD notes.
- Adding product surfaces, accounts, SaaS, backend, Mac, phone, export, or multi-profile work.
- Migrating to Unsloth or bitsandbytes.
- Running broad benchmarks.
- Claiming model quality from changed answers alone.
