# Recommended Next Goal Prompt

Use this as the next `/goal` after the value-first fact-ledger Colab T4 smoke lands:

```text
/goal Diagnose why OpenDistillation's value-first fact-ledger T4 smoke still failed 0/8 before spending more GPU.

Context:
Work in /Users/langqi/Developer/Projects/OpenDistillation on latest origin/main. Keep v0 narrow: TXT/MD notes only, one notes/school model only, Colab-first, MockTeacherEngine as the safe default, optional local/open-source Qwen teacher, optional short TRL/PEFT LoRA training, and honest before/after quality reporting. Do not build SaaS, Mac app, phone app, accounts, backend, GGUF export, local runtime packaging, multiple profiles, coding model, writing model, work model, phone model, broad benchmark suite, Unsloth migration, bitsandbytes migration, or a larger training platform.

Evidence:
The pre-fix 2026-06-08 fact-ledger Colab T4 smoke used 8 facts, 24 fact-ledger train rows, 8 held-out eval rows, zero train/eval leakage, and a 30-step `Qwen/Qwen2.5-0.5B-Instruct` TRL/PEFT LoRA adapter. It changed all 8 answers, but base and trained answers both hit 0/8 exact expected facts.

The follow-up local fix made fact-ledger train targets value-first, used direct exact-recall held-out eval wording, recorded `row_style` and `value` in the sidecar, and made the notebook use fact-ledger train/eval rows after a passing gate.

The revised value-first Colab T4 smoke then ran successfully through `google-colab-cli` on Tesla T4 at commit `9a173f4655cfca9385397717f4dd2c081b545e6d`. It used 8 facts, 24 value-first train rows, 8 direct held-out eval rows, zero exact leaks, zero near/token-overlap leaks, zero missing expected terms, `Qwen/Qwen2.5-0.5B-Instruct`, and 30 TRL/PEFT LoRA steps. It changed all 8 answers, but base and trained answers both hit 0/8 exact facts. Treat this as failed learning.

Important tooling note:
The temporary GPU runner printed per-answer expected terms out of order for some rows because the comparison helper reorders rows to cover distinct source chunks first. The corrected score by question label is still base 0/8 and trained 0/8. Fix product/reporting code so exact fact-hit scoring carries expected terms by fact/question identity, not by row position.

Task:
Do local diagnosis only unless a very small code change clearly makes another GPU run justified. Start by proving the exact SFT text the model saw, the exact held-out prompts, the adapter/base comparison path, and the expected-term scorer. Look for why the model is binding labels to wrong values or generic values after training.

Implementation priorities:
- Add tests that exact fact-hit scoring survives comparison row reordering.
- Make comparison/reporting carry fact IDs, labels, expected terms, and row styles where needed without changing the public JSONL schema.
- Add a local SFT preview/report that shows the exact prompt/completion text for each fact before training.
- Improve the plain-English report so a user understands that changed answers with wrong facts are failure.
- Decide whether the next data change should increase per-fact examples, reduce label/value ambiguity, train with label+value pairs more explicitly, or add an eval/reporting guard before any new GPU run.

Constraints:
Keep public JSONL schema stable: `instruction`, `response`, `source_chunk_id`. Fact metadata stays internal/sidecar. Notebook defaults stay `INSTALL_TRAINING_DEPS = False`, `RUN_REAL_TEACHER = False`, and `RUN_TRAINING = False`. Do not run another Colab/GPU job unless local evidence shows a specific fix worth exactly one bounded smoke.

Docs to keep aligned:
README.md, docs/current-decisions.md, docs/roadmap.md, docs/first-demo-flow.md, docs/colab-smoke-test-results.md, docs/dataset-schema.md, notebooks/README.md, and this file.

Verification:
Run unit tests, notebook JSON validation if touched, confirm no committed notebook outputs, local notebook smoke if notebook flow changes, git diff --check, secret scan, artifact/model/data scan, staged diff inspection, and git status. Commit and push only intended docs/source changes. Do not commit generated datasets, adapters, checkpoints, model files, secrets, or local config.

Finish:
Final response must include commit hash if any, diagnosis, files changed, verification results, whether Colab/GPU was avoided, and the next GPU-smoke condition if one exists.
```

## Why This Goal

The revised value-first GPU run failed the same exact fact-hit gate as the earlier fact-ledger run. Another training job without a sharper local hypothesis would likely spend compute to produce another changed-but-wrong adapter.

## Done Means

- The failed value-first T4 evidence is inspected and summarized.
- Exact fact-hit scoring is tied to fact/question identity, not row position.
- Any report change preserves the public JSONL schema.
- Safe committed notebook defaults stay off.
- Generated datasets, adapters, model artifacts, checkpoints, secrets, and local config stay out of git.
- Unit tests, notebook checks when relevant, `git diff --check`, secret scan, artifact scan, staged diff inspection, and git status are checked.
- Changes are committed and pushed if docs or code changed.

## Do Not Use This Goal For

- Running another GPU smoke without a specific local fix.
- Expanding beyond TXT/MD notes.
- Adding coding, writing, work, phone, or multi-profile flows.
- Implementing GGUF export or local runtime.
- Migrating to Unsloth or bitsandbytes.
- Claiming benchmark results.
