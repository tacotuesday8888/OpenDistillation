# Recommended Next Goal Prompt

Use this as the next `/goal` after the local quality-engine hardening lands:

```text
/goal Connect OpenDistillation's fact-ledger train/eval split to a bounded Colab T4 quality smoke for the v0 notes model.

Context:
Work in /Users/langqi/Developer/Projects/OpenDistillation on latest origin/main. Keep v0 narrow: TXT/MD notes only, one notes/school model only, Colab-first, MockTeacherEngine as the safe default, optional local/open-source Qwen teacher, optional short TRL/PEFT LoRA training, and honest before/after quality reporting. Do not build SaaS, Mac app, phone app, accounts, backend, GGUF export, local runtime packaging, multiple profiles, coding model, writing model, work model, phone model, broad benchmark suite, Unsloth migration, bitsandbytes migration, or a larger training platform.

Starting evidence:
The 2026-06-06 google-colab-cli T4 smoke at commit 0797ed21682960acc8e462db1d793ba357689258 trained a 30-step `Qwen/Qwen2.5-0.5B-Instruct` TRL/PEFT LoRA adapter from 24 mock rows. The adapter changed all four held-out sample-fact answers, but base and trained answers both hit 0/4 expected facts. Treat that as a failed learning signal, not a success.

The repo now has a hardened deterministic fact-ledger quality gate for explicit `Label: value` notes and safe bullet/list facts. The committed sample notes should produce 8 facts, 24 fact train rows, 8 held-out eval rows, zero exact train/eval leaks, zero near-duplicate leaks, and zero missing expected terms in the safe notebook path. Local tests also cover token-overlap leakage and strict expected-term scoring that does not accept partial-word matches.

Task:
Wire the hardened fact-ledger split into the next bounded sample-notes quality smoke. Use the fact-ledger train rows as the training/eval data source for the smoke, or document exactly why the current training path cannot use them yet and make the smallest local change needed. Keep the public JSONL schema as `instruction`, `response`, and `source_chunk_id`; keep fact metadata as sidecar/internal data. Score base and trained answers against the held-out fact-ledger eval rows with exact expected-term hits. Changed answers alone are not useful.

Run requirements:
- Use google-colab-cli first; use Chrome or Computer Use only if CLI auth/runtime control is blocked.
- Use the committed sample notes path, not uploaded notes.
- Use a T4 GPU runtime if available.
- Keep the run bounded: 8 facts, 24 fact train rows, 8 held-out eval rows, and a short `Qwen/Qwen2.5-0.5B-Instruct` LoRA run unless the code already exposes safer smaller defaults.
- Record package versions, GPU type, runtime, fact-ledger quality values, training steps, adapter path, base answers, trained answers, exact fact-hit counts, and final judgment: better, unchanged, or worse.
- Before launching training, print the hardened local quality-gate lines so the run shows fact count, fact coverage, exact leakage count, near-duplicate/token-overlap leakage count, and expected-term count.
- If Colab GPU quota or output panes block the run, recover evidence from CLI/status logs and document the blocker honestly.

Safe defaults:
Committed notebook defaults must remain off: `INSTALL_TRAINING_DEPS = False`, `RUN_REAL_TEACHER = False`, and `RUN_TRAINING = False`. Enable installs/training only in the live Colab runtime or temporary run script.

Docs to keep aligned:
README.md, docs/current-decisions.md, docs/open-source-tool-strategy.md, docs/roadmap.md, docs/first-demo-flow.md, docs/colab-smoke-test-results.md, docs/dataset-schema.md, notebooks/README.md, and this file.

Verification:
Run unit tests, notebook JSON validation, confirm no committed notebook outputs, default local notebook smoke path if touched, git diff --check, secret scan, artifact/model/data scan, and git status check. Commit and push only intended source/docs changes. Do not commit generated datasets, adapters, checkpoints, model files, secrets, or local config.

Finish:
Final response must include commit hash if any, whether Colab ran, GPU type if known, fact-ledger quality summary, training steps, base/trained exact fact-hit counts, judgment, blockers if any, and the next recommended goal.
```

## Why This Goal

The project now has the data/eval guardrail that was missing: a hardened fact ledger with separate train and held-out eval rows. The next useful question is whether the small adapter can improve exact held-out fact hits when trained and evaluated through that gate.

## Done Means

- Latest main is inspected before changing anything.
- The existing fact-ledger split is the basis for the bounded quality smoke.
- Train/eval leakage is checked before training.
- Exact expected-term hits are reported for base and trained answers.
- Useful learning is judged only by held-out fact-hit improvement.
- Safe committed defaults stay off.
- Generated datasets, adapters, model artifacts, checkpoints, secrets, and local config stay out of git.
- Unit tests, notebook JSON validation if touched, default local notebook smoke if touched, `git diff --check`, secret scan, artifact scan, and git status are checked.
- Changes are committed and pushed if docs or code changed.

## Do Not Use This Goal For

- Expanding beyond TXT/MD notes.
- Adding coding, writing, work, phone, or multi-profile flows.
- Implementing GGUF export or local runtime.
- Migrating to Unsloth or bitsandbytes.
- Claiming benchmark results.
