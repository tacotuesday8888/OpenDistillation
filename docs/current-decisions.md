# Current Decisions

This file records decisions that should not be reopened without a concrete reason.

## Decided

- OpenDistillation is an open-source personal model factory for the AI PC and AI phone era.
- The long-term direction supports multiple future personal model types, including notes/school, coding, writing, work, and phone models.
- The v0 product promise is narrower than the long-term vision: start with a notes / school model from TXT/MD input.
- The first real user experience is Colab-first.
- The local CLI comes later as a thin reproducible wrapper.
- v0 accepts `.txt` and `.md` notes only.
- v0 does not implement multiple model profiles.
- v0 uses one default teacher path, not a menu of teachers.
- v0 targets one recommended student model around 0.5B-1.5B parameters.
- The default training method is response distillation / supervised fine-tuning.
- Logits distillation is a later experimental track only if technically feasible.
- The first output should point toward local use through GGUF, llama.cpp, and/or Ollama-style instructions.
- No generated datasets, model weights, checkpoints, API keys, `.env` files, or local machine config should be committed.
- The project uses the Apache-2.0 license.
- The default teacher path remains deterministic local `MockTeacherEngine` so the notebook can run without downloads or GPU.
- The first optional real teacher path is `HuggingFaceLocalTeacherEngine` with `Qwen/Qwen2.5-1.5B-Instruct`.
- The optional real teacher path is local/open-source: model weights download from Hugging Face, but notes text is not sent to a paid or proprietary remote API.
- The notebook should call small helper interfaces so real engines and future model types can be plugged in later without changing the first flow.
- The first student model is `Qwen/Qwen2.5-0.5B-Instruct`.
- The first training backend is Hugging Face TRL `SFTTrainer` with PEFT LoRA adapters.
- The first training path is optional in the notebook and defaults to skipped. It should run only after the user opts into a Colab GPU runtime.
- The first training path does not use Unsloth or bitsandbytes by default. Those are future optimizations after the quality loop can show whether the adapter is learning from notes.
- The first before/after quality report uses Hugging Face Transformers chat generation, PEFT `PeftModel.from_pretrained()` to load the trained LoRA adapter, and PEFT `disable_adapter()` for the base-model side of the comparison. It uses four held-out fact questions for the committed sample notes and up to four generated chunk-diverse questions for uploaded notes.
- The committed sample-notes quality experiment uses fact-rich notes, six mock rows per chunk, four held-out sample-fact comparison questions, and a bounded optional 30-step LoRA run. Uploaded/user notes keep the generated chunk-diverse comparison fallback.
- The first dataset-quality loop uses deterministic local checks only. Hugging Face Evaluate and LightEval are deferred until the project has a stable held-out notes evaluation set.
- The notebook includes an explicit `INSTALL_TRAINING_DEPS = False` switch so the default local path never installs packages, downloads models, or starts training by accident.
- The optional Colab install command does not upgrade Colab's preinstalled GPU `torch` package. The runtime still checks that `torch` and CUDA are available before training starts.
- The notebook setup clones the GitHub repository in fresh Colab runtimes before importing local helpers.
- Runtime readiness checks should explain missing optional packages, installed-package import failures, missing CUDA GPU, adapter-path problems, and likely GPU memory failures in plain language.
- The manual Colab GPU smoke test should use `docs/colab-smoke-test-checklist.md` before the optional training/comparison path is called verified.
- `RUN_REAL_TEACHER = False` is the notebook default. Real teacher generation starts only after the user opts in and installs the optional Hugging Face packages.
- The notebook writes concise `OD_STATUS` markers and `/tmp/opendistillation_status.jsonl` in Colab so the run state can be recovered if Colab's output frame fails.
- The fact-ledger quality gate is deterministic and local. It extracts explicit `Label: value` facts plus safe bullet/list facts, creates separate train/eval rows, checks exact and near-duplicate train/eval leakage, and scores exact expected-term hits.
- The public v0 training JSONL remains `instruction`, `response`, and `source_chunk_id`. Fact metadata stays in an internal sidecar manifest unless a later goal proves a reason to expand the public schema.
- The local quality engine stays standard-library-only for now. Optional ML paths can use Hugging Face Datasets, Transformers, TRL, and PEFT after opt-in, but extra retrieval/fuzzy-matching/vector-store dependencies wait until they clearly improve the notes-model quality loop.
- Changed adapter answers are not evidence of useful note learning unless held-out fact-hit scores improve over the base model.
- The 2026-06-08 fact-ledger Colab T4 smokes are failed learning results, not passes. The pre-fix run and the revised value-first run each trained on 24 fact-ledger rows and changed all 8 held-out answers, but base and trained answers both hit 0/8 exact expected facts.
- The follow-up local diagnosis keeps TRL/PEFT as the training stack. TRL's current SFT docs support conversational prompt/completion rows with completion-only loss, and PEFT's current docs confirm `disable_adapter()` is the right base-vs-adapter comparison path. The likely local failure was weak product-layer learning signal: exact values were not consistently the first/simple target, and the held-out wording invited generic "note field" answers.
- Fact-ledger train rows now use a six-row-per-fact label/value signal. The rows keep exact values prominent, repeat canonical `Label: value` bindings, preserve answer-only targets, and keep held-out eval rows in direct exact-recall wording. The internal sidecar records `row_style` and `value` for diagnosis. The public JSONL schema is unchanged.
- The revised value-first Colab T4 smoke handled the previous CLI log/config blocker by running `google-colab-cli` in an unrestricted local environment. It used Tesla T4, 30 steps, commit `9a173f4655cfca9385397717f4dd2c081b545e6d`, and still scored base 0/8 and trained 0/8 exact fact hits.
- Comparison rows now carry internal fact IDs, labels, row styles, and expected terms through source-chunk reordering. The temporary T4 runner's row-position scoring issue is fixed in product code; exact fact-hit scoring should use fact/question identity, not list position.
- A local SFT preview now shows the exact prompt/completion pairs before training. The sample fact-ledger training signal has six rows per fact.
- The 2026-06-15 six-row label/value Colab T4 smoke used commit `d918f5d1845a4651ea00427fa91d139c3862d935`, Tesla T4, 8 facts, 48 fact-ledger train rows, 8 held-out eval rows, 30 TRL/PEFT LoRA steps, and fact/question-identity exact scoring. It changed all 8 answers and improved exact held-out fact hits from base 0/8 to trained 1/8. Treat this as a bounded positive learning signal, not proof of general model quality, because 7/8 held-out facts were still missed.
- Fact-ledger train rows now replace two generic rows per same-chunk fact with deterministic contrast rows: one asks for the target label's value instead of a nearby label's value, and one is a known-values-only anti-invention row. The anti-invention row lists real same-note values and explicitly warns against invented number, time, identifier, name, or color substitutes. The public JSONL schema is still unchanged. The sidecar manifest records contrast label/value metadata, and readiness reports disambiguation plus known-values coverage before optional training.
- The 2026-06-16 same-chunk disambiguation Colab T4 smoke used commit `f8090fcb533159a9e22668c6b42f9f5f9a61b0b3`, Tesla T4, 8 facts, 48 fact-ledger train rows, 16 disambiguation rows, 8 held-out eval rows, 30 TRL/PEFT LoRA steps, and fact/question-identity exact scoring. It changed all 8 answers but scored base 0/8 and trained 0/8 exact fact hits. Treat this as a failed learning result because it did not beat the previous best 1/8 and still missed every checked fact.
- Local fact-miss diagnostics now sit on top of exact fact-hit scoring. They do not give credit for wrong answers; they explain misses as same-chunk value confusion, known-value confusion, invented numeric/time/identifier values, label echo, answer-shape-without-fact, or generic misses. The 2026-06-16 disambiguation trained-answer fixture is covered locally and classifies all eight trained answers as invented-value misses. The latest anti-invention fixture is also covered locally: it scores trained 2/8, identifies 6/8 invented-value misses, fails the stricter 3/8 plus max-5/8 invented-miss gate, and prints each missed fact's training rows, row styles, same-chunk values, known-values warning, expected value shape, and invented value shape.
- The current anti-invention row signal uses those diagnostics locally. It replaces the risky swapped-value correction row with a `known_values_only_label_value` row that only offers recorded same-note values and tells the model not to invent number/time/identifier/name/color substitutes. The internal preflight manifest from `scripts/prepare_anti_invention_smoke.py` is a local run contract, not a product CLI: it records source hashes, fact counts, row-style counts, zero-leakage checks, SFT preview rows, training config, held-out comparison questions, and the pass rule.
- The 2026-06-16 manifest-gated anti-invention Colab T4 smoke used commit `633d3656a96c6c1085f19a95c0b20d0de1ee26db`, Tesla T4, 8 facts, 48 fact-ledger train rows, 8 same-chunk disambiguation rows, 8 known-values-only anti-invention rows, 8 held-out eval rows, 30 TRL/PEFT LoRA steps, and fact/question-identity exact scoring. It changed all 8 answers and improved exact held-out fact hits from base 0/8 to trained 2/8. This passed the predeclared rule because it beat the previous best trained score of 1/8, but it is still weak evidence because the adapter missed 6/8 facts and every miss was still classified as an invented numeric/time/identifier value.
- The next GPU evidence run should not repeat the same anti-invention smoke or chase training knobs. It should follow a targeted data-signal change guided by the local fact-miss context audit and must reach at least 3/8 exact held-out fact hits with at most 5/8 invented number/time/identifier/name/color misses.

## Not Decided Yet

- Which sidecar metadata fields, if any, should graduate into a future public dataset schema.
- Whether a hosted teacher path is ever needed after the local Qwen teacher is smoke-tested.
- Whether GGUF export is implemented in v0 or documented as the immediate next command.
- Which future model profile comes after the notes / school model.
- GitHub repository visibility and remote URL.

## Working Recommendation

Use this notes-model v0 flow as the default implementation plan:

1. User opens the Colab notebook.
2. User uses the sample notes file or uploads one `.txt` or `.md` notes file.
3. The notebook validates and previews the text.
4. OpenDistillation chunks the notes.
5. OpenDistillation extracts a fact ledger from simple explicit note facts when present.
6. The mock teacher generates question-answer pairs by default.
7. The user can opt into `HuggingFaceLocalTeacherEngine` for local Qwen-generated rows with the same schema.
8. The dataset is previewed, quality-checked, saved, and downloadable.
9. The fact-ledger train/eval split reports fact coverage, exact or near-duplicate train/eval leakage, and expected-term coverage.
10. Optional training prepares `Qwen/Qwen2.5-0.5B-Instruct` with TRL `SFTTrainer` and PEFT LoRA.
11. A short supervised fine-tuning run starts only when the user sets `RUN_TRAINING = True` in a Colab GPU runtime.
12. When the fact-ledger gate passes, the notebook trains on fact-ledger train rows and compares base-model answers against trained-adapter answers using held-out fact-ledger eval questions. Otherwise it falls back to held-out sample-fact questions for the committed sample notes, or chunk-diverse generated questions for uploaded notes.
13. The notebook saves the output.
14. The notebook exports to GGUF or shows the exact export command and limitation.
15. The user gets local run instructions.

Do not build coding, writing, work, or phone model flows until the notes model path works end to end.

## Verified Versus Deferred

Verified locally in this repository:

- Dataset validation and JSONL serialization.
- Dataset quality reporting for row count, chunk coverage, duplicate or near-duplicate questions, answer length sanity, missing fields, and source chunk IDs.
- Mock teacher generation.
- Mock teacher question-style variety for factual recall, explanation, flashcard, and misconception-check rows.
- Optional real teacher selection, JSONL parsing, schema validation, and failure classification using fake no-download dependencies.
- Training configuration, request construction, and TRL/PEFT dataset formatting tests.
- Before/after comparison request construction, adapter-path validation, dependency handling, bounded chunk-diverse question selection, reference-overlap scoring, and fake base-vs-adapter generation tests with the adapter disabled for base answers.
- Optional runtime helper behavior for install-command text, missing-package checks, GPU/no-GPU formatting, and beginner-readable failure explanations.
- Optional runtime helper behavior for installed-package import failures, including the Colab `torchvision::nms` mismatch seen after upgrading `torch`.
- Notebook JSON parsing and the default CPU path where training remains skipped.
- Notebook default CPU path with dataset quality report: sample notes produced 4 chunks, 16 mock-teacher rows, 16 schema-valid rows, 4/4 chunk coverage, 0 duplicate questions, 0 near-duplicate questions, 0 short answers, training skipped, comparison skipped, and export skipped.
- Current notebook default CPU path with the fact-rich sample notes: sample notes produced 4 chunks, 24 mock-teacher rows, 24 schema-valid rows, 4/4 chunk coverage, 0 duplicate questions, 0 near-duplicate questions, 0 short answers, 0 long answers, 4 held-out sample-fact comparison rows, training skipped, and comparison skipped.
- Fact-ledger extraction, safe bullet/list fact extraction, train/eval row building, train/eval leakage checks, expected-term checks, exact fact-hit scoring, and exact-miss diagnostics are covered by local tests. The current anti-invention signal keeps the committed sample notes at 8 fact cards, 48 fact-ledger train rows, 8 same-chunk disambiguation rows, 8 known-values-only anti-invention rows, 8 held-out eval rows, 8/8 train coverage, 8/8 eval coverage, zero exact train/eval leaks, zero near-duplicate leaks, and zero missing expected terms in the safe notebook path.
- Anti-invention T4 smoke preflight manifest generation is covered locally. The completed 2026-06-16 manifest pinned the committed sample-notes contract at 8 facts, 48 train rows, 8 held-out eval rows, 8 disambiguation rows, 8 known-values-only rows, zero leakage, six SFT preview rows, row hashes, comparison-question hashes, and a historical failure rule that changed answers were not progress unless exact held-out fact hits beat the previous best 1/8. The current future manifest gate is stricter than the new 2/8 best: at least 3/8 exact held-out hits and at most 5/8 invented-value misses.
- Six-row label/value fact-ledger row wording, direct held-out eval wording, sidecar `row_style`/`value` metadata, and less overconfident fact-ledger report language are covered by local tests. The notebook skeleton test also verifies that a passing fact-ledger gate supplies the optional training rows and held-out comparison rows.
- Comparison metadata survives source-chunk reordering in local tests, so expected terms remain attached to the selected fact. SFT preview helpers are also covered by local tests.
- Comparison score-ready outputs now preserve internal row ID and expected value metadata for exact-miss diagnostics without changing the public training JSONL schema.
- Sample-fact Colab attempt on 2026-06-03 used pushed commit `bef902cd0cd4005ec5931e6190e1247e98fa936b` as the intended GitHub source. The official Colab CLI authenticated and successfully ran a CPU VM probe, then failed to assign a GPU runtime before code execution. Chrome control then operated the Colab UI, selected T4, and clicked Connect, but Colab showed "Cannot connect to GPU backend" and "You cannot currently connect to a GPU due to usage limits in Colab." No new T4 training or answer-quality evidence was collected for the sample-fact experiment.
- Sample-fact Colab CLI T4 smoke on 2026-06-06 used commit `0797ed21682960acc8e462db1d793ba357689258`, Tesla T4, `mock-local-teacher`, 24 generated rows, dataset quality reporting with 24/24 schema-valid rows, 4/4 chunk coverage, zero duplicate/near-duplicate questions, zero answer-length warnings, a 30-step `Qwen/Qwen2.5-0.5B-Instruct` TRL/PEFT LoRA adapter, and a 4-question held-out before/after report. The adapter changed all four answers, but base and trained answers both hit 0/4 expected facts. The trained answers were wrong or hallucinated, so the quality judgment is worse / not useful yet.
- Fact-ledger Colab CLI T4 smoke on 2026-06-08 used commit `479b110773ad9d3382523a4d98c5cca1645e0cdd`, Tesla T4, 8 extracted facts, 24 fact-ledger train rows, 8 held-out eval rows, 8/8 train coverage, 8/8 eval coverage, zero exact train/eval leaks, zero near-duplicate/token-overlap leaks, zero missing expected terms, and a 30-step `Qwen/Qwen2.5-0.5B-Instruct` TRL/PEFT LoRA adapter. The adapter changed all 8 answers, but base and trained answers both hit 0/8 exact expected facts. The judgment is unchanged by exact fact-hit count and not useful as learned note memory.
- Revised value-first fact-ledger Colab CLI T4 smoke on 2026-06-08 used commit `9a173f4655cfca9385397717f4dd2c081b545e6d`, Tesla T4, 8 extracted facts, 24 value-first fact-ledger train rows, 8 direct held-out eval rows, 8/8 train coverage, 8/8 eval coverage, zero exact train/eval leaks, zero near-duplicate/token-overlap leaks, zero missing expected terms, and a 30-step `Qwen/Qwen2.5-0.5B-Instruct` TRL/PEFT LoRA adapter. The adapter changed all 8 answers, but base and trained answers both hit 0/8 exact expected facts. The judgment is unchanged and still not useful as learned note memory.
- Six-row label/value fact-ledger Colab CLI T4 smoke on 2026-06-15 used commit `d918f5d1845a4651ea00427fa91d139c3862d935`, Tesla T4, 8 extracted facts, 48 fact-ledger train rows, 8 held-out eval rows, 8/8 train coverage, 8/8 eval coverage, zero exact train/eval leaks, zero near-duplicate/token-overlap leaks, zero missing expected terms, a visible six-row SFT preview, and a 30-step `Qwen/Qwen2.5-0.5B-Instruct` TRL/PEFT LoRA adapter. The adapter changed all 8 answers and improved exact fact hits from base 0/8 to trained 1/8. This is a bounded positive learning signal, but still not enough for useful learned note memory because 7/8 checked facts were missed.
- Same-chunk disambiguation fact-ledger Colab CLI T4 smoke on 2026-06-16 used commit `f8090fcb533159a9e22668c6b42f9f5f9a61b0b3`, Tesla T4, 8 extracted facts, 48 fact-ledger train rows, 16 same-chunk disambiguation rows, 8 held-out eval rows, 8/8 train coverage, 8/8 eval coverage, zero exact train/eval leaks, zero near-duplicate/token-overlap leaks, zero missing expected terms, a visible six-row SFT preview, and a 30-step `Qwen/Qwen2.5-0.5B-Instruct` TRL/PEFT LoRA adapter. The adapter changed all 8 answers but scored base 0/8 and trained 0/8 exact fact hits, so the result failed the exact-hit rule and regressed below the previous best trained score of 1/8.
- Manifest-gated anti-invention fact-ledger Colab CLI T4 smoke on 2026-06-16 used commit `633d3656a96c6c1085f19a95c0b20d0de1ee26db`, Tesla T4, 8 extracted facts, 48 fact-ledger train rows, 8 same-chunk disambiguation rows, 8 known-values-only anti-invention rows, 8 held-out eval rows, 8/8 train coverage, 8/8 eval coverage, zero exact train/eval leaks, zero near-duplicate/token-overlap leaks, zero missing expected terms, manifest SHA `2a0288f351732c1f3bec271dd176e7f1e7e9130aa41df9ae603bf9d0b3d32e64`, a visible six-row SFT preview, and a 30-step `Qwen/Qwen2.5-0.5B-Instruct` TRL/PEFT LoRA adapter. The adapter changed all 8 answers and improved exact fact hits from base 0/8 to trained 2/8. This met the pass threshold exactly and beat the previous best trained score of 1/8, but still missed 6/8 facts; all six misses were invented-value failures.
- Notebook default install path where `INSTALL_TRAINING_DEPS = False`.
- Notebook setup structure for the fresh Colab clone fallback.
- Notebook status markers for setup, optional install, teacher generation, dataset save, optional training, and optional comparison.
- Ignore rules for generated datasets, adapters, checkpoints, model weights, Hugging Face caches, notebook checkpoints, and common trainer artifacts.
- Recovered Colab T4 run after dependency fixes: model download, 1-step LoRA training, adapter creation, and before/after comparison completed once.
- Clean GitHub-opened Colab T4 run: bounded optional dependency install succeeded without upgrading `torch`, `USE_SAMPLE_NOTES = True` loaded the sample notes, `RUN_TRAINING = True` created `/content/OpenDistillation/outputs/notes-lora/adapter`, and before/after comparison printed both answers.
- Real-teacher Colab attempt on 2026-06-03 opened the GitHub notebook from `origin/main` at `740d105`, but Chrome/Colab control timed out before runtime selection, teacher model load, QA generation, dataset validation, training, or comparison evidence could be collected.
- Real-teacher Colab verification on 2026-06-03 used commit `a04538dfc999047255ddc4747d91d89e9f0ed3f6`, Tesla T4, `Qwen/Qwen2.5-1.5B-Instruct`, one sample-notes chunk, 1 generated QA row, passed dataset validation, confirmed a 1-step LoRA adapter at `/content/OpenDistillation/outputs/notes-lora-real-teacher-smoke/adapter`, and ran before/after comparison.
- Uploaded-notes Colab rehearsal on 2026-06-03 used commit `7113f87fa915b789cc77bbfb423b405defd9b5ec`. Setup and install-skip status markers printed in Colab, but Chrome file-chooser control timed out on Colab's `files.upload()` iframe before a `.txt` or `.md` file could be attached. Computer Use was installed but no callable Computer control tool was exposed in that thread.
- Uploaded-notes Colab retry on 2026-06-03 started from clean `origin/main` at `6a74bd88371044d235928301d6f87f4c926c36dc`. Chrome extension health checks passed and a fresh selected-profile window opened, but Chrome browser control still timed out before tab access. Computer Use was installed locally but still did not expose a callable control tool in that thread.
- Uploaded `.txt` Colab rehearsal on 2026-06-03 started from clean `origin/main` at `6f7d9c66cacb07dc82571abb85b3232285f6961c`. Computer Use clicked the actual Colab `Choose Files` button and the native macOS Open dialog selected `/private/tmp/opendistillation-upload-rehearsal-notes.txt`. The run reached validation, 1 chunk, `mock-local-teacher`, 2 QA rows, dataset saved to `/tmp/opendistillation_training_data.jsonl`, training skipped, and comparison skipped. The Colab Terminal status log recorded setup ready, install skipped, teacher succeeded, dataset saved, training skipped, and comparison skipped.
- Earlier uploaded `.md` Colab rehearsal on 2026-06-03 reached the actual Colab upload widget from clean `origin/main` at `6f7d9c66cacb07dc82571abb85b3232285f6961c`, but `/private/tmp/opendistillation-upload-rehearsal-notes.md` could not be attached through the native Open dialog or Chrome file-chooser fallback. That blocker was later resolved and is kept only as historical context.
- Uploaded `.md` Colab rehearsal on 2026-06-03 started from clean `origin/main` at `0e53cdd1e860a1c93007cb20e4c143c90f0a7af9`. Chrome edited the runtime-only upload cell to `USE_SAMPLE_NOTES = False`; Computer Use clicked the actual Colab `Choose Files` button; the native macOS Open dialog selected `/private/tmp/opendistillation-upload-rehearsal-notes.md` by path/suggestion; and the run reached validation, 1 chunk, `mock-local-teacher`, 2 QA rows, dataset saved to `/tmp/opendistillation_training_data.jsonl`, training skipped, and comparison skipped. The Colab Terminal status log recorded setup ready, install skipped, teacher succeeded, dataset saved, training skipped, and comparison skipped.
- Colab GPU quality smoke on 2026-06-03 used commit `276a8d3e050379a78212fa02f787ac7a0b44e245`, Tesla T4, sample notes, `mock-local-teacher`, 16 generated rows, dataset quality reporting with 16/16 schema-valid rows, 4/4 chunk coverage, zero duplicate/near-duplicate questions, zero answer-length warnings, a 3-step `Qwen/Qwen2.5-0.5B-Instruct` TRL/PEFT LoRA adapter, and a 3-question before/after report. The adapter answers were identical to the base answers on all three questions. Follow-up diagnosis found the comparison code loaded the PEFT adapter before generating the base answer, which could compare the adapter-enabled model to itself because PEFT may modify the passed base model in place. The local comparison path now uses `disable_adapter()` for base answers and chooses distinct source chunks first.
- Follow-up Colab GPU quality smoke on 2026-06-03 used commit `6a98c92599d1defa2b4a61510f7372f399f5fd87`, T4 GPU, sample notes, `mock-local-teacher`, the same 16-row dataset quality path, a 3-step `Qwen/Qwen2.5-0.5B-Instruct` TRL/PEFT LoRA adapter, and the adapter-disabled comparison path. All three trained-adapter answers changed, proving the comparison can now see adapter-side movement, but the actual answers were generic or hallucinated. The quality judgment remains not improved.

Still deferred or unverified:

- Real teacher output quality beyond a tiny 1-row smoke test.
- Whether larger notes files or more generated rows fit comfortably on T4 without extra memory cleanup.
- Whether a tiny Colab adapter run can produce useful note-grounded answers after the adapter-disabled comparison fix. Later sample-fact and fact-ledger smokes produced visible answer movement, and the latest anti-invention smoke reached a weak best of 2/8 exact held-out fact hits, but 6/8 facts were still missed.
- Why the revised value-first fact-ledger Colab T4 run still changed all 8 answers but hit 0/8 exact held-out facts after the row-position scoring issue was ruled out.
- Why the anti-invention signal helped two facts but still left six invented-value misses.
- GGUF export and local runtime instructions.
- Adapter quality beyond deterministic local quality helpers and the bounded Colab quality smokes.
- Larger uploaded notes files and higher row counts beyond the tiny TXT/MD rehearsal files.
- `docs/colab-smoke-test-results.md` records the first failed Colab T4 attempt, the recovered-runtime pass, the clean GitHub-opened T4 pass, the real-teacher end-to-end T4 verification, the uploaded-notes rehearsals, the first unchanged multi-question Colab quality smoke, the second adapter-disabled quality smoke with changed but not improved answers, the sample-fact smoke attempt that was blocked by Colab GPU usage limits before notebook execution, the 30-step sample-fact CLI T4 run that changed answers but still hit 0/4 expected facts, the first 2026-06-08 fact-ledger CLI T4 run that changed answers but still hit 0/8 exact expected facts, the revised value-first fact-ledger CLI T4 run that also changed answers but hit 0/8 exact expected facts, the 2026-06-15 six-row label/value CLI T4 run that improved from 0/8 to 1/8 exact expected facts while still missing 7/8 facts, the 2026-06-16 disambiguation CLI T4 run that changed all answers but regressed to 0/8, and the 2026-06-16 manifest-gated anti-invention T4 run that improved to 2/8 while still missing 6/8 facts with invented values.

Why this path:

- Qwen's model card lists `Qwen/Qwen2.5-0.5B-Instruct` as a 0.5B-parameter instruction model supported by current Hugging Face Transformers.
- TRL documents `SFTTrainer` for supervised fine-tuning and prompt/completion datasets.
- TRL documents direct PEFT adapter training through `peft_config=LoraConfig()`.
- PEFT documents loading saved adapters with `PeftModel.from_pretrained()` for inference.
- Transformers documents chat generation with `apply_chat_template()` and `generate()`.
- Transformers documents chat-style text generation through message dictionaries and the text-generation pipeline.
- `Qwen/Qwen2.5-1.5B-Instruct` is an Apache-2.0 Qwen2.5 chat/instruct model with about 1.5B parameters, which keeps the first real teacher path small enough for the v0 Colab experiment and inside the open-source constraint.
- PEFT LoRA keeps the trained artifact small.
- bitsandbytes and Unsloth are useful memory/speed tools, but they introduce more install, quantization, and hardware assumptions than the first beginner quality loop needs.

Sources checked on 2026-06-03:

- [Transformers chat basics](https://huggingface.co/docs/transformers/conversations)
- [TRL SFTTrainer docs](https://huggingface.co/docs/trl/en/sft_trainer)
- [PEFT task types](https://huggingface.co/docs/peft/main/package_reference/peft_types)
- [Hugging Face Evaluate overview](https://huggingface.co/docs/evaluate/index)
- [TRL Unsloth integration](https://huggingface.co/docs/trl/en/unsloth_integration)

Sources checked on 2026-06-08:

- [TRL SFTTrainer docs](https://huggingface.co/docs/trl/sft_trainer)
- [PEFT PeftModel docs](https://huggingface.co/docs/peft/package_reference/peft_model)
- [PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora)

Sources checked on 2026-06-02:

- [Qwen/Qwen2.5-0.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [Qwen/Qwen2.5-1.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [TRL SFTTrainer docs](https://huggingface.co/docs/trl/en/sft_trainer)
- [PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora)
- [PEFT PeftModel docs](https://huggingface.co/docs/peft/package_reference/peft_model)
- [Transformers Trainer docs](https://huggingface.co/docs/transformers/trainer)
- [Transformers text generation docs](https://huggingface.co/docs/transformers/llm_tutorial)
- [Transformers chat templating docs](https://huggingface.co/docs/transformers/chat_templating)
- [Accelerate docs](https://huggingface.co/docs/accelerate/index)
- [bitsandbytes docs](https://huggingface.co/docs/bitsandbytes/index)
- [Unsloth fine-tuning guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide)
