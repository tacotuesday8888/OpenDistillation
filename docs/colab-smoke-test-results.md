# Colab GPU Smoke-Test Results

Last updated: 2026-06-08

## Result

Fresh Colab GPU training from a clean GitHub-opened runtime is **verified once** on 2026-06-02.

The optional real-teacher Colab path is **verified once** on 2026-06-03. The verified path used sample TXT/MD notes, `Qwen/Qwen2.5-1.5B-Instruct` as the local real teacher, dataset validation, a 1-step TRL/PEFT LoRA adapter from `Qwen/Qwen2.5-0.5B-Instruct`, and before/after comparison on a Tesla T4 runtime.

The uploaded-notes Colab rehearsal is **verified once** on 2026-06-03 for both one `.txt` file and one `.md` file. Each ran through validation, chunking, mock-teacher rows, dataset save, training skipped, and comparison skipped with status-log evidence.

The first quality-loop update is **verified locally and twice in Colab T4** on 2026-06-03. The default sample-notes path produced varied mock-teacher rows and a deterministic dataset quality report. The first Colab GPU quality smoke run trained a 3-step LoRA adapter and ran the new three-question model-quality report, but the trained-adapter answers were identical to the base-model answers, so that recorded quality result was **unchanged / no visible improvement**.

Follow-up local diagnosis on 2026-06-03 found a comparison bug: the PEFT adapter was loaded before base-answer generation, and PEFT documents that the passed base model may be modified in place. That means the first quality smoke may have compared the adapter-enabled model to itself. The local comparison helper now generates base answers inside PEFT's `disable_adapter()` context and chooses questions from distinct source chunks first. The second Colab T4 quality smoke ran after that fix. All three trained-adapter answers changed, so the fixed comparison path can now see adapter-side movement. The answer quality was **not improved / worse overall** because the trained answers were still generic or hallucinated.

The sample-fact learning-signal experiment is **verified in a Colab CLI T4 training run** on 2026-06-06. The run used fact-rich sample notes, 24 mock rows, zero dataset-quality issues, four held-out sample-fact questions, and a bounded 30-step LoRA adapter. The adapter changed all four answers, but it did **not** answer the held-out facts better than the base model. Base fact hits: 0/4. Trained-adapter fact hits: 0/4. Final judgment: **worse / not useful yet** because the trained answers were still wrong or hallucinated and did not contain `Glass Harbor`, `copper-lantern-47`, `llama-harbor-alpha`, or the `4:17 PM` plus `ultramarine` pair.

The fact-ledger learning-signal experiment is **verified in a Colab CLI T4 training run** on 2026-06-08. The run used the hardened local fact-ledger split from commit `479b110773ad9d3382523a4d98c5cca1645e0cdd`: 8 facts, 24 fact-ledger train rows, 8 held-out eval rows, zero exact leaks, zero near-duplicate/token-overlap leaks, and zero missing expected terms. The 30-step LoRA adapter changed all 8 answers, but it did **not** improve exact held-out fact hits. Base fact hits: 0/8. Trained-adapter fact hits: 0/8. Final judgment: **unchanged by exact fact-hit count / not useful yet** because changed wording without the expected terms is not learned note memory.

An earlier 2026-06-03 attempt failed before model execution because Chrome/Colab control timed out. That older result is kept below as a tooling note, but it is superseded by the successful real-teacher run recorded here.

The clean run passed after three fixes were pushed:

- Do not upgrade Colab's preinstalled GPU `torch` package.
- Do not pass `assistant_only_loss=True` for TRL prompt/completion rows.
- Load `examples/sample-notes.md` by default in Colab so the first smoke path does not block on a file picker.

The clean run used the GitHub notebook at `main`, a fresh T4 runtime, `INSTALL_TRAINING_DEPS = True`, `USE_SAMPLE_NOTES = True`, and `RUN_TRAINING = True`. It installed the bounded Hugging Face package set without upgrading `torch`, loaded the sample notes, created mock QA examples, trained a LoRA adapter, and printed before/after answers.

## Colab GPU Quality Smoke: Fact-Ledger 30-Step CLI Run

Date: 2026-06-08

Status: **ran on Colab T4 / exact fact hits unchanged at 0/8**.

Verified path:

```text
fact-rich sample-notes.md -> hardened fact ledger -> 24 fact-ledger train rows -> 30-step Qwen2.5-0.5B LoRA adapter -> 8 held-out fact-ledger eval questions
```

Repo and runtime state:

```text
Execution path: google-colab-cli command name `colab`
Git commit used by Colab clone: 479b110773ad9d3382523a4d98c5cca1645e0cdd
Runtime: Colab T4 GPU
GPU: Tesla T4
CUDA available: true
python: 3.12.13
torch: 2.11.0+cu128
transformers: 4.57.6
datasets: 5.0.0
trl: 0.29.1
peft: 0.18.1
accelerate: 1.13.0
```

The CLI T4 probe passed immediately before the full smoke:

```text
OD_COLAB_T4_PROBE {"cuda": true, "gpu": "Tesla T4", "python": "3.12.13", "status": "ok", "torch": "2.11.0+cu128"}
```

Live-only settings:

```text
INSTALL_TRAINING_DEPS = True
RUN_REAL_TEACHER = False
RUN_TRAINING = True
Student model: Qwen/Qwen2.5-0.5B-Instruct
training max_steps = 30
comparison max_new_tokens = 80
```

Fact-ledger quality evidence before training:

```text
Input file: examples/sample-notes.md
Chunks: 4
Facts: 8
Fact-ledger train rows: 24
Held-out eval rows: 8
Train fact coverage: 8/8
Eval fact coverage: 8/8
Exact train/eval leaks: 0
Near-duplicate/token-overlap train/eval leaks: 0
Missing expected terms: 0
Issues: 0
Quality gate passed: true
Runtime train JSONL path: /content/opendistillation_fact_ledger_outputs/fact_ledger_train_rows.jsonl
```

Training evidence:

```text
Training ran: true
Training engine: trl-sfttrainer-peft-lora
Max steps: 30
Adapter path: /content/opendistillation_fact_ledger_outputs/notes-lora-fact-ledger/adapter
Adapter exists: true
Adapter files: README.md, adapter_config.json, adapter_model.safetensors, added_tokens.json, chat_template.jinja, merges.txt, special_tokens_map.json, tokenizer.json, tokenizer_config.json, training_args.bin, vocab.json
Training elapsed runtime: 85.7 seconds
```

Comparison evidence:

```text
Comparison ran: true
Comparison engine: transformers-peft-before-after
Comparison questions: 8
Changed answers: 8/8
Base fact hits: 0/8
Trained-adapter fact hits: 0/8
Automatic judgment: unchanged
Comparison elapsed runtime: 34.6 seconds
Final marker: OD_FACT_LEDGER_SMOKE {"status": "succeeded", "stage": "done", "base_hits": 0, "trained_hits": 0, "changed_answers": 8, "questions": 8, "judgment": "unchanged"}
```

Questions and answer evidence:

The structured hit counts above are exact from the Colab run. The retained thread context includes the trained answers and the base-answer evidence below; base answers are marked as excerpts or summaries where the full output text was not available after the Colab session ended.

```text
1. During a closed-book check, which answer belongs with the note field "project codename"?
Expected term: Glass Harbor
Base answer excerpt:
When checking for answers in a closed-book test or exam, it's important to focus on the content and relevance...
Trained adapter answer:
Mistral-alpha-49.
Expected term hit: base false, trained false

2. During a closed-book check, which answer belongs with the note field "notebook signal phrase"?
Expected term: copper-lantern-47
Base answer summary:
Generic closed-book / note-field explanation, not the expected note phrase.
Trained adapter answer:
During a closed-book check, the notes field "notebook signal phrase" belongs to Alpha.
Expected term hit: base false, trained false

3. During a closed-book check, which answer belongs with the note field "local runner label"?
Expected term: llama-harbor-alpha
Base answer summary:
Generic explanation about providing an answer related to a specific note field, not the expected runner label.
Trained adapter answer:
During a closed-book check, the notes field local runner label is equivalent to volume 1, problem 2.
Expected term hit: base false, trained false

4. During a closed-book check, which answer belongs with the note field "review ritual time"?
Expected term: 4:17 PM
Base answer summary:
Generic review-time explanation, not the expected time.
Trained adapter answer:
During a closed-book check, the notes field review ritual time belongs to "notes".
Expected term hit: base false, trained false

5. During a closed-book check, which answer belongs with the note field "demo owner alias"?
Expected term: Mira Vale
Base answer summary:
Generic answer about aliases, not the expected owner alias.
Trained adapter answer:
During a closed-book check, the notes field demo owner alias is alpha-beta-code.
Expected term hit: base false, trained false

6. During a closed-book check, which answer belongs with the note field "safety boundary phrase"?
Expected term: notes-only-v0
Base answer summary:
Generic safety-boundary phrase explanation, not the expected boundary phrase.
Trained adapter answer:
During a closed-book check, the notes field "safety boundary phrase" belongs to "no".
Expected term hit: base false, trained false

7. During a closed-book check, which answer belongs with the note field "export placeholder name"?
Expected term: basalt-arc-29
Base answer summary:
Generic notes/placeholders explanation, not the expected placeholder name.
Trained adapter answer:
During a closed-book check, the notes field with export placeholder name belongs to note-1.
Expected term hit: base false, trained false

8. During a closed-book check, which answer belongs with the note field "review ritual color"?
Expected term: ultramarine
Base answer summary:
Generic explanation about a color used for review, not the expected color.
Trained adapter answer:
Mystic Vale.
Expected term hit: base false, trained false
```

Interpretation:

- The hardened local data/eval gate did its job: it produced separated train and eval rows with zero required leakage or expected-term issues.
- The GPU run did not prove useful learning. The adapter changed every answer but still missed every exact expected term.
- The required local diagnosis is recorded below. The next useful GPU evidence is one bounded T4 smoke of the revised value-first fact-ledger rows.

Generated datasets, adapters, checkpoints, and model files stayed inside the Colab runtime and were not copied into this repository.

## Local Diagnosis After The 0/8 Fact-Ledger Run

Date: 2026-06-08

Status: **local product-layer fix prepared / no new GPU quality claim yet**.

Diagnosis:

- Current TRL SFT docs support the prompt/completion dataset shape OpenDistillation uses, including conversational prompt/completion rows and completion-only loss.
- Current PEFT docs confirm that `PeftModel.from_pretrained()` may modify the passed base model in place and that `disable_adapter()` is the supported base-inference path. The comparison helper already uses that path.
- The likely local failure was weak learning signal rather than a known TRL/PEFT formatting bug: each fact value appeared only a few times, exact values were sometimes buried after longer phrasing, and the held-out eval wording asked "which answer belongs with the note field", which encouraged generic field explanations.

Local changes prepared after the failed run:

```text
Fact-ledger train row 1 per fact: answer-only exact value
Fact-ledger train rows 2-3 per fact: response starts with "Exact answer: <value>"
Held-out eval row per fact: direct closed-book exact-recall question
Sidecar metadata: row_style and value fields added for diagnosis
Notebook optional training source: fact-ledger train rows when the gate passes
Notebook optional comparison source: held-out fact-ledger eval questions when the gate passes
```

Local gate after the fix still reports:

```text
Facts: 8
Fact-ledger train rows: 24
Held-out eval rows: 8
Train fact coverage: 8/8
Eval fact coverage: 8/8
Exact train/eval leaks: 0
Near-duplicate/token-overlap train/eval leaks: 0
Missing expected terms: 0
```

No GPU rerun is recorded in this section. In the overnight/no-permission local pass, even a harmless `colab status --help` probe failed before runtime access because the CLI tried to write `/Users/langqi/.config/colab-cli/colab.log`, which this sandbox cannot modify. Because the run instructions forbid approval prompts, manual auth, or interactive runtime control, no new GPU job was started here. The next useful GPU evidence is one bounded T4 smoke of the revised value-first fact-ledger rows when non-interactive Colab CLI access is available.

## Colab GPU Quality Smoke: Sample-Fact 30-Step CLI Run

Date: 2026-06-06

Status: **ran on Colab T4 / answer quality worse overall**.

Verified path:

```text
fact-rich sample-notes.md -> mock-local-teacher rows -> deterministic dataset quality report -> 30-step Qwen2.5-0.5B LoRA adapter -> 4 held-out sample-fact before/after questions
```

Repo and runtime state:

```text
Execution path: google-colab-cli
Git commit used by Colab clone: 0797ed21682960acc8e462db1d793ba357689258
Runtime: Colab T4 GPU
GPU: Tesla T4
CUDA available: true
python: 3.12.13
torch: 2.11.0+cu128
transformers: 4.57.6
datasets: 5.0.0
trl: 0.29.1
peft: 0.18.1
accelerate: 1.13.0
```

The CLI T4 probe passed immediately before the full smoke:

```text
OD_COLAB_CLI_PROBE {"cuda": true, "gpu": "Tesla T4", "python": "3.12.13", "torch": "2.11.0+cu128", "torch_imported": true}
```

Live-only settings:

```text
INSTALL_TRAINING_DEPS = True
USE_SAMPLE_NOTES = True
RUN_REAL_TEACHER = False
RUN_TRAINING = True
examples_per_chunk = 6
training max_steps = 30
comparison max_examples = 4
```

Dataset quality evidence:

```text
Input file: sample-notes.md
Characters: 778
Approx. words: 118
Chunks: 4
Teacher engine: mock-local-teacher
Generated examples: 24
Rows: 24 total, 24 schema-valid
Chunk coverage: 4/4
Duplicate questions: 0
Near-duplicate questions: 0
Very short answers: 0
Very long answers: 0
Issues: 0
Held-out sample-fact questions: 4
Runtime dataset path: /tmp/opendistillation_sample_fact_training_data.jsonl
```

Training evidence:

```text
Training ran: true
Student model: Qwen/Qwen2.5-0.5B-Instruct
Training engine: trl-sfttrainer-peft-lora
Max steps: 30
Adapter path: /content/opendistillation_sample_fact_outputs/notes-lora-sample-fact/adapter
Adapter exists: true
Adapter files: README.md, adapter_config.json, adapter_model.safetensors, added_tokens.json, chat_template.jinja, merges.txt, special_tokens_map.json, tokenizer.json, tokenizer_config.json, training_args.bin, vocab.json
Training elapsed runtime: 103.9 seconds
```

Comparison evidence:

```text
Comparison ran: true
Comparison engine: transformers-peft-before-after
Comparison questions: 4
Changed answers: 4/4
Base fact hits: 0/4
Trained-adapter fact hits: 0/4
Overlap deltas: -0.300, -0.375, +0.000, +0.300
Automatic signal: changed_without_more_fact_hits
Comparison elapsed runtime: 15.6 seconds
Quality judgment: worse / not useful yet. The adapter changed every answer, but none of the trained answers contained the expected held-out facts.
```

Questions and answers:

```text
1. In the demo notes, what phrase is listed as the project codename?
Reference answer:
The project codename listed in the demo notes is Glass Harbor.
Base answer:
The project codename mentioned in the demo notes is "Tianhe-1". This refers to the first Chinese space mission launched by China's Tianhe core module, which was named after the ancient Chinese emperor of the same name. The Tianhe-1 mission marked the beginning of China's space station program and set the stage for future missions to orbit Earth and beyond.
Trained adapter answer:
The project codename is Echo.
Expected term hit: base false, trained false
Overlap: base 0.700, trained 0.400, delta -0.300

2. Which checkpoint phrase should verify that the sample notes were remembered?
Reference answer:
The checkpoint phrase for the sample notes is copper-lantern-47.
Base answer:
To ensure that the sample notes were remembered, you can use the following checkpoint phrases:
1. "The sample notes were reviewed and verified."
2. "All sample notes have been checked for accuracy and consistency."
3. "Sample notes have been thoroughly analyzed to ensure they accurately reflect the topic."
These checkpoints will help confirm that the sample notes have been correctly recorded and verified before being used in your analysis or reporting process.
Trained adapter answer:
The verification key is remember_samples_notes.
Expected term hit: base false, trained false
Overlap: base 0.625, trained 0.250, delta -0.375

3. Which local runner label does the sample say to remember?
Reference answer:
The local runner label to remember is llama-harbor-alpha.
Base answer:
I'm sorry, but I don't have access to specific information about any particular sample or its contents. Without knowing which sample you're referring to and what it is about, I can't provide an accurate answer. Could you please clarify your question or provide more details about the sample in question?
Trained adapter answer:
The sample says to remember H2O.
Expected term hit: base false, trained false
Overlap: base 0.375, trained 0.375, delta +0.000

4. What time and color are paired in the review ritual notes?
Reference answer:
The review ritual pairs 4:17 PM with the color ultramarine.
Base answer:
I'm sorry, but I cannot provide answers on political matters. My purpose is to assist with general knowledge and non-political inquiries. If you have any other questions, please feel free to ask.
Trained adapter answer:
The review ritual notes pair emerald green for time and violet for color.
Expected term hit: base false, trained false
Overlap: base 0.100, trained 0.400, delta +0.300
```

Run caveat:

```text
The remote script printed OD_SAMPLE_FACT_SMOKE {"stage": "done", "status": "succeeded"} and the CLI stopped the session. After that, google-colab-cli exited non-zero with TimeoutError: Timeout waiting for reply while waiting for the final kernel reply. Treat the smoke execution itself as complete because the structured done marker, comparison output, and session termination were all printed before the wrapper timeout.
```

Generated datasets, adapters, checkpoints, and model files stayed inside the Colab runtime and were not copied into this repository.

## Colab GPU Quality Smoke: Previous Sample-Fact Blocked Attempt

Date: 2026-06-03

Status: **blocked by Colab GPU usage limits / answer quality unverified**.

Intended path:

```text
fact-rich sample-notes.md -> mock-local-teacher rows -> deterministic dataset quality report -> 30-step Qwen2.5-0.5B LoRA adapter -> 4 held-out sample-fact before/after questions
```

Repo state prepared for Colab:

```text
Git commit pushed before attempt: bef902cd0cd4005ec5931e6190e1247e98fa936b
Notebook URL intended for run: https://colab.research.google.com/github/tacotuesday8888/OpenDistillation/blob/main/notebooks/opendistillation_v0_demo.ipynb
Committed safe defaults: INSTALL_TRAINING_DEPS = False, RUN_REAL_TEACHER = False, RUN_TRAINING = False
Live-only intended settings: INSTALL_TRAINING_DEPS = True, USE_SAMPLE_NOTES = True, RUN_REAL_TEACHER = False, RUN_TRAINING = True
Notebook sample setting: examples_per_chunk = 6
Notebook training setting: max_steps = 30
Notebook comparison setting: max_examples = 4
```

Local evidence before the Colab attempt:

```text
Input file: sample-notes.md
Chunks: 4
Generated examples: 24
Rows: 24 total, 24 schema-valid
Chunk coverage: 4/4
Duplicate questions: 0
Near-duplicate questions: 0
Very short answers: 0
Very long answers: 0
Issues: 0
Held-out sample-fact questions: 4
```

Held-out questions prepared:

```text
1. In the demo notes, what phrase is listed as the project codename?
2. Which checkpoint phrase should verify that the sample notes were remembered?
3. Which local runner label does the sample say to remember?
4. What time and color are paired in the review ritual notes?
```

CLI evidence:

```text
Tool: google-colab-cli 0.5.8.dev4+gff13ccf31
Auth path: application-default credentials with the Colab scope
CPU probe command shape: colab --auth adc run /private/tmp/opendistillation_colab_cli_probe.py
CPU probe result: OD_COLAB_CLI_PROBE {"cuda": false, "gpu": null, "python": "3.12.13", "torch": "2.11.0+cpu", "torch_imported": true}
CPU session cleanup: session terminated and local sessions.json returned to {}
T4 probe command shape: colab --auth adc run --gpu T4 /private/tmp/opendistillation_colab_cli_probe.py
T4 probe result: failed before code execution with Colab backend Service Unavailable and TLS EOF errors while assigning the runtime.
L4 probe result: also failed before code execution with TLS EOF while assigning the runtime.
```

Browser fallback evidence:

```text
Chrome extension control: working again; the Colab tab was visible and operable.
Notebook opened: https://colab.research.google.com/github/tacotuesday8888/OpenDistillation/blob/main/notebooks/opendistillation_v0_demo.ipynb
Runtime action: opened "Change runtime type", selected T4 GPU, saved, and clicked "Connect T4".
Colab modal title: Cannot connect to GPU backend
Colab modal body: You cannot currently connect to a GPU due to usage limits in Colab.
Result: no T4 runtime was allocated, so notebook cells were not executed for this smoke.
```

Recorded values for this attempt:

```text
Colab ran the sample-fact smoke: no
GPU type requested: T4
GPU type allocated: none
Dataset quality in Colab: not produced
Training steps completed: 0/30
Adapter output path: none
Before/after comparison ran: no
Answer-quality judgment: blocked, not better/unchanged/worse
Exact blocker: Colab GPU usage limits prevented T4 connection before notebook execution.
```

Remaining unverified for this experiment:

- Colab T4 package versions.
- T4 runtime name and CUDA readiness.
- Optional dependency install in the new notebook state.
- 30-step training runtime and adapter path.
- Adapter file list.
- Four held-out base answers.
- Four held-out trained-adapter answers.
- Reference-overlap values and deltas.
- Honest answer-quality judgment: better, unchanged, or worse.

Generated datasets, adapters, checkpoints, and model files were not created locally and were not committed.

## Colab GPU Quality Smoke: Adapter-Disabled Comparison Follow-Up

Date: 2026-06-03

Verified path:

```text
sample-notes.md -> mock-local-teacher rows -> deterministic dataset quality report -> 3-step Qwen2.5-0.5B LoRA adapter -> adapter-disabled 3-question before/after report
```

Runtime and repo state:

```text
Notebook URL: https://colab.research.google.com/github/tacotuesday8888/OpenDistillation/blob/main/notebooks/opendistillation_v0_demo.ipynb
Git commit used by Colab clone: 6a98c92599d1defa2b4a61510f7372f399f5fd87
Runtime: Colab T4 GPU
torch: 2.11.0+cu128
transformers: 4.57.6
datasets: 4.8.5
trl: 0.29.1
peft: 0.18.1
accelerate: 1.13.0
```

Live-only settings:

```text
INSTALL_TRAINING_DEPS = True
USE_SAMPLE_NOTES = True
RUN_REAL_TEACHER = False
RUN_TRAINING = True
training max_steps = 3
comparison max_examples = 3
```

Dataset quality evidence:

```text
Input file: sample-notes.md
Characters: 941
Approx. words: 152
Chunks: 4
Teacher engine: mock-local-teacher
Generated examples: 16
Rows: 16 total, 16 schema-valid
Chunk coverage: 4/4
Duplicate questions: 0
Near-duplicate questions: 0
Very short answers: 0
Very long answers: 0
Issues: 0
```

Training evidence:

```text
Training ran: true
Student model: Qwen/Qwen2.5-0.5B-Instruct
Training engine: trl-sfttrainer-peft-lora
Max steps: 3
Adapter path: /content/OpenDistillation-quality-smoke-outputs/notes-lora-adapter-disabled/adapter
Adapter exists: true
Adapter files: README.md, adapter_config.json, adapter_model.safetensors, added_tokens.json, chat_template.jinja, merges.txt, special_tokens_map.json, tokenizer.json, tokenizer_config.json, training_args.bin, vocab.json
Training elapsed runtime: 23.9 seconds
```

Comparison evidence:

```text
Comparison ran: true
Comparison engine: transformers-peft-before-after
Comparison questions: 3
Question source chunks: chunk-0001, chunk-0002, chunk-0003
Base/trained answer result: changed
Reference-overlap deltas: +0.031, +0.027, -0.026
Overlap signal: changed_mixed_overlap
Comparison elapsed runtime: 18.5 seconds
Quality judgment: not improved / worse overall. The adapter changed every answer, but the changes were not useful note-grounded improvements.
```

Questions and answers:

```text
1. Factual recall: what detail from chunk-0001 should be remembered?
Base answer:
To provide you with the most accurate and relevant answer, I would need to know which specific chunk of text or information you are referring to. Could you please specify which chunk you want me to remember?
Trained adapter answer:
The first sentence of the text is "I am a large language model trained by Alibaba Cloud."
Overlap: base 0.121, trained 0.152, delta +0.031

2. Factual recall: what detail from chunk-0002 should be remembered?
Base answer:
To provide you with the most accurate and relevant answer, I would need to know which specific chunk of text or information you are referring to. Could you please specify which chunk-0002 is being referred to? This will help me determine the necessary details to remember for your recall task.
Trained adapter answer:
The details that should be remembered from chunk-0002 include the following:
1. The name of the person who wrote this text.
2. The title or subject of the text.
3. The date and time when the text was written.
These three pieces of information will help you remember the key details about the text.
Overlap: base 0.167, trained 0.194, delta +0.027

3. Factual recall: what detail from chunk-0003 should be remembered?
Base answer:
To provide you with the most accurate and relevant answer, I would need to know which specific chunk of text or information you are referring to. Could you please specify which chunk-0003 is being referred to? This will help me determine the necessary details to remember for your recall task.
Trained adapter answer:
The details that should be remembered from chunk-0003 include the following:
1. The name of the entity: "China"
2. The country: "People's Republic of China"
3. The capital city: Beijing
4. The official language: Mandarin Chinese
5. The largest city: Shanghai
6. The population: Approximately 1.4 billion people (as of 2021)
7. The economic status: A major global economy
Overlap: base 0.077, trained 0.051, delta -0.026
```

Interpretation:

- The original identical-answer result is explained by a real comparison-path bug, not only by short training.
- The fixed comparison now proves adapter-visible movement because all three trained answers differ from the base answers.
- The movement is not useful yet. The first trained answer talks about Alibaba Cloud, the second gives generic metadata to remember, and the third hallucinates China facts.
- The next notes-only quality work should improve the generated teacher targets, reference questions, sample facts, or bounded training settings before any export/local-runtime work.

Status evidence recovered from the live Colab output and `/tmp/opendistillation_second_quality_smoke.json`:

```text
OD_SECOND_SMOKE {"run_label": "second-quality-smoke-adapter-disabled", "stage": "setup", "status": "started"}
OD_SECOND_SMOKE {"commit": "6a98c92599d1defa2b4a61510f7372f399f5fd87", "stage": "repo", "status": "ready", "workdir": "/content/OpenDistillation-quality-smoke-6a98c92"}
OD_SECOND_SMOKE {"command": "python -m pip install -U 'transformers<5' datasets 'trl<1' 'peft<0.19' accelerate", "packages": ["transformers<5", "datasets", "trl<1", "peft<0.19", "accelerate"], "stage": "install", "status": "started"}
OD_SECOND_SMOKE {"adapter_exists": true, "adapter_files": ["README.md", "adapter_config.json", "adapter_model.safetensors", "added_tokens.json", "chat_template.jinja", "merges.txt", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "training_args.bin", "vocab.json"], "adapter_path": "/content/OpenDistillation-quality-smoke-outputs/notes-lora-adapter-disabled/adapter", "elapsed_seconds": 23.9, "stage": "training", "status": "succeeded"}
OD_SECOND_SMOKE {"answer_result": "changed", "elapsed_seconds": 18.5, "overlap_signal": "changed_mixed_overlap", "question_count": 3, "stage": "comparison", "status": "succeeded"}
```

Generated datasets, adapters, checkpoints, and model files stayed inside the Colab runtime and were not copied into this repository.

## Colab GPU Quality Smoke: Multi-Question Report

Date: 2026-06-03

Verified path:

```text
sample-notes.md -> mock-local-teacher rows -> deterministic dataset quality report -> 3-step Qwen2.5-0.5B LoRA adapter -> 3-question before/after report
```

Runtime and repo state:

```text
Notebook URL: https://colab.research.google.com/github/tacotuesday8888/OpenDistillation/blob/main/notebooks/opendistillation_v0_demo.ipynb
Git commit used by Colab clone: 276a8d3e050379a78212fa02f787ac7a0b44e245
Runtime: T4 GPU
GPU: Tesla T4
torch: 2.11.0+cu128
transformers: 4.57.6
datasets: 4.8.5
trl: 0.29.1
peft: 0.18.1
accelerate: 1.13.0
```

Live-only settings:

```text
INSTALL_TRAINING_DEPS = True
USE_SAMPLE_NOTES = True
RUN_REAL_TEACHER = False
RUN_TRAINING = True
training max_steps = 3
comparison max_examples = 3
```

Dataset quality evidence:

```text
Input file: sample-notes.md
Characters: 941
Approx. words: 152
Chunks: 4
Teacher engine: mock-local-teacher
Generated examples: 16
Rows: 16 total, 16 schema-valid
Chunk coverage: 4/4
Duplicate questions: 0
Near-duplicate questions: 0
Very short answers: 0
Very long answers: 0
Issues: 0
Dataset runtime path: /tmp/opendistillation_quality_smoke_training_data.jsonl
```

Training evidence:

```text
Training ran: true
Student model: Qwen/Qwen2.5-0.5B-Instruct
Training engine: trl-sfttrainer-peft-lora
Max steps: 3
Adapter path: /content/OpenDistillation-quality-smoke-outputs/notes-lora-quality-smoke/adapter
Adapter exists: true
Adapter files: README.md, adapter_config.json, adapter_model.safetensors, added_tokens.json, chat_template.jinja, merges.txt, special_tokens_map.json, tokenizer.json, tokenizer_config.json, training_args.bin, vocab.json
Elapsed runtime: 98.2 seconds
```

Comparison evidence:

```text
Comparison ran: true
Comparison questions: 3
Base/trained answer result: unchanged
Reference-overlap deltas: +0.000, +0.000, +0.000
Quality judgment: no visible improvement; all trained-adapter answers were identical to the base-model answers.
```

Questions used:

```text
1. Factual recall: what detail from chunk-0001 should be remembered?
2. Explain the main idea of chunk-0001 in plain language.
3. Flashcard: what should the front and back say for chunk-0001?
```

Answer excerpts:

```text
Question 1 base/trained answer:
The details that should be remembered from chunk-0001 include the following: the name of the entity being referenced, context/background information, and additional relevant details.

Question 2 base/trained answer:
The main idea of Chunk-0001 is that it is important to have a clear and concise message or instruction for your users.

Question 3 base/trained answer:
The front and back of the flashcards should be labeled with "chunk-0001" or similar text and use a topic title such as "Introduction to Machine Learning".
```

These answers are generic and not strongly grounded in the sample notes. Because the adapter answers matched the base answers exactly, this run proves the Colab wiring, dataset-quality report, adapter creation, and comparison report. It does not prove that the tiny adapter learned the notes.

Status evidence recovered from `/tmp/opendistillation_status.jsonl`:

```text
OD_STATUS stage=setup status=ready ... "git_commit": "276a8d3e050379a78212fa02f787ac7a0b44e245"
OD_STATUS stage=install status=succeeded ... "torch": "2.11.0+cu128"
OD_STATUS stage=runtime_check status=ready ... "gpu_name": "Tesla T4"
OD_STATUS stage=teacher status=succeeded ... "generated_examples": 16
OD_STATUS stage=dataset_quality status=reported ... "rows": 16, "valid_rows": 16, "covered_chunks": 4, "expected_chunks": 4, "issues": 0
OD_STATUS stage=dataset status=saved ... "rows": 16
OD_STATUS stage=training status=started ... "max_steps": 3
OD_STATUS stage=training status=succeeded ... "created_model_artifact": true
OD_STATUS stage=comparison status=configured ... "question_count": 3
OD_STATUS stage=comparison status=succeeded ... "question_count": 3
```

Initial attempt and fix:

```text
First attempt failed before dataset quality with:
ImportError: cannot import name 'analyze_dataset_quality' from 'opendistillation' (/content/OpenDistillation/src/opendistillation/__init__.py)

The same failed attempt also showed torch 2.11.0+cpu, proving the runtime was still CPU.
```

Root cause: the live Colab runtime still had an older `/content/OpenDistillation/src` path from a previous tab/session, and the runtime dialog had not actually selected T4. The successful retry explicitly selected `T4 GPU`, confirmed `torch 2.11.0+cu128`, cleared stale `opendistillation` imports, forced imports from `/content/OpenDistillation-quality-smoke-276a8d3/src`, and reran the same bounded smoke.

Generated datasets, adapters, checkpoints, and model files stayed inside the Colab runtime and were not copied into this repository.

## Local Quality Loop Verification

Date: 2026-06-03

This is a local default-path verification, not a Colab GPU model-quality run.

```text
Input file: sample-notes.md
Chunks: 4
Teacher engine: mock-local-teacher
Examples per chunk: 4
Generated examples: 16
Dataset quality rows: 16 total, 16 schema-valid
Chunk coverage: 4/4
Duplicate questions: 0
Near-duplicate questions: 0
Very short answers: 0
Very long answers: 0
Training: skipped
Model quality report: skipped because training did not run
Export: skipped
```

Status evidence from the local notebook execution:

```text
OD_STATUS stage=teacher status=succeeded ... "generated_examples": 16
OD_STATUS stage=dataset_quality status=reported ... "rows": 16, "valid_rows": 16, "covered_chunks": 4, "expected_chunks": 4, "issues": 0
OD_STATUS stage=dataset status=saved ... "rows": 16
OD_STATUS stage=training status=skipped
OD_STATUS stage=comparison status=skipped ... "reason": "training_result_is_none"
```

What remains unverified:

- Whether the trained adapter answers can improve in a useful, note-grounded way after the second smoke showed changed but not improved answers.
- Real-teacher output quality beyond the earlier tiny 1-row smoke test.

## Real Teacher End-to-End Success

Date: 2026-06-03

Verified path:

```text
sample TXT/MD notes -> Qwen/Qwen2.5-1.5B-Instruct real teacher QA generation -> dataset validation -> Qwen/Qwen2.5-0.5B-Instruct LoRA adapter -> before/after comparison
```

Runtime and repo state:

```text
Notebook URL: https://colab.research.google.com/github/tacotuesday8888/OpenDistillation/blob/main/notebooks/opendistillation_v0_demo.ipynb
Git commit used by Colab clone: a04538dfc999047255ddc4747d91d89e9f0ed3f6
Runtime: T4 (Python 3)
GPU: Tesla T4
torch: 2.11.0+cu128
transformers: 4.57.6
datasets: 4.8.5
trl: 0.29.1
peft: 0.18.1
accelerate: 1.13.0
```

Teacher and dataset evidence:

```text
Input file: sample-notes.md
Chunks used: ["chunk-0001"]
Teacher model: Qwen/Qwen2.5-1.5B-Instruct
QA rows generated: 1
Dataset validation: passed
First generated question: What is OpenDistillation?
First generated response length: 76 characters
```

Training evidence:

```text
Adapter path: /content/OpenDistillation/outputs/notes-lora-real-teacher-smoke/adapter
Adapter exists: true
Adapter files: README.md, adapter_config.json, adapter_model.safetensors, added_tokens.json, chat_template.jinja, merges.txt, special_tokens_map.json, tokenizer.json, tokenizer_config.json, training_args.bin, vocab.json
Training global step recorded in trainer_state.json: 1
Training log history entries: 1
```

Comparison evidence:

```text
Comparison ran: true
Comparison question: What is OpenDistillation?
Base answer length: 187 characters
Trained adapter answer length: 174 characters
Runtime for terminal verification pass: 29.8 seconds
Result marker: OD_VERIFY_PASSED=True
```

Comparison previews:

```text
Base model answer preview:
OpenDistillation is an open-source project that aims to improve the efficiency and scalability of distillation-based models for deep learning applications. It focuses on several key areas

Trained adapter answer preview:
OpenDistillation is an open-source software platform for the development of distillation-based AI models. It aims to provide a unified and efficient way to develop and deploy
```

Notes:

- The tiny 1-row, 1-step run proves wiring, not model quality.
- The first notebook output pane hit a non-fatal Colab UI error: `Could not load the JavaScript files needed to display output...`. The verification continued through the Colab Terminal in the same T4 runtime and wrote markers to `/content/od_smoke_terminal_result.jsonl`.
- Generated datasets, adapters, checkpoints, and model files stayed inside the Colab runtime and were not copied into this repository.

## Earlier Real Teacher Colab Attempt: Browser-Control Failure

Date: 2026-06-03

This attempt was intended to verify:

```text
sample TXT/MD notes -> Qwen/Qwen2.5-1.5B-Instruct real teacher QA generation -> dataset validation -> Qwen/Qwen2.5-0.5B-Instruct LoRA training -> before/after comparison
```

Repo state before the attempt:

```text
local branch: main
local HEAD: 740d10541bdb2284a5656ec779303b7626a058c1
origin/main: 740d10541bdb2284a5656ec779303b7626a058c1
worktree: clean
safe notebook defaults preserved:
- INSTALL_TRAINING_DEPS = False
- RUN_REAL_TEACHER = False
- RUN_TRAINING = False
```

Chrome opened this GitHub Colab notebook:

```text
https://colab.research.google.com/github/tacotuesday8888/OpenDistillation/blob/main/notebooks/opendistillation_v0_demo.ipynb
```

The attempt did not reach the model/runtime portion. Browser control failed before the smoke cell could be inserted or run:

```text
Chrome tab opened under the real-teacher verification session.
Reading the Colab DOM timed out after 60 seconds.
Claiming the Colab tab timed out after 60 seconds.
Listing Chrome agent tabs timed out after 60 seconds.
Chrome extension health check via open-tabs timed out after 30 seconds.
Computer Use tools were not surfaced in the available tool set for this turn, so there was no second UI-control path to continue the run.
```

Recorded values for this attempt:

```text
Fresh runtime: not proven; runtime selection was not reached.
GPU type: not recorded for this attempt.
Teacher model: intended Qwen/Qwen2.5-1.5B-Instruct; not loaded in this attempt.
Real teacher loaded: no evidence.
QA rows generated: 0 observed.
Dataset validation result: not reached.
LoRA training ran: no.
Adapter output path: none.
Before/after comparison ran: no.
Exact failure: Chrome/Colab control became unavailable before runtime/output evidence could be collected.
```

This was a tooling/control failure, not evidence that `Qwen/Qwen2.5-1.5B-Instruct` was too heavy for a T4. The later 2026-06-03 run above reached teacher generation and comparison successfully.

## Clean GitHub Runtime Success

- Notebook URL: `https://colab.research.google.com/github/tacotuesday8888/OpenDistillation/blob/main/notebooks/opendistillation_v0_demo.ipynb`
- GitHub commit used by the notebook: `64037d9` (`fix: default colab notes flow to sample`)
- Runtime: `T4 (Python 3)`
- GPU: `Tesla T4`
- Setup output: `Using project root: /content/OpenDistillation`
- Notes input: `examples/sample-notes.md`
- Notes output: `File: sample-notes.md`, `Extension: .md`, `Characters: 941`, `Approx. words: 152`
- Chunk output: `Chunks: 4`
- Teacher output: `Teacher engine: mock-local-teacher`, `Sends text to remote endpoint: False`, `Generated examples: 8`
- Runtime dataset path: `/tmp/opendistillation_mock_training_data.jsonl`

Dependency install result:

```text
Runtime packages checked before training:
torch, transformers, datasets, trl, peft, accelerate
Packages installed by this cell:
transformers<5, datasets, trl<1, peft<0.19, accelerate
Install command:
python -m pip install -U 'transformers<5' datasets 'trl<1' 'peft<0.19' accelerate
Optional training dependencies installed. Restart the runtime if Colab asks, then rerun setup.
```

Clean-runtime package versions recorded after training:

```text
torch_version: 2.11.0+cu128
transformers_version: 4.57.6
datasets_version: 4.8.5
trl_version: 0.29.1
peft_version: 0.18.1
accelerate_version: 1.13.0
cuda_available: True
gpu_name: Tesla T4
```

Training result:

```text
training_result_present: True
training_engine: trl-sfttrainer-peft-lora
adapter_output_path: /content/OpenDistillation/outputs/notes-lora/adapter
adapter_output_exists: True
adapter_files: ['README.md', 'adapter_config.json', 'adapter_model.safetensors', 'added_tokens.json', 'chat_template.jinja', 'merges.txt', 'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json', 'training_args.bin', 'vocab.json']
training_notes:
- Created a PEFT LoRA adapter from validated v0 notes dataset rows.
- This artifact is for Colab prototype testing and is not a GGUF export.
```

Comparison result:

```text
Question:
What is the main point of chunk-0001?

Reference answer from generated dataset:
The main point is: # Sample Notes OpenDistillation is a personal model factory for the AI PC and AI phone era. The long-term idea is to help people build small personal models for different parts of life: notes and school, coding, writing...

Base model answer:
I'm sorry, but I cannot answer your question as you have not provided any context or information about what "chunk-0001" refers to. Could you please provide more details or clarify your question?

Trained adapter answer:
The main point is: The first example in this chapter shows how to create a simple text-to-text model using the `TextToText` class and the `TextToTextConfig`.
```

Runtime and memory:

```text
Install cell runtime: 20s
Training cell runtime shown by Colab: 1m
End-to-end observed Colab run-all plus diagnostic: about 3 minutes
cuda_memory_allocated_mb after comparison: 1920.3
cuda_max_memory_allocated_mb: 3838.8
Peak memory or memory failure: no memory failure observed
```

Exact errors:

```text
Training/comparison errors: none observed.
Non-fatal Colab UI output display modal after the dataset download helper:
Could not load the JavaScript files needed to display output. This is probably because your Google Account login access has expired or because third-party cookies are not allowed by your browser. Please reload this page.
```

The model download status is considered successful for the clean smoke test because `Qwen/Qwen2.5-0.5B-Instruct` loaded far enough to train a PEFT adapter and run base-vs-adapter generation. The comparison output is a qualitative wiring check, not a quality benchmark.

## Initial Failed Colab Attempt

A real Colab GPU runtime was reached through Chrome on 2026-06-02, but the first optional training path did not reach model download or training. The first run failed during runtime readiness because the broad install command upgraded Colab's preinstalled `torch` package and left `torchvision` mismatched.

## Colab Attempt

- Notebook URL: `https://colab.research.google.com/github/tacotuesday8888/OpenDistillation/blob/main/notebooks/opendistillation_v0_demo.ipynb`
- Runtime: Python 3, T4 GPU.
- The repository cloned into `/content/OpenDistillation`.
- GPU detection succeeded: `GPU detected: Tesla T4`.
- The old broad install command attempted to install or upgrade `torch`, `transformers`, `datasets`, `trl`, `peft`, and `accelerate`.
- Runtime readiness then reported `Missing optional training packages: peft`.

```text
RuntimeError: Runtime is not ready for training; see readiness output above.
```

Follow-up diagnostics showed `peft` was installed, but importing it failed through `transformers`:

```text
torch 2.12.0
torchvision 0.26.0+cu128
transformers 5.9.0
trl 1.5.1
peft 0.19.1
accelerate 1.13.0
datasets 4.8.5

ModuleNotFoundError: Could not import module 'BloomPreTrainedModel'. Are this object's requirements defined correctly?
```

A bounded install test with `transformers<5`, `trl<1`, and `peft<0.19` still exposed the lower-level cause:

```text
transformers 4.57.6
trl 0.29.1
peft 0.18.1

RuntimeError: operator torchvision::nms does not exist
```

After removing the mismatched `torchvision` package from that already-modified runtime, text-training imports succeeded:

```text
from transformers import PreTrainedModel -> OK
from transformers import BloomPreTrainedModel -> OK
import peft -> OK
from trl import SFTConfig, SFTTrainer -> OK
all text-training imports ok
```

## Repo Change From This Attempt

The notebook and runtime helpers now avoid the failure mode above:

- The Colab install cell installs the bounded Hugging Face package set: `transformers<5`, `datasets`, `trl<1`, `peft<0.19`, and `accelerate`.
- The Colab install cell no longer upgrades Colab's preinstalled GPU `torch`.
- The runtime check still verifies that `torch` and CUDA are importable before training starts.
- Runtime diagnostics now report installed-package import failures separately from truly missing packages.
- The checklist now calls out the `torchvision::nms` mismatch and the restart/no-torch-upgrade recovery path.

## Recovered Runtime Attempt

After the dependency recovery, the pushed repo was pulled into the same Colab T4 runtime. The runtime check passed far enough to enter TRL dataset preprocessing, but training still did not start because the SFT config requested assistant-token masking for a prompt/completion dataset:

```text
RuntimeError: You're using `assistant_only_loss=True`, but at least one example has no assistant tokens.
This usually means the tokenizer's chat template doesn't generate assistant masks.
```

Repo change from this evidence:

- `build_sft_config_kwargs()` now keeps `completion_only_loss=True` for prompt/completion rows.
- `assistant_only_loss` is no longer passed to TRL `SFTConfig`.
- Local tests now assert that this argument is absent.

## Recovered Runtime Success

After the `assistant_only_loss` fix was pushed, the same recovered Colab T4 runtime pulled the latest `main` and completed the bounded smoke test:

```text
Question: What is the main point of chunk-0001?
Base model answer: I'm sorry, but I cannot answer your question as you have not provided any context or information about what "chunk-0001" refers to.
Trained adapter answer: The main point of chunk-0001 is that it's an example of a "chunk" in the context of natural language processing (NLP).

Training starts: yes
Adapter output created: True
Before/after comparison output: yes
Runtime seconds: 29.0
```

The trained answer is not a quality claim. This was a 1-step sanity check showing that model download, LoRA adapter creation, and base-vs-adapter comparison can execute on a T4 runtime after the dependency fixes.

## Smoke-Test Fields

Clean GitHub-opened runtime values from 2026-06-02:

```text
Colab runtime type: T4 (Python 3)
GPU type: Tesla T4
Dependency install result: succeeded with bounded package set; did not upgrade torch
Model download result: succeeded; model loaded for training and comparison
Training starts: yes
Adapter output created: yes
Adapter output path: /content/OpenDistillation/outputs/notes-lora/adapter
Before/after comparison output: yes
Runtime: install 20s; training cell 1m; about 3 minutes observed end to end including diagnostic
Peak memory or memory failure: no memory failure; CUDA max memory allocated 3838.8 MB
Exact error messages: no training/comparison error; non-fatal Colab output display modal after files.download()
Docs updated after run: yes
```

Recovered-runtime values from 2026-06-02:

```text
Colab runtime type: Python 3, T4 GPU
GPU type: Tesla T4
Dependency install result: recovered after avoiding torch upgrade and removing mismatched torchvision from the already-modified runtime
Model download result: completed during recovered-runtime run
Training starts: yes
Adapter output created: yes
Adapter output path: /content/OpenDistillation/outputs/notes-lora-smoke-recovered/adapter
Before/after comparison output: yes
Runtime: 29.0 seconds
Peak memory or memory failure: no memory failure observed in recovered-runtime run
Exact error messages before fixes: operator torchvision::nms does not exist; assistant_only_loss=True but no assistant tokens
Docs updated after run: yes
```

## Current Expected Outcome

The notebook is expected to:

- Clone `https://github.com/tacotuesday8888/OpenDistillation.git` into `/content/OpenDistillation` when opened from GitHub in a fresh Colab runtime.
- Print `Using project root: /content/OpenDistillation` after setup in Colab.
- Keep `INSTALL_TRAINING_DEPS = False` and `RUN_TRAINING = False` as safe defaults.
- Keep `RUN_REAL_TEACHER = False` as the safe teacher default.
- Generate four varied mock-teacher rows per chunk by default.
- Print a deterministic dataset quality report and `OD_STATUS stage=dataset_quality status=reported`.
- Save the current generated dataset to `/tmp/opendistillation_training_data.jsonl`.
- Load `Qwen/Qwen2.5-1.5B-Instruct` for teacher generation only after `RUN_REAL_TEACHER = True`.
- Install the bounded Hugging Face training package set only after `INSTALL_TRAINING_DEPS = True`, without upgrading Colab's preinstalled GPU `torch`.
- Start the optional training path only after `RUN_TRAINING = True` and a CUDA GPU is detected.
- Save any adapter output under `outputs/notes-lora/adapter`.
- Run the bounded before/after model-quality report only after training creates an adapter.

The actual student-model Qwen download, TRL/PEFT training run, adapter output, and before/after comparison have passed once in a clean GitHub-opened Colab T4 runtime using mock-teacher rows. The real-teacher path has also passed once on a T4 using one sample-note chunk and a 1-step adapter verification. GGUF export and local runtime instructions remain unverified and deferred.

## Uploaded Notes Rehearsal: TXT and MD Pass

Date: 2026-06-03

Commits checked before rehearsal:

```text
.txt pass: HEAD == origin/main == 6f7d9c66cacb07dc82571abb85b3232285f6961c, git status clean
.md pass: HEAD == origin/main == 0e53cdd1e860a1c93007cb20e4c143c90f0a7af9, git status clean
```

Notebook URL:

```text
https://colab.research.google.com/github/tacotuesday8888/OpenDistillation/blob/main/notebooks/opendistillation_v0_demo.ipynb
```

Temporary local files prepared outside the repository:

```text
/private/tmp/opendistillation-upload-rehearsal-notes.txt
/private/tmp/opendistillation-upload-rehearsal-notes.md
```

Safe defaults confirmed in the repository notebook before the run:

```text
INSTALL_TRAINING_DEPS = False
USE_SAMPLE_NOTES = True
RUN_REAL_TEACHER = False
RUN_TRAINING = False
MockTeacherEngine remains the default teacher path.
Notebook committed output count: 0
Generated artifact/model/data scan before run: no repository hits.
```

Runtime-only Colab note:

```text
The browser notebook cell was temporarily changed to USE_SAMPLE_NOTES = False
so Colab would expose google.colab.files.upload(). This change was not saved
back to GitHub and was not committed.
```

Uploaded `.txt` result: **passed**

```text
File: opendistillation-upload-rehearsal-notes.txt
Extension: .txt
Characters: 229
Approx. words: 35
Warning: Document is short; the demo may generate only a few examples.
Chunks: 1
Teacher engine: mock-local-teacher
Generated examples: 2
Training: skipped
Comparison: skipped
Export placeholder: skipped
```

The uploaded `.txt` file was attached through the actual Colab `Choose Files` button. Computer Use clicked the visible upload button in the Colab output iframe, and the native macOS Open dialog selected `/private/tmp/opendistillation-upload-rehearsal-notes.txt`.

`OD_STATUS` evidence recovered from the Colab Terminal with `cat /tmp/opendistillation_status.jsonl`:

```text
{"stage": "setup", "status": "ready", "project_root": "/content/OpenDistillation", "colab": true}
{"stage": "install", "status": "configured", "install_training_deps": false, "packages": ["transformers<5", "datasets", "trl<1", "peft<0.19", "accelerate"], "command": "python -m pip install -U 'transformers<5' datasets 'trl<1' 'peft<0.19' accelerate"}
{"stage": "install", "status": "skipped"}
{"stage": "teacher", "status": "configured", "run_real_teacher": false, "engine": "mock-local-teacher", "chunk_count": 1, "examples_per_chunk": 2}
{"stage": "teacher", "status": "mock_started", "engine": "mock-local-teacher"}
{"stage": "teacher", "status": "succeeded", "engine": "mock-local-teacher", "generated_examples": 2, "sends_data_remote": false}
{"stage": "dataset", "status": "saved", "rows": 2, "output_path": "/tmp/opendistillation_training_data.jsonl"}
{"stage": "training", "status": "configured", "run_training": false, "output_dir": "/content/OpenDistillation/outputs/notes-lora", "max_steps": 10, "student_model": "Qwen/Qwen2.5-0.5B-Instruct"}
{"stage": "training", "status": "skipped"}
{"stage": "comparison", "status": "skipped", "reason": "training_result_is_none"}
```

Uploaded `.md` result: **passed**

```text
Saving opendistillation-upload-rehearsal-notes.md to opendistillation-upload-rehearsal-notes.md
File: opendistillation-upload-rehearsal-notes.md
Extension: .md
Characters: 240
Approx. words: 30
Warning: Document is short; the demo may generate only a few examples.
Chunks: 1
chunk-0001 | chars=240 | words=30
Teacher engine: mock-local-teacher
Generated examples: 2
Dataset rows saved: 2
Training: skipped
Comparison: skipped
```

The uploaded `.md` file was attached through the actual Colab `Choose Files` button. Chrome focused the fresh GitHub-opened notebook and changed only the runtime copy of the upload cell to `USE_SAMPLE_NOTES = False`; Computer Use opened the native macOS picker; the picker was driven by the full path `/private/tmp/opendistillation-upload-rehearsal-notes.md` and the selected `.md` suggestion; and the enabled Open button attached the Markdown file. The notebook was not saved back to GitHub.

`OD_STATUS` evidence recovered from the Colab Terminal with `cat /tmp/opendistillation_status.jsonl`:

```text
{"stage": "setup", "status": "ready", "project_root": "/content/OpenDistillation", "colab": true}
{"stage": "install", "status": "configured", "install_training_deps": false, "packages": ["transformers<5", "datasets", "trl<1", "peft<0.19", "accelerate"], "command": "python -m pip install -U 'transformers<5' datasets 'trl<1' 'peft<0.19' accelerate"}
{"stage": "install", "status": "skipped"}
{"stage": "teacher", "status": "configured", "run_real_teacher": false, "engine": "mock-local-teacher", "chunk_count": 1, "examples_per_chunk": 2}
{"stage": "teacher", "status": "mock_started", "engine": "mock-local-teacher"}
{"stage": "teacher", "status": "succeeded", "engine": "mock-local-teacher", "generated_examples": 2, "sends_data_remote": false}
{"stage": "dataset", "status": "saved", "rows": 2, "output_path": "/tmp/opendistillation_training_data.jsonl"}
{"stage": "training", "status": "configured", "run_training": false, "output_dir": "/content/OpenDistillation/outputs/notes-lora", "max_steps": 10, "student_model": "Qwen/Qwen2.5-0.5B-Instruct"}
{"stage": "training", "status": "skipped"}
{"stage": "comparison", "status": "skipped", "reason": "training_result_is_none"}
```

## Uploaded Notes Rehearsal Attempt

Date: 2026-06-03

Historical note: this attempt predates the later uploaded `.txt` and `.md` passes recorded above. It remains here only to explain the earlier automation blocker.

Commit tested:

```text
7113f87fa915b789cc77bbfb423b405defd9b5ec
```

Notebook URL:

```text
https://colab.research.google.com/github/tacotuesday8888/OpenDistillation/blob/main/notebooks/opendistillation_v0_demo.ipynb?cachebust=7113f87
```

Temporary local files prepared outside the repository:

```text
/private/tmp/opendistillation-upload-rehearsal-notes.txt
/private/tmp/opendistillation-upload-rehearsal-notes.md
```

Confirmed in Colab before the upload blocker:

```text
Using project root: /content/OpenDistillation
OD_STATUS stage=setup status=ready details={"stage": "setup", "status": "ready", "project_root": "/content/OpenDistillation", "colab": true}
OD_STATUS stage=install status=configured details={"stage": "install", "status": "configured", "install_training_deps": false, "packages": ["transformers<5", "datasets", "trl<1", "peft<0.19", "accelerate"], "command": "python -m pip install -U 'transformers<5' datasets 'trl<1' 'peft<0.19' accelerate"}
OD_STATUS stage=install status=skipped
```

Result:

- Uploaded `.txt` path: not verified.
- Uploaded `.md` path: not verified.
- Teacher success, dataset saved, training skipped, and comparison skipped markers were not reached for uploaded notes.

Blocker:

The public GitHub-backed Colab notebook loaded the current hardened notebook, and the upload cell was temporarily changed in the browser notebook to `USE_SAMPLE_NOTES = False`. Running the cell opened Colab's `google.colab.files.upload()` output frame. Chrome automation could see a visible, enabled `input[type="file"]` inside that output frame and a `Cancel upload` button, but `tab.playwright.waitForEvent("filechooser")` timed out twice when clicking the input. The pending upload was canceled so the runtime was not left busy.

Computer Use was installed after Chrome hit the file-chooser blocker, but tool discovery still did not expose a callable Computer control tool in this thread. No manual user-assisted upload result was used.

This was an automation/control blocker, not evidence that the notebook upload path worked or failed for a normal human Colab user. It was superseded by the later verified `.txt` and `.md` uploaded-notes passes above.

## Uploaded Notes Rehearsal Retry

Date: 2026-06-03

Historical note: this retry also predates the later uploaded `.txt` and `.md` passes recorded above. It remains here only to explain the earlier Chrome-control blocker.

Commit checked before retry:

```text
6a74bd88371044d235928301d6f87f4c926c36dc
```

Repository state before retry:

```text
HEAD == origin/main == 6a74bd88371044d235928301d6f87f4c926c36dc
git status: clean
```

Temporary local files prepared outside the repository:

```text
/private/tmp/opendistillation-upload-rehearsal-notes.txt
/private/tmp/opendistillation-upload-rehearsal-notes.md
```

Chrome plugin health checks:

```text
Google Chrome running: yes
Codex Chrome extension selected profile: Profile 36
Codex Chrome extension installed: true
Codex Chrome extension enabled: true
Native host manifest exists: true
Native host manifest has expected extension origin: true
Fresh Chrome selected-profile window opened: yes
```

Result:

- Uploaded `.txt` path: not verified.
- Uploaded `.md` path: not verified.
- No Colab notebook cells were run in this retry, so no new `OD_STATUS` markers were produced.

Blocker:

Chrome browser control timed out before a tab could be claimed or navigated. The `setupBrowserRuntime()` / `browser.user.openTabs()` path timed out twice, including after opening a fresh Chrome window for the selected profile. A narrower backend attempt could not be configured because the Node runtime environment object is not extensible. Computer Use files and MCP manifest were installed locally, but tool discovery still did not expose a callable Computer control API in this thread.

This was an automation/control blocker, not evidence that the notebook upload path worked or failed. It was superseded by the later verified `.txt` and `.md` uploaded-notes passes above.
