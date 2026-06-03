# Colab GPU Smoke-Test Results

Last updated: 2026-06-03

## Result

Fresh Colab GPU training from a clean GitHub-opened runtime is **verified once** on 2026-06-02.

The optional real-teacher Colab path is **verified once** on 2026-06-03. The verified path used sample TXT/MD notes, `Qwen/Qwen2.5-1.5B-Instruct` as the local real teacher, dataset validation, a 1-step TRL/PEFT LoRA adapter from `Qwen/Qwen2.5-0.5B-Instruct`, and before/after comparison on a Tesla T4 runtime.

The uploaded-notes Colab rehearsal is **partially verified** on 2026-06-03: one uploaded `.txt` file passed the default mock-teacher path end to end with status-log evidence; the uploaded `.md` path is still blocked at file attachment and is not verified.

An earlier 2026-06-03 attempt failed before model execution because Chrome/Colab control timed out. That older result is kept below as a tooling note, but it is superseded by the successful real-teacher run recorded here.

The clean run passed after three fixes were pushed:

- Do not upgrade Colab's preinstalled GPU `torch` package.
- Do not pass `assistant_only_loss=True` for TRL prompt/completion rows.
- Load `examples/sample-notes.md` by default in Colab so the first smoke path does not block on a file picker.

The clean run used the GitHub notebook at `main`, a fresh T4 runtime, `INSTALL_TRAINING_DEPS = True`, `USE_SAMPLE_NOTES = True`, and `RUN_TRAINING = True`. It installed the bounded Hugging Face package set without upgrading `torch`, loaded the sample notes, created mock QA examples, trained a LoRA adapter, and printed before/after answers.

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
- Save the current generated dataset to `/tmp/opendistillation_training_data.jsonl`.
- Load `Qwen/Qwen2.5-1.5B-Instruct` for teacher generation only after `RUN_REAL_TEACHER = True`.
- Install the bounded Hugging Face training package set only after `INSTALL_TRAINING_DEPS = True`, without upgrading Colab's preinstalled GPU `torch`.
- Start the optional training path only after `RUN_TRAINING = True` and a CUDA GPU is detected.
- Save any adapter output under `outputs/notes-lora/adapter`.
- Run before/after comparison only after training creates an adapter.

The actual student-model Qwen download, TRL/PEFT training run, adapter output, and before/after comparison have passed once in a clean GitHub-opened Colab T4 runtime using mock-teacher rows. The real-teacher path has also passed once on a T4 using one sample-note chunk and a 1-step adapter verification. GGUF export and local runtime instructions remain unverified and deferred.

## Uploaded Notes Rehearsal: TXT Pass, MD Blocked

Date: 2026-06-03

Commit checked before rehearsal:

```text
HEAD == origin/main == 6f7d9c66cacb07dc82571abb85b3232285f6961c
git status: clean
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

Uploaded `.md` result: **not verified**

The `.md` rehearsal reached a fresh setup/install-skipped state and the actual Colab upload widget, but the Markdown file could not be attached. Observed blocker:

```text
The native Open dialog showed /private/tmp/opendistillation-upload-rehearsal-notes.md,
but the Markdown row was not exposed as a clickable file element and the Open
button stayed disabled. Computer Use could click the Colab Choose Files button
and operate visible dialog controls, but could not complete the .md selection.
Chrome's file-chooser listener returned no usable chooser object for this Colab
output iframe, so setFiles() could not be used as a fallback.
The waiting upload cell was interrupted and ended with KeyboardInterrupt.
```

No `.md` validation, chunking, teacher, dataset, training-skipped, or comparison-skipped evidence was produced. This is still an upload-control blocker, not evidence that `.md` parsing fails after a Markdown file is loaded.

## Uploaded Notes Rehearsal Attempt

Date: 2026-06-03

Historical note: this attempt predates the later uploaded `.txt` pass recorded above. It remains here only to explain the earlier automation blocker.

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

This is an automation/control blocker, not evidence that the notebook upload path works or fails for a normal human Colab user. The uploaded-notes path remains unverified until a browser-control path can attach the `.txt` and `.md` files and the notebook reaches teacher, dataset, training-skipped, and comparison-skipped markers.

## Uploaded Notes Rehearsal Retry

Date: 2026-06-03

Historical note: this retry also predates the later uploaded `.txt` pass recorded above. It remains here only to explain the earlier Chrome-control blocker.

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

This remains an automation/control blocker, not evidence that the notebook upload path works or fails. The uploaded-notes path remains unverified until Chrome can attach a local file to Colab's `files.upload()` widget or an actually callable Computer control tool can operate the native file picker.
