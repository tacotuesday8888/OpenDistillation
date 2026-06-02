# Colab GPU Smoke-Test Results

Date: 2026-06-02

## Result

Fresh Colab GPU training from a clean GitHub-opened runtime is **verified once** on 2026-06-02.

The clean run passed after three fixes were pushed:

- Do not upgrade Colab's preinstalled GPU `torch` package.
- Do not pass `assistant_only_loss=True` for TRL prompt/completion rows.
- Load `examples/sample-notes.md` by default in Colab so the first smoke path does not block on a file picker.

The clean run used the GitHub notebook at `main`, a fresh T4 runtime, `INSTALL_TRAINING_DEPS = True`, `USE_SAMPLE_NOTES = True`, and `RUN_TRAINING = True`. It installed the bounded Hugging Face package set without upgrading `torch`, loaded the sample notes, created mock QA examples, trained a LoRA adapter, and printed before/after answers.

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

The actual student-model Qwen download, TRL/PEFT training run, adapter output, and before/after comparison have passed once in a clean GitHub-opened Colab T4 runtime. The optional real teacher implementation exists locally but still needs a clean Colab GPU smoke test. GGUF export and local runtime instructions remain unverified and deferred.
