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
- The v0 skeleton uses a deterministic local `MockTeacherEngine` before any real teacher-model path is chosen.
- The notebook should call small helper interfaces so real engines and future model types can be plugged in later without changing the first flow.
- The first student model is `Qwen/Qwen2.5-0.5B-Instruct`.
- The first training backend is Hugging Face TRL `SFTTrainer` with PEFT LoRA adapters.
- The first training path is optional in the notebook and defaults to skipped. It should run only after the user opts into a Colab GPU runtime.
- The first training path does not use Unsloth or bitsandbytes by default. Those are future optimizations after the plain TRL/PEFT path is smoke-tested.
- The first before/after comparison uses the first generated dataset question, Hugging Face Transformers chat generation, and PEFT `PeftModel.from_pretrained()` to load the trained LoRA adapter.
- The notebook includes an explicit `INSTALL_TRAINING_DEPS = False` switch so the default local path never installs packages, downloads models, or starts training by accident.
- The optional Colab install command does not upgrade Colab's preinstalled GPU `torch` package. The runtime still checks that `torch` and CUDA are available before training starts.
- The notebook setup clones the GitHub repository in fresh Colab runtimes before importing local helpers.
- Runtime readiness checks should explain missing optional packages, installed-package import failures, missing CUDA GPU, adapter-path problems, and likely GPU memory failures in plain language.
- The manual Colab GPU smoke test should use `docs/colab-smoke-test-checklist.md` before the optional training/comparison path is called verified.

## Not Decided Yet

- Exact teacher model or hosted teacher path.
- Exact dataset schema fields beyond the required v0 `instruction`, `response`, and `source_chunk_id`.
- Whether GGUF export is implemented in v0 or documented as the immediate next command.
- Which future model profile comes after the notes / school model.
- GitHub repository visibility and remote URL.

## Working Recommendation

Use this notes-model v0 flow as the default implementation plan:

1. User opens the Colab notebook.
2. User uses the sample notes file or uploads one `.txt` or `.md` notes file.
3. The notebook validates and previews the text.
4. OpenDistillation chunks the notes.
5. The mock teacher generates question-answer pairs in the current skeleton.
6. A future real teacher path replaces the mock teacher behind the same interface.
7. The dataset is previewed, saved, and downloadable.
8. Optional training prepares `Qwen/Qwen2.5-0.5B-Instruct` with TRL `SFTTrainer` and PEFT LoRA.
9. A short supervised fine-tuning run starts only when the user sets `RUN_TRAINING = True` in a Colab GPU runtime.
10. The notebook compares the base-model answer and trained-adapter answer for one generated question.
11. The notebook saves the output.
12. The notebook exports to GGUF or shows the exact export command and limitation.
13. The user gets local run instructions.

Do not build coding, writing, work, or phone model flows until the notes model path works end to end.

## Verified Versus Deferred

Verified locally in this repository:

- Dataset validation and JSONL serialization.
- Mock teacher generation.
- Training configuration, request construction, and TRL/PEFT dataset formatting tests.
- Before/after comparison request construction, adapter-path validation, dependency handling, and fake base-vs-adapter generation tests.
- Optional runtime helper behavior for install-command text, missing-package checks, GPU/no-GPU formatting, and beginner-readable failure explanations.
- Optional runtime helper behavior for installed-package import failures, including the Colab `torchvision::nms` mismatch seen after upgrading `torch`.
- Notebook JSON parsing and the default CPU path where training remains skipped.
- Notebook default install path where `INSTALL_TRAINING_DEPS = False`.
- Notebook setup structure for the fresh Colab clone fallback.
- Ignore rules for generated datasets, adapters, checkpoints, model weights, Hugging Face caches, notebook checkpoints, and common trainer artifacts.
- Recovered Colab T4 run after dependency fixes: model download, 1-step LoRA training, adapter creation, and before/after comparison completed once.

Requires a clean/fresh Colab GPU smoke test:

- Installing the optional Hugging Face training packages from a clean GitHub-opened notebook without upgrading Colab's preinstalled GPU `torch`.
- Confirming the runtime check prints the Colab GPU name from that clean runtime.
- Repeating `SFTLoRATrainingEngine.train()` and the before/after comparison from that clean runtime.
- Confirming adapter quality, comparison output, runtime, and memory use from the clean runtime.
- `docs/colab-smoke-test-results.md` currently records the first real Colab T4 attempt, two blockers, and one recovered-runtime pass.

Why this path:

- Qwen's model card lists `Qwen/Qwen2.5-0.5B-Instruct` as a 0.5B-parameter instruction model supported by current Hugging Face Transformers.
- TRL documents `SFTTrainer` for supervised fine-tuning and prompt/completion datasets.
- TRL documents direct PEFT adapter training through `peft_config=LoraConfig()`.
- PEFT documents loading saved adapters with `PeftModel.from_pretrained()` for inference.
- Transformers documents chat generation with `apply_chat_template()` and `generate()`.
- PEFT LoRA keeps the trained artifact small.
- bitsandbytes and Unsloth are useful memory/speed tools, but they introduce more install, quantization, and hardware assumptions than the first beginner path needs.

Sources checked on 2026-06-02:

- [Qwen/Qwen2.5-0.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [TRL SFTTrainer docs](https://huggingface.co/docs/trl/en/sft_trainer)
- [PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora)
- [PEFT PeftModel docs](https://huggingface.co/docs/peft/package_reference/peft_model)
- [Transformers Trainer docs](https://huggingface.co/docs/transformers/trainer)
- [Transformers text generation docs](https://huggingface.co/docs/transformers/llm_tutorial)
- [Accelerate docs](https://huggingface.co/docs/accelerate/index)
- [bitsandbytes docs](https://huggingface.co/docs/bitsandbytes/index)
- [Unsloth fine-tuning guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide)
