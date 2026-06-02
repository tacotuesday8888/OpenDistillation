# Colab GPU Smoke-Test Results

Date: 2026-06-02

## Result

Real Colab GPU training is **not verified yet**.

A real Colab GPU runtime was reached through Chrome on 2026-06-02, but the optional training path did not reach model download or training. The first run failed during runtime readiness because the broad install command upgraded Colab's preinstalled `torch` package and left `torchvision` mismatched.

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

## Smoke-Test Fields

Fill these in after running `docs/colab-smoke-test-checklist.md` in Colab:

```text
Colab runtime type:
GPU type:
Dependency install result:
Model download result:
Training starts: yes/no
Adapter output created: yes/no
Adapter output path:
Before/after comparison output: yes/no
Runtime:
Peak memory or memory failure:
Exact error messages:
Docs updated after run:
```

## Current Expected Outcome

The notebook is expected to:

- Clone `https://github.com/tacotuesday8888/OpenDistillation.git` into `/content/OpenDistillation` when opened from GitHub in a fresh Colab runtime.
- Print `Using project root: /content/OpenDistillation` after setup in Colab.
- Keep `INSTALL_TRAINING_DEPS = False` and `RUN_TRAINING = False` as safe defaults.
- Install the bounded Hugging Face training package set only after `INSTALL_TRAINING_DEPS = True`, without upgrading Colab's preinstalled GPU `torch`.
- Start the optional training path only after `RUN_TRAINING = True` and a CUDA GPU is detected.
- Save any adapter output under `outputs/notes-lora/adapter`.
- Run before/after comparison only after training creates an adapter.

The actual Qwen download, TRL/PEFT training run, adapter output, and before/after comparison remain unverified until a real Colab GPU run records results here.
