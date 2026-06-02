# Colab GPU Smoke-Test Results

Date: 2026-06-02

## Result

Real Colab GPU training is **not verified yet** from this Codex workspace.

No Colab notebook cells were executed in a fresh GPU runtime from this environment. The available local environment does not provide an authenticated Colab runtime, a Colab CLI, or browser automation capable of running the notebook in Google Colab.

## Evidence From This Environment

Local checks performed before falling back to manual documentation:

```text
command -v google-chrome -> exit 1, no executable found
command -v chromium -> exit 1, no executable found
command -v jupyter -> exit 1, no executable found
command -v colab -> exit 1, no executable found
```

Node browser automation package checks:

```text
playwright -> Module not found: playwright
puppeteer -> Module not found: puppeteer
@playwright/test -> Module not found: @playwright/test
```

Because those checks do not provide access to a fresh Colab GPU runtime, this result file records the blocker instead of pretending the GPU path was tested.

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
- Install `torch`, `transformers`, `datasets`, `trl`, `peft`, and `accelerate` only after `INSTALL_TRAINING_DEPS = True`.
- Start the optional training path only after `RUN_TRAINING = True` and a CUDA GPU is detected.
- Save any adapter output under `outputs/notes-lora/adapter`.
- Run before/after comparison only after training creates an adapter.

The actual Qwen download, TRL/PEFT training run, adapter output, and before/after comparison remain unverified until a real Colab GPU run records results here.
