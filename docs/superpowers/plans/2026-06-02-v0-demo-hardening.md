# V0 Demo Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing OpenDistillation v0 notes-model notebook clearer and safer for both the default local path and the optional Colab GPU training/comparison path.

**Architecture:** Keep the product scope fixed: TXT/MD notes, `MockTeacherEngine`, one Qwen student model, one TRL/PEFT backend, and one before/after comparison. Add small lazy-loaded runtime helper functions for dependency/GPU checks and beginner failure messages, then wire them into the notebook without importing heavy ML packages on the local default path.

**Tech Stack:** Python standard library, unittest, Jupyter notebook JSON, Hugging Face Transformers, TRL `SFTTrainer`, PEFT LoRA/PeftModel, Accelerate, Colab.

---

### Task 1: Runtime Readiness Helpers

**Files:**
- Create: `src/opendistillation/runtime.py`
- Modify: `src/opendistillation/__init__.py`
- Test: `tests/test_runtime.py`

- [x] **Step 1: Write failing runtime tests**

Add tests for:

```python
from opendistillation.runtime import (
    OPTIONAL_TRAINING_PACKAGES,
    build_pip_install_command,
    check_training_runtime,
    explain_runtime_failure,
    format_runtime_check,
)
```

Expected behaviors:

- Importing `opendistillation.runtime` does not import `torch`, `transformers`, `datasets`, `trl`, `peft`, or `accelerate`.
- `build_pip_install_command()` returns `python -m pip install -U torch transformers datasets trl peft accelerate`.
- Missing packages are reported with that install command.
- A fake CUDA runtime reports ready and includes the GPU name.
- A fake no-CUDA runtime reports that Colab users should switch to GPU.
- `explain_runtime_failure(RuntimeError("CUDA out of memory"))` tells the user to restart the runtime or reduce `max_steps`/`max_length`.

- [x] **Step 2: Run the runtime tests and verify red**

Run:

```bash
python3 -m unittest tests/test_runtime.py
```

Expected: fail because `opendistillation.runtime` does not exist yet.

- [x] **Step 3: Implement minimal runtime helpers**

Create `runtime.py` with:

- `OPTIONAL_TRAINING_PACKAGES`
- `OPTIONAL_COMPARISON_PACKAGES`
- `RuntimeCheck`
- `build_pip_install_command()`
- `check_training_runtime(importer=import_module)`
- `format_runtime_check(check)`
- `explain_runtime_failure(exc)`

- [x] **Step 4: Run runtime tests and verify green**

Run:

```bash
python3 -m unittest tests/test_runtime.py
```

Expected: pass.

### Task 2: Ignore Rules For Demo Artifacts

**Files:**
- Modify: `.gitignore`
- Test: `tests/test_gitignore.py`

- [x] **Step 1: Write failing ignore-rule test**

Assert `.gitignore` includes:

- `outputs/`
- `adapters/`
- `hf_cache/`
- `huggingface_cache/`
- `.cache/`
- `events.out.tfevents*`
- `trainer_state.json`
- model-weight patterns already present.

- [x] **Step 2: Run the ignore test and verify red**

Run:

```bash
python3 -m unittest tests/test_gitignore.py
```

Expected: fail until the missing ignore patterns are added.

- [x] **Step 3: Add missing ignore patterns**

Update `.gitignore` under generated data / caches.

- [x] **Step 4: Run ignore test and verify green**

Run:

```bash
python3 -m unittest tests/test_gitignore.py
```

Expected: pass.

### Task 3: Notebook Runtime Hardening

**Files:**
- Modify: `notebooks/opendistillation_v0_demo.ipynb`
- Modify: `tests/test_notebook.py`

- [x] **Step 1: Write failing notebook tests**

Assert the notebook contains:

- `INSTALL_TRAINING_DEPS = False`
- `OPTIONAL_TRAINING_PACKAGES`
- `check_training_runtime`
- `format_runtime_check`
- `explain_runtime_failure`
- `Manual Colab smoke-test checklist`
- `RUN_TRAINING = False`

- [x] **Step 2: Run notebook test and verify red**

Run:

```bash
python3 -m unittest tests/test_notebook.py
```

Expected: fail until notebook cells are updated.

- [x] **Step 3: Update notebook JSON**

Add:

- A dependency-install cell that is skipped by default.
- A runtime-readiness section that checks optional packages and CUDA only when `RUN_TRAINING = True`.
- Beginner-friendly exception printing around training and comparison.
- A manual Colab smoke-test checklist markdown cell.

- [x] **Step 4: Run notebook test and verify green**

Run:

```bash
python3 -m unittest tests/test_notebook.py
```

Expected: pass.

### Task 4: Documentation Alignment

**Files:**
- Create: `docs/colab-smoke-test-checklist.md`
- Modify: `docs/current-decisions.md`
- Modify: `docs/first-demo-implementation-plan.md`
- Modify: `docs/first-demo-flow.md`
- Modify: `docs/github-issue-plan.md`
- Modify: `docs/github-launch-checklist.md`
- Modify: `notebooks/README.md`
- Modify: `README.md` if public status needs a wording correction

- [x] **Step 1: Add manual Colab checklist**

Document how to record:

- Install success.
- GPU name.
- Model download.
- Training start.
- Adapter output path.
- Comparison output.
- Runtime.
- Memory failures.

- [x] **Step 2: Update verified/deferred docs**

State that local verification covers tests and the skipped notebook path. State that real Colab GPU training/comparison remains unverified unless a future smoke test records evidence.

### Task 5: Final Verification, Commit, Push

**Files:**
- All changed files

- [x] **Step 1: Run verification**

Run:

```bash
python3 -m unittest discover -s tests
python3 -m json.tool notebooks/opendistillation_v0_demo.ipynb >/dev/null
python3 - <<'PY'
import json
from pathlib import Path

notebook = json.loads(Path('notebooks/opendistillation_v0_demo.ipynb').read_text(encoding='utf-8'))
namespace = {'__name__': '__main__'}
for index, cell in enumerate(notebook['cells']):
    if cell.get('cell_type') != 'code':
        continue
    source = ''.join(cell.get('source', []))
    print(f'Executing code cell {index}')
    exec(compile(source, f'notebook-cell-{index}', 'exec'), namespace)
print('Notebook default code path completed')
PY
git diff --check
```

- [x] **Step 2: Run scans**

Run artifact and secret scans from the working agreements.

- [x] **Step 3: Review diff**

Run:

```bash
git status --short --branch --untracked-files=all
git diff --stat
git diff
```

- [x] **Step 4: Commit and push**

Run:

```bash
git add <intended files>
git commit -m "feat: harden v0 demo runtime flow"
git push origin main
```

---

## Self-Review

- Spec coverage: plan covers repo inspection, HF checks, notebook clarity, local default path, Colab opt-in path, dependency/runtime checks, failure messages, ignored artifacts, manual Colab checklist, docs status, tests, scans, commit, and push.
- Placeholder scan: no `TBD` or unspecified implementation steps.
- Type consistency: runtime helper names are reused consistently across tests, notebook, exports, and docs.
