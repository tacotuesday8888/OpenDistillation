# Fact-Ledger Label/Value Signal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strengthen the local fact-ledger training signal so TXT/MD notes produce clearer label/value SFT rows, safer held-out eval metadata, and honest exact fact-hit reporting before another GPU run.

**Architecture:** Keep the public JSONL row schema unchanged and concentrate the upgrade in `src/opendistillation/fact_ledger.py`. The fact ledger remains the internal source of truth for labels, values, expected terms, train/eval split metadata, leakage checks, SFT preview metadata, and exact scoring reports.

**Tech Stack:** Python standard library, existing OpenDistillation dataset validation, Colab-compatible TRL/PEFT row shape through existing training helpers.

---

## Task 1: Strengthen Fact Rows And Readiness

**Files:**
- Modify: `src/opendistillation/fact_ledger.py`
- Modify: `src/opendistillation/__init__.py`
- Test: `tests/test_fact_ledger.py`

- [x] Write failing tests for six-row default label/value training rows, manifest integrity checks, and a plain-language readiness report.
- [x] Run the targeted fact-ledger tests and confirm they fail for missing behavior.
- [x] Add a fact-ledger default row count constant, richer label/value templates, manifest coverage checks, and readiness formatting.
- [x] Re-run targeted tests and confirm they pass.

## Task 2: Add Metadata-Aware Exact Scoring

**Files:**
- Modify: `src/opendistillation/fact_ledger.py`
- Modify: `src/opendistillation/__init__.py`
- Test: `tests/test_fact_ledger.py`

- [x] Write failing tests for preserved scoring metadata, unscored empty expected terms, and per-fact learned/missed/unchanged/worse outcomes.
- [x] Run the targeted tests and confirm they fail for missing behavior.
- [x] Extend score dataclasses and report formatting without changing comparison inputs.
- [x] Re-run targeted tests and confirm they pass.

## Task 3: Wire Notebook Defaults

**Files:**
- Modify: `notebooks/opendistillation_v0_demo.ipynb`
- Modify: `tests/test_notebook.py`

- [x] Write failing notebook skeleton expectations for stronger fact-ledger defaults and readiness output.
- [x] Update notebook imports and cells to use the stronger default and print the readiness report before training.
- [x] Validate notebook JSON and notebook skeleton tests.

## Task 4: Final Verification

**Files:**
- No new production files.

- [x] Run full unit tests.
- [x] Validate notebook JSON.
- [x] Run `git diff --check`.
- [x] Scan for secrets and generated model/data artifacts.
- [x] Inspect `git status` and the final diff before reporting.
