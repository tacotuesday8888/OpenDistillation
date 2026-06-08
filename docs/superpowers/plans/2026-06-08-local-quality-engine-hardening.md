# Local Quality Engine Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the deterministic fact-ledger train/eval engine so bad data, leakage, and weak scoring are caught locally before any GPU run.

**Architecture:** Keep the public JSONL schema unchanged: `instruction`, `response`, and `source_chunk_id`. Improve only the local product layer: fact extraction, train/eval question generation, leakage checks, expected-term scoring, and beginner-readable reports.

**Tech Stack:** Python standard library, `unittest`, existing OpenDistillation helpers. No new runtime dependency for this pass unless standard-library checks cannot satisfy the tests.

---

### Task 1: Fact Extraction Tests

**Files:**
- Modify: `tests/test_fact_ledger.py`
- Modify: `src/opendistillation/fact_ledger.py`

- [x] **Step 1: Add a failing test for bullet/list facts**

Add a test that feeds chunks containing `Label: value`, `Label - value`, `Label = value`, numbered list items, and unsafe long prose. Assert that safe bullet/list pairs are extracted, unsafe prose is ignored, duplicate label/value pairs are collapsed, and the first existing sample fact stays stable.

- [x] **Step 2: Run the targeted test**

Run: `python3 -m unittest tests.test_fact_ledger.FactLedgerTests.test_extract_fact_ledger_captures_safe_bullet_list_facts -v`

Expected: fail because only `Label: value` extraction exists.

- [x] **Step 3: Implement extraction**

Add a small standard-library parser that only accepts explicit Markdown/TXT list facts with short labels and non-empty values. Keep `label_value` for colon facts and use a separate `list_pair` kind for dash/equal list facts.

- [x] **Step 4: Re-run the targeted test**

Run: `python3 -m unittest tests.test_fact_ledger.FactLedgerTests.test_extract_fact_ledger_captures_safe_bullet_list_facts -v`

Expected: pass.

### Task 2: Leakage And Scoring Tests

**Files:**
- Modify: `tests/test_fact_ledger.py`
- Modify: `src/opendistillation/fact_ledger.py`

- [x] **Step 1: Add failing leakage/scoring tests**

Add tests that require train questions and eval questions to use clearly different wording, catch token-overlap leaks even when sequence similarity is low, treat punctuation/case as harmless for expected terms, and reject partial word matches.

- [x] **Step 2: Run targeted tests**

Run: `python3 -m unittest tests.test_fact_ledger.FactLedgerTests -v`

Expected: fail on the new leakage/scoring expectations.

- [x] **Step 3: Implement stricter checks**

Add token Jaccard overlap beside `SequenceMatcher`, improve eval question templates, and make expected-term matching case-insensitive but token-boundary aware.

- [x] **Step 4: Re-run targeted tests**

Run: `python3 -m unittest tests.test_fact_ledger.FactLedgerTests -v`

Expected: pass.

### Task 3: Report And Docs

**Files:**
- Modify: `src/opendistillation/fact_ledger.py`
- Modify: `README.md`
- Modify: `docs/current-decisions.md`
- Modify: `docs/dataset-schema.md`
- Modify: `docs/next-goal-prompt.md`
- Create: `docs/open-source-tool-strategy.md`
- Optionally modify: `notebooks/README.md`
- Optionally modify: `notebooks/opendistillation_v0_demo.ipynb`

- [x] **Step 1: Improve report wording**

Make `format_fact_quality_report()` explain pass/fail in plain English: what was checked, what a leak means, and why expected terms matter.

- [x] **Step 2: Update docs**

Document which OSS tools are used now, later, or not yet. Keep the message beginner-readable and clear that no GPU/Colab training was run in this goal.

- [x] **Step 3: Update next goal prompt**

Set the next recommended goal to run a bounded Colab quality smoke using the hardened local gate.

### Task 4: Verification, Commit, Push

**Files:**
- Inspect all touched files.

- [x] **Step 1: Run unit tests**

Run: `python3 -m unittest discover -s tests -v`

- [x] **Step 2: Validate notebook JSON and outputs**

Run: `python3 -m json.tool notebooks/opendistillation_v0_demo.ipynb >/tmp/opendistillation_notebook.json`

Run: `python3 -m unittest tests.test_notebook -v`

- [x] **Step 3: Run default local notebook smoke**

Run the existing CPU/default notebook smoke path without installs, GPU, model downloads, or training. If no dedicated command exists, use the closest existing local tests that exercise notebook helpers and state the limitation.

- [x] **Step 4: Run repository hygiene checks**

Run `git diff --check`, a secret scan over the diff, an artifact/model/data scan over git status, and `git status --short --branch`.

- [ ] **Step 5: Commit and push**

Inspect the staged diff, stage only intended files, commit to `main`, and push `main`.
