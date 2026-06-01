# OpenDistillation Project Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the initial local open-source project folder and documentation foundation for OpenDistillation.

**Architecture:** Start with documentation and repository structure before implementation. The project will be Colab-first, with a future Python package and thin CLI sharing the same underlying workflow.

**Tech Stack:** Markdown, Git, future Python package, future Colab notebook, future Hugging Face/Unsloth/PEFT/TRL/llama.cpp integrations.

---

### Task 1: Create Project Skeleton

**Files:**
- Create: `README.md`
- Create: `docs/product-vision.md`
- Create: `docs/roadmap.md`
- Create: `notebooks/.gitkeep`
- Create: `src/opendistillation/.gitkeep`
- Create: `.gitignore`

- [ ] **Step 1: Add README with product positioning**

Create a README that explains OpenDistillation as an open-source productized tool, not a closed SaaS or research-only framework.

- [ ] **Step 2: Add product vision**

Create `docs/product-vision.md` with the target user, core promise, product shape, and long-term direction.

- [ ] **Step 3: Add roadmap**

Create `docs/roadmap.md` with phase 0 through phase 4.

- [ ] **Step 4: Add empty implementation directories**

Create `notebooks/` and `src/opendistillation/` with `.gitkeep` files so the intended structure is visible before code exists.

- [ ] **Step 5: Initialize Git and commit**

Run:

```bash
git init
git add README.md docs/product-vision.md docs/roadmap.md docs/superpowers/plans/2026-06-01-project-bootstrap.md notebooks/.gitkeep src/opendistillation/.gitkeep .gitignore
git commit -m "chore: bootstrap OpenDistillation project"
```

Expected result: a clean local repository with one bootstrap commit.

### Task 2: Decide GitHub Repository Creation

**Files:**
- No file changes required.

- [ ] **Step 1: Confirm repository visibility**

Use a public GitHub repository if the user wants early open-source visibility. Use a private repository if the user wants to keep positioning and README drafts hidden until the first prototype exists.

- [ ] **Step 2: Create remote repository**

After visibility is chosen, create the GitHub repository named `OpenDistillation`.

- [ ] **Step 3: Push local main branch**

Run:

```bash
git remote add origin <github-url>
git branch -M main
git push -u origin main
```

Expected result: GitHub has the initial OpenDistillation repository.
