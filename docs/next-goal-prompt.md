# Recommended Next Goal Prompt

Use this as the next `/goal` after the clean Colab GPU smoke-test results are committed:

```text
/goal Choose and prototype the first real teacher-generation path for the OpenDistillation v0 notes model. Work in /Users/langqi/Developer/Projects/OpenDistillation on the current branch. Keep the product scope narrow: TXT/MD notes only, one notes/school model only, Colab-first, deterministic MockTeacherEngine as the safe fallback. Check current official/reliable docs before choosing a teacher model or package path. Do not build a SaaS, Mac app, phone app, account system, cloud backend, multi-profile system, GGUF export, local runtime, or real large training pipeline. Define the teacher path, document whether text stays local or is sent remote, add the smallest prototype hook behind the existing teacher interface, keep generated datasets/model artifacts/secrets out of git, add focused tests, update README/docs/current-decisions.md/docs/first-demo-flow.md/docs/first-demo-implementation-plan.md/docs/github-issue-plan.md with verified versus unverified status, run verification, review the diff for secrets/artifacts, commit locally, and push.
```

## Why This Goal

The clean GitHub-opened Colab T4 smoke test now verifies the optional sample-notes training and before/after comparison path once. The next useful product risk is replacing or supplementing the deterministic mock teacher with one real notes teacher path.

## Done Means

- One real teacher path is selected with current docs checked.
- The docs say plainly whether notes text stays local or is sent to a remote endpoint.
- `MockTeacherEngine` remains available and safe.
- The prototype is behind the existing teacher interface.
- Tests cover the new teacher-selection or request-building behavior without requiring real model downloads by default.
- Real teacher execution is either verified with evidence or clearly marked as still unverified.
- Real training, GGUF export, local runtime, and future model profiles remain deferred.
- No generated datasets, model artifacts, checkpoints, secrets, or local config are committed.

## Do Not Use This Goal For

- Expanding beyond TXT/MD notes.
- Adding coding, writing, work, phone, or multi-profile flows.
- Building account systems, cloud services, or paid API infrastructure.
- Implementing GGUF export or local runtime.
- Claiming model quality from a tiny smoke run.
