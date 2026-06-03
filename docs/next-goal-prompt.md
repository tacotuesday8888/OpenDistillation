# Recommended Next Goal Prompt

Use this as the next `/goal` after the uploaded `.txt` and `.md` Colab rehearsals are both documented as passed:

```text
/goal Prepare OpenDistillation's first public demo release candidate without expanding scope. Work in /Users/langqi/Developer/Projects/OpenDistillation on latest main. Keep v0 narrow: TXT/MD notes only, one notes/school model only, Colab-first, MockTeacherEngine as the safe default, optional local Qwen real teacher, optional short TRL/PEFT LoRA training, and optional before/after comparison. Do not build SaaS, Mac app, phone app, accounts, backend, GGUF export, local runtime packaging, multiple profiles, coding model, writing model, or broad training platform. Starting evidence: sample notes, uploaded .txt, uploaded .md, clean GitHub-opened T4 training/comparison, and one real-teacher T4 wiring check have all passed once and are documented. Make the repo ready for a first public demo by tightening the demo script, issue/milestone status, GitHub launch checklist, and the narrow next implementation plan for GGUF/local-runtime handoff without implementing export. Confirm docs agree with each other, run local verification, review the diff for secrets/artifacts, commit, and push.
```

## Why This Goal

The sample-notes path, uploaded `.txt` path, uploaded `.md` path, optional mock-teacher training path, and one real-teacher T4 wiring path have all passed once. The next risk is public-demo clarity: someone should be able to open the repo, understand the narrow notes-model demo, and see exactly what is verified versus deferred.

## Done Means

- The sample-notes default remains safe and CPU-runnable.
- README, notebook docs, smoke-test results, roadmap/issue docs, and launch checklist agree on verified versus deferred status.
- The first public demo script is concrete and short enough for a non-expert to follow.
- The next GGUF/local-runtime work is scoped as a plan or issue set only; no export implementation is built yet.
- No generated datasets, model artifacts, checkpoints, secrets, or local config are committed.

## Do Not Use This Goal For

- Improving model quality.
- Expanding beyond TXT/MD notes.
- Adding coding, writing, work, phone, or multi-profile flows.
- Implementing GGUF export or local runtime.
- Claiming benchmark results.
