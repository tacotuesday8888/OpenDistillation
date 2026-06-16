# Personal Model Learning Core Design

Date: 2026-06-06

## Plain-Language Goal

OpenDistillation is building a small factory for personal models.

For v0, the factory takes one `.txt` or `.md` notes file, turns it into high-quality question-answer examples, trains a small notes model, and produces an honest report that says whether the trained model actually improved.

The important product promise is not "we ran training." The promise is:

> Your notes became controlled training data, the model was tested on questions it did not see during training, and the quality report shows what improved, what failed, and what is still unsafe to claim.

The latest Colab T4 result proved that the pipeline can run, but it did not prove useful learning quality. The 2026-06-16 same-chunk disambiguation fact-ledger smoke trained on 48 fact-ledger rows with 16 disambiguation rows and scored 8 held-out eval questions. The adapter changed all 8 answers but hit 0/8 exact facts, below the previous best 1/8. The next work should diagnose the local learning signal before another GPU run, not broaden the product.

## Scope

In scope for this design:

- TXT/MD notes only.
- One notes / school personal model.
- Colab-first workflow.
- Deterministic mock teacher as the safe default.
- One open-source real teacher path.
- One small student model around 0.5B-1.5B parameters.
- Response distillation through supervised fine-tuning.
- Honest dataset and model-quality reports.
- Optional RAG design for exact note lookup.

Out of scope for now:

- SaaS.
- Mac app.
- Phone app.
- Account system.
- Backend.
- GGUF export implementation.
- Local runtime packaging.
- Multi-profile system.
- Coding, writing, work, or phone model flows.
- Broad benchmark suite.
- Claims of novel ML algorithms.
- Committing generated datasets, adapters, checkpoints, model files, API keys, `.env` files, or local config.

## Recommended Approach

Use a data-first learning loop.

The failed 30-step runs are more likely data and evaluation problems than pure optimizer problems. A tiny LoRA adapter cannot reliably learn exact personal facts if the training examples are weak, too few, poorly targeted, or evaluated through a leaky or mismatched test.

The recommended loop is:

1. Load and chunk notes.
2. Build a fact ledger from the notes.
3. Use a stronger teacher to create train rows and separate eval rows from that ledger.
4. Block train/eval question leakage before training.
5. Train the small student with TRL `SFTTrainer` and PEFT LoRA.
6. Evaluate the base model and adapter on held-out paraphrase questions.
7. Report exact fact hits, answer evidence, failures, and caveats.
8. Only then try training accelerators or model swaps.

Two alternatives were rejected:

- Training-tweak-first: increasing steps, rank, or learning rate without fixing the data loop can produce different answers without better answers. That is what the latest smoke already showed.
- RAG-only: retrieval is excellent for exact facts, but it does not create a smaller model that has learned the user's preferred material, style, quiz patterns, or recurring explanations.

## Where Distillation, Fine-Tuning, And RAG Fit

Distillation means a larger teacher model turns source notes into examples that a smaller model can learn from. In v0, this is response distillation: the teacher writes the desired answer text. Logits distillation is out of scope because it needs teacher token probabilities and a more complex training path.

Fine-tuning means the small student model learns from those examples. For v0 this should stay supervised fine-tuning with LoRA adapters, not full model training. The training artifact is an adapter that changes behavior without saving a full model copy.

RAG means retrieval-augmented generation. At question time, the system searches the user's notes, puts the most relevant chunks into the prompt, and asks the model to answer from that evidence.

RAG is useful because it gives exact memory. If the user asks for a date, phrase, formula, or quote from notes, retrieval is often safer than hoping the model memorized the detail in its weights.

RAG does not replace training because retrieval does not teach the model a durable behavior. Without the index, the model has not learned the notes. RAG also does not teach study-question style, explanations, summaries, misconception checks, or the user's recurring patterns. The product-grade path should eventually support both:

- Trained adapter for learned behavior and note-shaped answering.
- RAG for exact facts, citations, and "open-book" reliability.

For the first quality gate, evaluate the adapter without RAG first. That shows whether training alone learned the controlled notes signal. Later, add a separate "adapter + RAG" report so exact-memory quality is not mixed with model-memory quality.

## Product Learning Loop

### 1. Note Ingestion

Keep the current input surface:

- Accept one `.txt` or `.md` file.
- Normalize text.
- Split into stable source chunks with IDs like `chunk-0001`.
- Keep source text local unless the user explicitly opts into a hosted teacher.

The chunker should optimize for clean facts, not maximum context length. For notes, headings, bullets, short paragraphs, and colon-delimited facts are more useful than arbitrary huge chunks.

### 2. Fact Ledger

Before asking a teacher model to write rows, build a simple fact ledger.

A fact ledger is a private intermediate table, not the training dataset. Each row should describe one learnable unit from the notes:

- `fact_id`
- `source_chunk_id`
- `source_text` or source span
- `expected_terms`
- `fact_kind`, such as exact phrase, date/time, definition, relationship, or explanation
- `source_hash`

For v0, this can start simple:

- Extract explicit note facts from bullets, headings, and `label: value` lines.
- Let the teacher propose additional atomic facts from each chunk.
- Keep only facts whose answer text is visibly supported by the source chunk.

This matters because the current dataset is row-first. A product-grade loop should be fact-first. The system needs to know what fact each training and eval question is testing.

### 3. Teacher Data Generation

The teacher should receive one chunk or one fact card at a time and generate multiple row types:

- Direct recall question.
- Paraphrased recall question.
- Explanation question.
- Flashcard row.
- Misconception row with an explicit "do not invent" answer.
- Optional negative question whose correct answer is "not stated in the notes."

Every teacher row must be grounded in the source chunk. The teacher should not be allowed to invent facts just to make the dataset look larger.

The training JSONL can keep the current public schema:

```json
{"instruction":"...","response":"...","source_chunk_id":"chunk-0001"}
```

The product core should also save a sidecar manifest for internal checks:

```json
{
  "row_id": "train-000001",
  "fact_id": "fact-0001",
  "split": "train",
  "source_chunk_id": "chunk-0001",
  "source_hash": "...",
  "expected_terms": ["Glass Harbor"],
  "teacher_model": "Qwen/Qwen2.5-7B-Instruct",
  "prompt_version": "notes_teacher_v1"
}
```

This keeps the beginner-facing dataset simple while giving the system enough metadata to prevent leakage and write a real quality report.

### 4. Train/Eval Split Without Cheating

The model is allowed to see the fact during training. Otherwise a no-RAG model has no fair way to answer personal facts. The model must not see the same eval question during training.

The safe v0 split should work like this:

1. Build fact cards first.
2. For each selected fact, generate training questions and one separate eval question.
3. Put only training rows into the trainer.
4. Keep eval rows in a separate file or manifest section.
5. Run contamination checks before training:
   - no exact duplicate normalized instructions across train and eval;
   - no near-duplicate train/eval instructions above a fixed similarity threshold;
   - no eval row accidentally included in the training JSONL;
   - stable split seed recorded in the manifest;
   - source file hash recorded so old eval rows are not reused after notes change.

For exact-fact tests, the held-out eval question should be a different paraphrase for a fact that appears in training responses. That tests whether the small model can answer a new wording from learned notes. Later, a harder chunk-holdout test can measure generalization, but it should not be the first memory gate.

### 5. Dataset Quality Gate

Before training, the generated dataset must pass deterministic checks:

- Schema-valid rows.
- Each train row maps to a known source chunk.
- Each train row maps to a known fact card.
- Required chunk or fact coverage is met.
- No duplicate or near-duplicate questions inside the training split.
- No train/eval question leakage.
- Answers are not too short or too long for a tiny SFT run.
- Exact-fact answers contain the expected terms.
- Teacher output is rejected if it includes unsupported facts.

The report should explain failures in plain language. Example:

> This dataset is not ready for training because 3 eval questions are near-duplicates of training questions. The model could pass by memorizing the question wording.

### 6. Training

Training should stay boring and proven.

Use Hugging Face `datasets.Dataset` for the training table, Transformers for tokenizer/model loading and chat templates, TRL `SFTTrainer` for supervised fine-tuning, and PEFT LoRA for adapters.

Do not write a custom optimizer, custom LoRA implementation, custom trainer, custom tokenizer, or custom GPU memory manager.

The current student can remain `Qwen/Qwen2.5-0.5B-Instruct` until the data loop works. `Qwen/Qwen3-0.6B` is a reasonable later controlled student comparison, but it should not be mixed into the next data-quality change.

### 7. Evaluation And Honest Quality Report

The first report should compare:

- Base student model.
- Trained adapter.
- Later: trained adapter with RAG.

For the first gate, use deterministic fact-hit scoring:

- exact expected term hit;
- required pair hit for paired facts, such as time plus color;
- refusal/unknown check for questions not answered by the notes;
- optional lexical overlap as a weak supporting signal only.

Hugging Face Evaluate can be useful later for standard metrics such as exact match or SQuAD-style F1 formatting, but v0 personal notes quality needs custom fact-aware checks first. General text-generation metrics like BLEU or ROUGE are not enough to prove personal fact learning.

The quality report must be allowed to say "failed." A changed answer is not an improvement unless it contains the expected note facts and avoids unsupported claims.

## First Believable Quality Gate

The next quality gate should be small but stricter than the current smoke.

Recommended gate:

- Use a committed fact-rich TXT/MD sample with at least 8 atomic facts across several chunks.
- Generate a fact ledger.
- Generate 6 training rows per fact with explicit label/value signal.
- Generate 1 held-out eval question per fact.
- Prove zero train/eval question leakage.
- Train the current Qwen2.5-0.5B LoRA adapter in a bounded Colab T4 run.
- Compare base versus adapter on the held-out eval questions.

Pass condition:

- Dataset quality gate passes with zero required errors.
- Contamination check passes with zero train/eval question leaks.
- Adapter exact fact hits are at least 5/8.
- Adapter improves over base by at least 3 exact fact hits.
- No more than 1/8 answers is an unsupported hallucination on the checked fact.
- The report prints the failed questions and raw answers.

This is not a benchmark claim. It is the first credible product smoke that says the small model can learn a controlled set of personal note facts from generated training data.

Update on 2026-06-08: this gate shape was exercised twice with 8 facts, 24 train rows, 8 held-out eval rows, zero leakage, and 30-step Colab T4 LoRA runs. The first run used earlier fact-ledger wording and scored 0/8 exact held-out fact hits, the same as the base model. Follow-up local diagnosis found the SFT prompt/completion format matches current TRL docs and the adapter-disabled comparison path matches current PEFT docs, so the rows were changed to put exact values first, use direct exact-recall eval questions, and record row styles in the sidecar. The revised value-first run also scored base 0/8 and trained 0/8.

Current evidence update: expected-term scoring now survives comparison row reordering, the SFT preview is visible before training, and the local fact-ledger signal now uses six label/value rows per fact with same-chunk disambiguation and swapped-value correction rows where contrast facts exist. The 2026-06-16 T4 smoke tested exactly that shape and failed: 8 facts, 48 fact-ledger train rows, 8 held-out eval rows, 16 disambiguation rows, zero leakage, 30-step `Qwen/Qwen2.5-0.5B-Instruct` LoRA, 8 changed answers, and trained 0/8 exact fact hits. The pass condition above remains the bar. If trained answers change but still miss the expected facts, the result is failed.

## Library Choices

Use proven open-source tools:

- PyTorch: underlying tensor and CUDA runtime used by the model libraries.
- Hugging Face Transformers: model loading, tokenizer loading, chat templates, and generation.
- Hugging Face Datasets: in-memory and JSONL-backed train/eval datasets, mapping, filtering, and split handling.
- TRL `SFTTrainer`: supervised fine-tuning for instruction/prompt-completion data.
- PEFT LoRA: parameter-efficient adapters and adapter loading through `PeftModel`.
- Unsloth: optional acceleration layer after the normal TRL/PEFT loop proves quality or hits memory limits.
- Sentence Transformers plus FAISS or a simple lexical BM25 library: later RAG retrieval path for exact memory.
- Hugging Face Evaluate: later standard metric wrapper where a standard metric is genuinely appropriate.

Do not build custom ML algorithms in v0. OpenDistillation's value is the product-grade personal-model workflow: data generation, split safety, evaluation, privacy choices, and beginner-readable reporting.

## Where Each Model Family Fits

### Student

Keep one small student for the next loop:

- Current default: `Qwen/Qwen2.5-0.5B-Instruct`.
- Later candidate: `Qwen/Qwen3-0.6B`.

Do not change the student and the dataset loop in the same experiment. If the quality gate improves, then compare student choices.

### Cheap/Local Teacher

Use these for safe local demos and smoke tests:

- `MockTeacherEngine` as the deterministic no-download default.
- `Qwen/Qwen2.5-1.5B-Instruct` as the current cheap local open-source teacher smoke path.
- `Qwen/Qwen3-1.7B` as a later current-generation cheap local teacher candidate if it fits the target runtime.

These are useful for wiring, privacy, and cheap iteration. They should not be treated as enough evidence for product-grade data quality until their generated rows pass the new fact-ledger checks.

### Colab/Hugging Face Teacher

Use a stronger open-weight teacher for the first believable data-quality path:

- Preferred next teacher target: `Qwen/Qwen2.5-7B-Instruct` or `Qwen/Qwen3-8B`, depending on Colab fit and dependency stability.
- Keep generation bounded by chunk/fact card so the teacher does not need the whole notebook context.
- Use deterministic generation first.

This teacher can run locally in Colab if resources allow, or through a Hugging Face-hosted path only with explicit user opt-in.

### Future Stronger Teacher

For later product quality, use larger open-weight or hosted teachers as optional modes:

- `Qwen/Qwen2.5-14B-Instruct` or larger Qwen-family teachers.
- DeepSeek-style reasoning/distilled Qwen models such as `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` as teacher or critic candidates.

Use these to generate and critique training data, not as proof that the small student learned. Do not train on hidden chain-of-thought text. If a hosted provider is used, label privacy clearly and require opt-in before sending personal notes.

## Where Colab Fits

Colab remains the first-run product surface because it gives beginners a GPU without local setup.

The committed notebook defaults should remain safe:

- `INSTALL_TRAINING_DEPS = False`
- `RUN_REAL_TEACHER = False`
- `RUN_TRAINING = False`

The optional Colab path should be:

1. Install bounded Hugging Face dependencies without upgrading Colab's preinstalled GPU `torch`.
2. Run teacher generation only after opt-in.
3. Run training only after opt-in.
4. Save generated datasets, adapters, and checkpoints only inside the Colab runtime or ignored output paths.
5. Print status markers so long cells can be audited after UI failures.

Use `google-colab-cli` for verification when possible. Browser or Chrome control should be fallback only when CLI cannot answer the question.

## Privacy Stance

Personal notes are sensitive.

The default path must remain local and deterministic. A hosted teacher is acceptable for demos, public sample notes, and explicitly opted-in synthetic data. It is not acceptable to quietly send personal notes to a hosted model.

Every teacher option should carry a clear label:

- Local/no remote text transfer.
- Hosted/sends note text to provider.
- Demo-only/sample data.

If a secret or token is ever exposed in logs or git, treat it as compromised and recommend rotation.

## What To Implement Next

The next implementation goal should be:

> Diagnose the failed fact-ledger disambiguation learning signal locally before another GPU run.

This should happen before Unsloth migration, export, local runtime packaging, new model profiles, another GPU smoke, or a training-parameter sweep.

The exact next local work is:

- Compare the 2026-06-15 six-row 1/8 signal against the 2026-06-16 disambiguation 0/8 signal at the exact prompt/completion level.
- Explain why the disambiguation rows taught answer shape and invented numeric values without exact value binding.
- Add local diagnostics or row changes only when they directly make exact label/value facts clearer.
- Preserve the public JSONL schema, train/eval split, leakage checks, SFT preview, and exact fact-hit scoring by fact/question identity.
- Keep treating changed answers as failure when expected facts are still missed.
- Do not spend another T4 run until the local signal has a concrete, testable reason to beat the previous best trained score of 1/8.

## Source Notes Checked

Current docs and model cards checked on 2026-06-06:

- Hugging Face TRL `SFTTrainer`: https://huggingface.co/docs/trl/sft_trainer
- Hugging Face TRL PEFT integration: https://huggingface.co/docs/trl/peft_integration
- Hugging Face PEFT LoRA guide: https://huggingface.co/docs/peft/developer_guides/lora
- Hugging Face Transformers chat templates: https://huggingface.co/docs/transformers/chat_templating
- Hugging Face Datasets processing and splitting: https://huggingface.co/docs/datasets/process
- Hugging Face Evaluate metric guidance: https://huggingface.co/docs/evaluate/choosing_a_metric
- Hugging Face TRL Unsloth integration: https://huggingface.co/docs/trl/unsloth_integration
- Unsloth official fine-tuning docs: https://docs.unsloth.ai/get-started/fine-tuning-guide
- Sentence Transformers semantic search docs: https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html
- FAISS documentation: https://faiss.ai/
- Qwen model cards on Hugging Face for Qwen2.5 and Qwen3 small/teacher candidates.
- DeepSeek R1 Distill Qwen model card on Hugging Face.
