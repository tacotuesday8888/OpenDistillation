# Open-Source Tool Strategy

Checked on 2026-06-08. The rule for v0 is simple: use proven open-source ML tools for model loading, training, adapters, and future retrieval, but keep OpenDistillation's custom code focused on the product layer: fact extraction, data/eval generation, leakage prevention, scoring, privacy choices, and plain-English reporting.

## Used Now

- Python standard library for the local fact-ledger quality gate. This keeps the CPU/default path light, inspectable, and runnable without installs.
- Hugging Face Transformers for optional local teacher generation and optional comparison.
- Hugging Face Datasets inside the optional training engine, where rows become a trainer-ready in-memory dataset.
- TRL `SFTTrainer` and PEFT LoRA for opt-in supervised fine-tuning. This keeps training on a proven path instead of custom GPU/trainer code.

The 2026-06-08 bounded Colab T4 fact-ledger smoke used the standard open-source stack already chosen for v0: Colab's GPU `torch`, Transformers `4.57.6`, Datasets `5.0.0`, TRL `0.29.1`, PEFT `0.18.1`, and Accelerate `1.13.0`. That run completed a 30-step LoRA adapter but still scored 0/8 exact held-out fact hits before and after training. The result points to a learning-signal problem, not a need for custom GPU code.

Follow-up doc checks on 2026-06-08 found no reason to replace the training stack. Current TRL docs support conversational prompt/completion datasets and completion-only loss for prompt/completion SFT. Current PEFT docs confirm that `PeftModel.from_pretrained()` may modify the passed base model in place and that `disable_adapter()` is the supported way to run base-model inference. OpenDistillation's local fix therefore stays in the product layer: clearer fact rows, clearer eval wording, and better reporting.

## Later

- RapidFuzz: likely first add if local near-duplicate checks need stronger fuzzy matching at larger row counts. Not needed now because the standard-library gate catches exact, sequence-similar, and token-overlap leaks covered by tests.
- Hugging Face Evaluate: useful later for standard exact-match/F1 style reporting once the project has a stable held-out notes evaluation set. The current personal-fact check remains custom because the important question is whether exact note terms appear.
- scikit-learn: useful later for a lightweight TF-IDF or lexical retrieval baseline before heavier RAG infrastructure.
- FAISS or Chroma: useful later when OpenDistillation adds a real RAG path for exact memory and citations. FAISS is a vector similarity library; Chroma adds document/metadata storage and retrieval features around vector search.
- LlamaIndex or LangChain: useful later if RAG orchestration becomes complex. They are too broad for the current v0 gate, which only needs one TXT/MD notes path and deterministic local checks.
- Unsloth: useful later for speed and memory if the standard TRL/PEFT quality loop proves useful learning or hits Colab limits. It should not replace the local quality gate.

## Not Yet

- No new dependency for fact extraction, leakage checks, or exact-term scoring in this goal.
- No custom optimizer, custom LoRA implementation, custom tokenizer, custom trainer, or GPU memory manager.
- No SaaS, backend, vector database, broad benchmark suite, export path, or multi-profile system while the notes-model quality loop is still unproven.
- No Unsloth migration yet. Faster training would not fix that the current bounded fact-ledger run changed answers without improving exact fact hits.

## Sources Checked

- Hugging Face Datasets loading and in-memory data: https://huggingface.co/docs/datasets/loading
- Hugging Face Evaluate metric selection: https://huggingface.co/docs/evaluate/choosing_a_metric
- Hugging Face TRL PEFT and Unsloth integration docs: https://huggingface.co/docs/trl/peft_integration and https://huggingface.co/docs/trl/unsloth_integration
- Hugging Face TRL SFTTrainer docs: https://huggingface.co/docs/trl/sft_trainer
- Hugging Face PEFT PeftModel docs and LoRA guide: https://huggingface.co/docs/peft/package_reference/peft_model and https://huggingface.co/docs/peft/developer_guides/lora
- RapidFuzz string matching docs: https://rapidfuzz.github.io/RapidFuzz/
- scikit-learn text feature extraction docs: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
- FAISS docs: https://faiss.ai/
- Chroma docs: https://docs.trychroma.com/docs/overview/introduction
- LlamaIndex RAG docs: https://docs.llamaindex.ai/en/stable/understanding/rag/
- LangChain retrieval docs: https://docs.langchain.com/oss/python/langchain/retrieval
- Unsloth fine-tuning docs: https://docs.unsloth.ai/get-started/fine-tuning-llms-guide
