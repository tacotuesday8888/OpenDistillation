# Open-Source Tool Strategy

Checked on 2026-06-08. The rule for v0 is simple: use proven open-source ML tools for model loading, training, adapters, and future retrieval, but keep OpenDistillation's custom code focused on the product layer: fact extraction, data/eval generation, leakage prevention, scoring, privacy choices, and plain-English reporting.

## Used Now

- Python standard library for the local fact-ledger quality gate. This keeps the CPU/default path light, inspectable, and runnable without installs.
- Hugging Face Transformers for optional local teacher generation and optional comparison.
- Hugging Face Datasets inside the optional training engine, where rows become a trainer-ready in-memory dataset.
- TRL `SFTTrainer` and PEFT LoRA for opt-in supervised fine-tuning. This keeps training on a proven path instead of custom GPU/trainer code.

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

## Sources Checked

- Hugging Face Datasets loading and in-memory data: https://huggingface.co/docs/datasets/loading
- Hugging Face Evaluate metric selection: https://huggingface.co/docs/evaluate/choosing_a_metric
- Hugging Face TRL PEFT and Unsloth integration docs: https://huggingface.co/docs/trl/peft_integration and https://huggingface.co/docs/trl/unsloth_integration
- RapidFuzz string matching docs: https://rapidfuzz.github.io/RapidFuzz/
- scikit-learn text feature extraction docs: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
- FAISS docs: https://faiss.ai/
- Chroma docs: https://docs.trychroma.com/docs/overview/introduction
- LlamaIndex RAG docs: https://docs.llamaindex.ai/en/stable/understanding/rag/
- LangChain retrieval docs: https://docs.langchain.com/oss/python/langchain/retrieval
- Unsloth fine-tuning docs: https://docs.unsloth.ai/get-started/fine-tuning-llms-guide
