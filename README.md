# NLP 2026 — Quiz Game Challenge
**Group:** *GliEmbeddingRuspanti*  
Andrea Rossi · Carolina Parenti · Fabio Marco Floris · Francesco Sarra · Roberto Sonzini Gobbi

---

## Overview

This project tackles an automated quiz-game challenge using a progression of NLP techniques: from vanilla LLM prompting, through Retrieval-Augmented Generation and agentic pipelines, up to a speech-enabled interface. The work is split across focused development notebooks and consolidated in one main file (**`NLP_project.ipynb`** ) where all parts are organized

## Approach Summary

1. **Models Analysis** — Evaluated Llama 3.1 8B, Gemma-2 9B, and Qwen2.5 7B with TF-IDF, sBERT, and CrossEncoder similarity functions across four prompt strategies. Best configs combined into a confidence-based ensemble.

2. **RAG** — Built a custom vector database from the Italian Wikipedia dump using `BAAI/bge-m3` embeddings and LanceDB. Hybrid retrieval (vector DB + DuckDuckGo) reranked with `bge-reranker-v2-m3` feeds context to Qwen2.5-7B.

3. **Agentic (Tool Use & ReAct)** — Extended the LLM with math-specific tools. ReAct showed higher accuracy but consistently exceeded the 30-second timeout; Tool Use was retained as the practical option.

4. **Agentic PAL** — Program-Aided Language model: routes theoretical questions through RAG and computational ones through generated Python code, satisfying both accuracy and latency constraints.

5. **Speech Interface** — Audio transcription pipeline with `CleanerPreprocessing` (Unicode normalisation, semantic option filtering, optional LLM cleaning) bridging speech input to the text-based answering model.

## Requirements

All dependencies are installed at runtime inside each notebook. 
