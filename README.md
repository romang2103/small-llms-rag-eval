# 📚 RAG Evaluation with Small and Quantised LLMs

This repository contains the full implementation and evaluation pipeline for the dissertation project:

**"Evaluating Small-Scale Large Language Models and Quantization for Question-Answering in Retrieval-Augmented Generation on Online Course Metadata"**

---

## 🧠 Overview

This project investigates how **small language models (1B–4B)** perform in a **Retrieval-Augmented Generation (RAG)** setup, specifically for **question-answering over course metadata** from IBM’s SkillsBuild platform. It also explores the impact of **post-training quantization** on response quality, latency, and memory efficiency.

---

## 🔍 Objectives

* Build a RAG system using structured educational metadata.
* Evaluate 5 small open-source LLMs for semantic accuracy and performance.
* Apply post-training quantisation (BF16, Q8\_0, Q4\_0) to Gemma 3 4B.
* Benchmark trade-offs in **accuracy**, **latency**, **tokens/sec**, and **memory usage**.
* Analyse performance across question types (factual, multi-hop, comparative, unanswerable).

---

## 🛠️ Technologies & Tools

* Python 3.10+
* [LangChain](https://github.com/langchain-ai/langchain)
* [FAISS](https://github.com/facebookresearch/faiss)
* [Ollama](https://ollama.com) (for local LLM inference)
* [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
* SentenceTransformers, Memory Profiler, Tiktoken, Evaluate
* Models from Hugging Face and Ollama (`gemma3`, `llama3.2`, `phi4-mini`)

---

## 📁 Repository Structure

```
.
├── data/
│   ├── course_metadata.json         # Source knowledge base (IBM SkillsBuild Course Metadata)
│   └── eval_dataset.json            # 20-question evaluation set (4 types)
├── faiss_index/                     # Faiss index
├── results/
│   └── quantised/                   # CSV results from 3x runs + averages
├── scripts/
│   └── rag_pipeline.py              # Main RAG logic and evaluation loop
├── README.md
```

---

## 🚀 How to Run

### 1. Clone the Repo and Set Up a Virtual Environment

```bash
git clone https://github.com/your-username/rag-small-llms-eval.git
cd rag-small-llms-eval

```


```python
python3 -m venv .venv
.venv\Scripts\activate
```

### 2. Prepare Ollama models

Ensure Ollama is installed and models are downloaded locally:

```bash
ollama pull gemma3:1b
ollama pull gemma3:4b
ollama pull llama3.2:1b
ollama pull llama3.2:latest
ollama pull phi4-mini:latest
ollama pull hf.co/unsloth/gemma-3-4b-it-GGUF:BF16
ollama pull hf.co/unsloth/gemma-3-4b-it-GGUF:Q4_0
ollama pull hf.co/unsloth/gemma-3-4b-it-GGUF:Q8_0
```

### 3. Run each block in the evaluation notebook (rag.ipynb)

This will:
* Install dependencies
* Import required libraries
* Load the dataset
* Retrieve relevant documents from the FAISS index
* Generate answers using each model
* Record latency, memory, and quality metrics
* Save full and averaged results in `results & results/quantised/`

---

## 📊 Output Metrics

Each model is evaluated using:

* **Lexical Metrics:** BLEU, ROUGE-L, F1
* **Semantic Metrics:** BERTScore, Semantic Similarity
* **Faithfulness:** Alignment with retrieved context
* **Efficiency:** Time to First Token, Total Time, Tokens/sec, Peak RAM, Model Size

Results are saved as:

* `rag_evaluation_results_1.csv`, `..._2.csv`, `..._3.csv`
* `rag_evaluation_results_avg.csv`
* `rag_model_blocks_avg.csv` (summary by question type)

---

## 📌 Notes

* All evaluations use **CPU-only inference** for reproducibility.
* Quantised model variants (Q4\_0, Q8\_0) are pulled from Hugging Face in GGUF format.
* Prompt design ensures grounded, concise answers with hallucination safeguards.

---

## 🔗 Links

* 🧠 [Ollama](https://ollama.com/)
* 🤗 [Hugging Face Gemma 3 4B GGUF](https://huggingface.co/unsloth/gemma-3-4b-it-GGUF)

---
