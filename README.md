# AmbedkarGPT – AI Intern Task (Kalpit Pvt Ltd, UK)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python" />
  <img src="https://img.shields.io/badge/LangChain-Community-orange?logo=chainlink" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector%20Store-green?logo=postgresql" />
  <img src="https://img.shields.io/badge/HuggingFace-Embeddings-yellow?logo=huggingface" />
  <img src="https://img.shields.io/badge/Ollama-Mistral%207B-red?logo=cloudsmith" />
  <img src="https://img.shields.io/badge/License-MIT-brightgreen" />
</p>



This repository contains my submission for the **AI Intern Assignment  
It includes two components:

- **Assignment 1:** A functional RAG-based Q&A system (local, no APIs)
- **Assignment 2:** A minimal evaluation framework with retrieval metrics

The entire project runs **locally**, uses **open-source tools**, and requires **no API keys**.

---

## 🚀 Assignment 1 — RAG Q&A Prototype

### ✔ Features
- Loads `speech.txt` (Ambedkar excerpt)
- Splits text into manageable chunks
- Generates embeddings using  
  **sentence-transformers/all-MiniLM-L6-v2**
- Stores vectors using **ChromaDB** (local vector DB)
- Retrieves relevant chunks for a question
- Uses **Ollama (Mistral 7B)** as the local LLM
- Provides a command-line Q&A interface

### ▶️ Run Assignment 1

Make sure **Ollama** is running:

```bash
ollama serve
```

## Start the Q&A system:
```bash
python main.py
```
---

## 📘 Assignment 2 — Evaluation Framework

This assignment evaluates the retrieval quality of the RAG system
using the provided document corpus and dataset of 25 test questions.

What the evaluation does

- Loads all documents from the corpus/ folder

- Builds a vector database using the same embedding model

- Retrieves top-K (K = 3) chunks per question

- Compares retrieved documents with ground-truth file names

- Computes standard metrics:

    - Hit@3

    - MRR (Mean Reciprocal Rank)
 
### ▶️ Run Assignment 2

```bash
python evaluation.py
```

## Results are saved to:
```bash
simple_results.json
```

## 📁 Repository Structure

```bash
project-root/
│
├── main.py # RAG pipeline (Assignment 1)
├── evaluation.py # Retrieval evaluation (Assignment 2)
├── validate_dataset.py # Dataset integrity checker
│
├── speech.txt # Source text for Assignment 1
├── corpus/ # 6 documents for Assignment 2
├── test_dataset.json # 25 evaluation questions
│
├── requirements.txt
└── README.md
```

## ⚙️ Installation & Setup
### 1️⃣ Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```
###2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
###3️⃣ Install & pull Ollama model
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull mistral
ollama serve
```
---

## 🧠 Technologies Used
- Python 3.9+

- LangChain (community components)

- ChromaDB (local vector store)

- HuggingFace Sentence Transformers

- Ollama (Mistral 7B model)

- RecursiveCharacterTextSplitter

- JSON evaluation

---

## 🔍 RAG Architecture Overview

                ┌────────────────────────────┐
                │        User Query           │
                └──────────────┬─────────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │  Embedding Retriever     │
                 │ (ChromaDB + MiniLM-L6)   │
                 └──────────────┬───────────┘
                               │ Top-K chunks
                               ▼
                 ┌──────────────────────────┐
                 │    Retrieved Context      │
                 │   (Relevant Text Chunks)  │
                 └──────────────┬───────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │        LLM (Ollama)       │
                 │       Mistral 7B Model    │
                 └──────────────┬───────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │     Final Answer          │
                  │(Context-aware Generation) │
                  └──────────────────────────┘


---

## 📜 License

```markdown
MIT License

Copyright (c) 2025 Arpan Kumar Moharana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 👤 Author

### Arpan Kumar Moharana
