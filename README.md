# NLP Litgation scoring Pipeline

![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![Gensim](https://img.shields.io/badge/gensim-4.0%2B-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-yellowgreen)

A robust NLP pipeline for document processing and semantic analysis with TF-IDF and Word2Vec embeddings.

## Table of Contents
- [Features](#features)
- [Pipeline Architecture](#pipeline-architecture)
- [Installation](#installation)


## Features

### Text Processing
- ✔️ Automated text cleaning pipeline  
- ✔️ Customizable stopword filtering  
- ✔️ Punctuation and special character removal  

### Feature Extraction
- 🎯 TF-IDF vectorization with scikit-learn  
- 🎯 Word2Vec embedding training  
- 🎛️ Configurable hyperparameters via `global_options.py`  

### Semantic Analysis
- 🔍 Seed word similarity scoring  
- 📊 Document-level semantic profiling  
- 💾 Results export to CSV  

## Pipeline Architecture

```text
text_processing_pipeline/
│
├── data/
│   ├── input/                  # Raw documents (.txt)
│   ├── processed/              # Cleaned text and intermediate files
│   └── dictionaries/           # Seed words and stopwords
│
├── models/                     # Serialized Word2Vec models
│   └── word_vectors.kv         # Pretrained embeddings
│
├── outputs/
│   ├── word_similarities/      # Per-seed-word similarity scores
│   └── df_listscore.csv     # Final aggregated scores
│
├── config/
│   └── global_options.py       # Path configurations
│
└── scripts/                    # Processing modules
    ├── NER_pipeline.py
    ├── preprocessing.py
    ├── ML.py
    ├── feature_engineering.py
    └── litigation_score_final.py
