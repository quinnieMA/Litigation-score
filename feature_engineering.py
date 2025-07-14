# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 10:24:38 2025

@author: FM
"""

import warnings
warnings.filterwarnings("ignore")
import os
import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pathlib import Path
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import LineSentence  # Correct import for LineSentence
from global_options import DataPaths, AnalysisOptions

# ========================
# Text Processing Utilities
# ========================

def load_stopwords() -> set:
    """Load stopwords from predefined path"""
    with open(DataPaths.STOPWORDS_FILE, 'r', encoding='utf-8') as file:
        return set(file.read().split())

def clean_text(text: str, stopwords: set) -> str:
    """Apply text cleaning pipeline (punctuation and stopword removal)"""
    # Remove punctuation
    text = "".join([char for char in text if char not in string.punctuation])
    # Remove stopwords
    words = [word for word in text.split() if word.lower() not in stopwords]
    return ' '.join(words)

# ========================
# TF-IDF Vectorization
# ========================

def compute_tfidf_matrix() -> pd.DataFrame:
    """
    Compute TF-IDF matrix from processed documents
    Returns DataFrame with document IDs as index
    """
    # Load processed documents
    with open(DataPaths.PROCESSED_DIR / "processed_text.txt", 'r', encoding='utf-8') as f:
        corpus = [line.strip() for line in f if line.strip()]  # Remove empty lines
    
    # Verify corpus is not empty
    if not corpus:
        raise ValueError("No valid documents found in processed_text.txt")
    
    # Load document IDs
    with open(DataPaths.DOC_IDS, 'r', encoding='utf-8') as f:
        doc_ids = [line.strip() for line in f if line.strip()]
    
    # Verify document count matches
    if len(corpus) != len(doc_ids):
        print(f"Warning: Document count mismatch - Corpus: {len(corpus)}, IDs: {len(doc_ids)}")
        # Use generated IDs if mismatch
        doc_ids = [f"doc_{i}" for i in range(len(corpus))]
    
    # Compute TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Create DataFrame
    weights = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=vectorizer.get_feature_names_out(),
        index=doc_ids
    )
    return weights

# ========================
# Word2Vec Model Training
# ========================

def train_word_embeddings() -> Word2Vec:
    """Train and return Word2Vec model using processed corpus"""
    # Use LineSentence directly from gensim.models.word2vec
    sentences = LineSentence(str(DataPaths.PROCESSED_DIR / "processed_text.txt"))
    model = Word2Vec(
        sentences,
        vector_size=AnalysisOptions.W2V_DIM,
        window=AnalysisOptions.W2V_WINDOW,
        min_count=AnalysisOptions.PHRASE_MIN_COUNT,
        workers=AnalysisOptions.N_CORES
    )
    return model

def save_word_metadata(model: Word2Vec) -> None:
    """Save word vectors and vocabulary metadata"""
    # Save word vectors
    word_vectors_path = DataPaths.MODEL_DIR / "word_vectors.kv"
    model.wv.save(str(word_vectors_path))
    
    # Save vocabulary mapping
    vocab = pd.DataFrame({
        'word': model.wv.index_to_key,
        'index': range(len(model.wv.index_to_key))
    })
    vocab_path = DataPaths.PROCESSED_DIR / "index_to_key.csv"
    vocab.to_csv(vocab_path, index=False)

# ========================
# Main Processing Pipeline
# ========================

def text_processing_pipeline():
    """Execute complete text processing pipeline"""
    print("Starting text processing pipeline...")
    
    try:
        # 1. Load stopwords
        stopwords = load_stopwords()
        print(f"Loaded {len(stopwords)} stopwords")
        
        # 2. Preprocess text
        input_path = DataPaths.PROCESSED_DIR / "document_trigram.txt"
        output_path = DataPaths.PROCESSED_DIR / "processed_text.txt"
        
        # Verify input file exists
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                cleaned = clean_text(line, stopwords)
                if cleaned.strip():  # Only write non-empty lines
                    outfile.write(cleaned + '\n')
        print(f"Processed text saved to {output_path}")
        
        # 3. Compute TF-IDF
        tfidf_weights = compute_tfidf_matrix()
        tfidf_output = DataPaths.PROCESSED_DIR / "tfidf_weights.csv"
        tfidf_weights.to_csv(tfidf_output)
        print(f"TF-IDF weights saved to {tfidf_output}")
        
        # 4. Train word embeddings
        print("Training Word2Vec model...")
        w2v_model = train_word_embeddings()
        save_word_metadata(w2v_model)
        print(f"Word embeddings saved to {DataPaths.MODEL_DIR}")
        
        # 5. Vocabulary summary
        vocab_size = len(w2v_model.wv)
        print(f"\nTraining completed. Vocabulary size: {vocab_size}")
        print("Sample words:", w2v_model.wv.index_to_key[:10])
        
    except Exception as e:
        print(f"Error in processing pipeline: {e}")
        raise

if __name__ == "__main__":
    text_processing_pipeline()