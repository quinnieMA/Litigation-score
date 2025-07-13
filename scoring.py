# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 15:24:48 2025

@author: FM
"""

import pandas as pd
import os
from gensim.models import KeyedVectors
from gensim.models import word2vec
from global_options import DataPaths, AnalysisOptions
# ----------------------------
# SECTION 1: Read Seed Words from File
# ----------------------------

def read_seed_words(file_path):
    """
    Read seed words from a text file where words are separated by newlines.
    Example file content:
    word1
    word2
    word3
    
    Args:
        file_path (str): Path to the seed words text file
    
    Returns:
        list: List of seed words
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        # Read all lines, strip whitespace, and filter out empty lines
        words = [line.strip() for line in f.readlines()]
        words = [word for word in words if word]  # Remove empty strings
    return words

# Path to seed words file
seed_words_path = r"C:\Users\FM\OneDrive\NLP\data\dictionaries\seedwords.txt"

# Read seed words (newline-separated format)
try:
    list2 = read_seed_words(seed_words_path)
    print(f"Successfully read {len(list2)} seed words from file.")
    print("First 5 words:", list2[:5])  # Print sample for verification
except FileNotFoundError:
    print(f"Error: Seed words file not found at {seed_words_path}")
    list2 = []  # Fallback empty list
except Exception as e:
    print(f"Error reading seed words file: {str(e)}")
    list2 = []  # Fallback empty list
# ----------------------------
# SECTION 2: Word Vector Processing
# ----------------------------
s = word2vec.LineSentence(DataPaths.PROCESSED_DIR / "processed_text.txt")
model = word2vec.Word2Vec(s, window=10, min_count=5, workers=4)

# ----------------------------
# SECTION 1: Saving and Loading Word Vectors
# ----------------------------

# Store just the words + their trained embeddings from the model
word_vectors = model.wv
word_vectors.save("word2vec.wordvectors")

# Load back with memory-mapping (read-only, shared across processes)
wv = KeyedVectors.load("word2vec.wordvectors", mmap='r')

# Get vocabulary list
word_list = model.wv.index_to_key if hasattr(model, 'wv') else []

# ----------------------------
# SECTION 3: Calculate Similarity Scores
# ----------------------------

# Define output directory
score_path = r"C:/Users/FM/OneDrive/NLP/outputs"
os.makedirs(score_path, exist_ok=True)  # Create directory if it doesn't exist

for seed_word in list2:
    similarity_scores = []
    similar_words = []
    
    if hasattr(model, 'wv') and seed_word in model.wv.key_to_index:
        for vocab_word in word_list:
            try:
                similarity = model.wv.similarity(vocab_word, seed_word)
                similarity_scores.append(similarity)
                similar_words.append(vocab_word)
            except KeyError:
                print(f"Vocabulary word '{vocab_word}' not present for '{seed_word}'")
        
        # Create and save results DataFrame
        result_df = pd.DataFrame(similarity_scores, 
                               index=similar_words, 
                               columns=[seed_word])
        output_file = os.path.join(score_path, f"{seed_word}.csv")
        result_df.to_csv(output_file)
        print(f"Saved similarity scores for '{seed_word}' to {output_file}")
    else:
        print(f"Warning: '{seed_word}' not present in the model vocabulary")

# ----------------------------
# SECTION 4: Save Vocabulary (Optional)
# ----------------------------

if word_list:
    word_df = pd.DataFrame(word_list, columns=['word'])
    vocab_output_path = os.path.join(score_path, 'index-to-key.csv')
    word_df.to_csv(vocab_output_path, index=False)
    print(f"Vocabulary saved to {vocab_output_path}")