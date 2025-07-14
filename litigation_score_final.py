

import pandas as pd
import numpy as np
import os
from gensim.models import KeyedVectors, word2vec
from gensim.models.word2vec import LineSentence
from global_options import DataPaths, AnalysisOptions
from pathlib import Path 

# ----------------------------
# SECTION 1: Read Seed Words from File
# ----------------------------

def read_seed_words(file_path):
    """
    Read seed words from a text file where words are separated by newlines.
    
    Args:
        file_path (str): Path to the seed words text file
    
    Returns:
        list: List of seed words
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f.readlines()]
        words = [word for word in words if word]  # Remove empty strings
    return words

# Read seed words using global configuration
try:
    expanded_seedwords = read_seed_words(DataPaths.SEEDWORDS_FILE)
    print(f"Successfully read {len(expanded_seedwords)} seed words.")
    print("First 5 words:", expanded_seedwords[:5])
except FileNotFoundError:
    print(f"Error: Seed words file not found at {DataPaths.SEEDWORDS_FILE}")
    expanded_seedwords = []
except Exception as e:
    print(f"Error reading seed words: {str(e)}")
    expanded_seedwords = []

# Load TF-IDF weights
tfidf_weights = pd.read_csv(DataPaths.PROCESSED_DIR / 'tfidf_weights.csv')

# ----------------------------
# SECTION 2: Word Vector Processing
# ----------------------------

# Train Word2Vec model
s = word2vec.LineSentence(DataPaths.PROCESSED_DIR / "processed_text.txt")
model = word2vec.Word2Vec(s, window=10, min_count=5, workers=4)

# Save and load word vectors
word_vectors = model.wv
word_vectors.save("word2vec.wordvectors")
wv = KeyedVectors.load("word2vec.wordvectors", mmap='r')

# Get vocabulary list from the model
word_list = model.wv.index_to_key if hasattr(model, 'wv') else []

# Create DataFrame from vocabulary - MODIFIED TO FIX ERROR
wordlist = pd.DataFrame(word_list, columns=['word'])  # Explicit column naming
wordlist.index = wordlist['word']  # Changed from wordlist[0] to wordlist['word']

# Rest of your original code remains exactly the same...
wordlist = wordlist.rename(columns={'word': 'word'})  # This line is redundant but kept as-is

# Merge TF-IDF with vocabulary
merged_features = tfidf_weights.T.join(wordlist, how='inner').drop(columns=['word'])
row_names = merged_features.index.tolist()

# ----------------------------
# SECTION 3: Calculate Similarity Scores
# ----------------------------

# Create output directory
os.makedirs(DataPaths.SCORE_FILE, exist_ok=True)

# Calculate and save similarity scores
for seed_word in expanded_seedwords:
    if hasattr(model, 'wv') and seed_word in model.wv.key_to_index:
        similarity_scores = []
        similar_words = []
        
        for vocab_word in row_names:
            try:
                similarity = model.wv.similarity(vocab_word, seed_word)
                similarity_scores.append(similarity)
                similar_words.append(vocab_word)
            except KeyError:
                print(f"Vocabulary word '{vocab_word}' not present for '{seed_word}'")
        
        result_df = pd.DataFrame(similarity_scores, index=similar_words, columns=[seed_word])
        output_file = DataPaths.SCORE_FILE / f"{seed_word}.csv"
        result_df.to_csv(output_file)
        print(f"Saved similarity scores for '{seed_word}' to {output_file}")
    else:
        print(f"Warning: '{seed_word}' not present in the model vocabulary")

# ----------------------------
# SECTION 4: Document Scoring
# ----------------------------

# Read document IDs
with open(DataPaths.PROCESSED_DIR / 'document_ids.txt', 'r', encoding='utf-8') as f:
    ids = [line.strip('\n') for line in f.readlines()]

# Process scored files
os.chdir(str(DataPaths.SCORE_FILE))
scored_files = [file.stem for file in Path(DataPaths.SCORE_FILE).glob("*.csv") if file.stem != 'df_listscore']

# Calculate document scores
df_listscore = pd.DataFrame(ids)
n_columns = merged_features.shape[1]

for name2 in scored_files:
    df = pd.read_csv(f"{name2}.csv")[name2]
    list_score = [np.dot(df.values.astype(float), merged_features.iloc[:, i].values.astype(float)) 
                 for i in range(n_columns)]
    df_listscore[name2] = pd.DataFrame(list_score)

# Save final results
df_listscore.to_csv(DataPaths.SCORE_FILE / "df_listscore.csv", index=False)
print(f"Results saved to: {DataPaths.SCORE_FILE / 'df_listscore.csv'}")