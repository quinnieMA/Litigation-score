# -*- coding: utf-8 -*-
"""
Dictionary Expansion Pipeline
- Loads pre-trained Word2Vec model
- Expands vocabulary using seed words
- Saves expanded dictionary
"""

import os
from pathlib import Path
from gensim.models import Word2Vec
from global_options import DataPaths

def expand_dictionary():
    """Main function to execute dictionary expansion workflow"""
    try:
        # 1. LOAD PRE-TRAINED MODEL
        model_path = DataPaths.MODEL_DIR / "word2vec.model"
        print(f"Loading model from: {model_path}")
        w2v_model = Word2Vec.load(str(model_path))
        
        # 2. LOAD SEED WORDS
        print(f"Using seed words from: {DataPaths.SEEDWORDS_FILE}")
        with open(DataPaths.SEEDWORDS_FILE, 'r', encoding='utf-8') as f:
            seed_words = [word.strip() for word in f if word.strip()]
        
        # 3. PERFORM EXPANSION
        expanded_words = set(seed_words)  # Start with seed words
        for word in seed_words:
            try:
                # Get top 100 similar words
                similar = w2v_model.wv.most_similar(word, topn=100)
                expanded_words.update([w for w, _ in similar])
            except KeyError:
                print(f"[Warning] Seed word not in vocabulary: '{word}'")
                continue
        
        # 4. SAVE RESULTS
        output_path = DataPaths.DICTIONARIES_DIR / "expanded_dict.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted(expanded_words)))
        
        # 5. OUTPUT SUMMARY
        print("\nEXPANSION COMPLETE")
        print(f"Initial seeds: {len(seed_words)}")
        print(f"Expanded terms: {len(expanded_words)}")
        print(f"Saved to: {output_path}")

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        raise

if __name__ == "__main__":
    expand_dictionary()