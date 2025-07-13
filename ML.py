# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 14:40:12 2025

@author: FM
"""

import cntext as ct
import os
from global_options import DataPaths

# Initialize model with current directory and English setting
model = ct.W2VModels(cwd=os.getcwd(), lang='english')

# Train word2vec model using document corpus from global paths
model.train(input_txt_file=str(DataPaths.PROCESSED_DIR / "processed_text.txt"))

# Expand dictionary using seed words from global paths
model.find(seedword_txt_file=str(DataPaths.SEEDWORDS_FILE), 
           topn=100)