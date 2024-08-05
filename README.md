# Ontology-Population
# Text Extraction

This repository contains a Python script for extracting important phrases from text using Word2Vec and TextRank algorithms.

## Requirements

- Python 3.x
- NLTK
- Gensim
- NetworkX
- scikit-learn
- SpaCy

## Dataset
The dataset used for testing the script is located in the dataset directory. Each text file in this directory should have a .txt extension. The dataset contains various textual documents that are processed to extract important phrases.

## Ontology
The ontology used is based on the core ontology ONTO-TDM and includes additional domain-specific terms and relations. The resulting ontology can be found in the ontology directory.

## Code
The main Python script for text extraction is named text_extraction.py. It performs the following steps:

Preprocesses text by tokenizing, cleaning, and lemmatizing.
Trains a Word2Vec model on the text corpus.
Computes a similarity matrix between phrases using the trained Word2Vec model.
Converts the similarity matrix into a graph and applies TextRank to rank phrases.
Extracts and saves important phrases to a CSV file named phrases.csv.

## Notes
Ensure that stop words.txt is in the same directory as your script or provide the correct path to it. it can be located in the directory as stopwords.txt.
Adjust the similarity threshold in the matrix_to_graph function if needed.
