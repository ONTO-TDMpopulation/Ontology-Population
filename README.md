# Ontology-Population

This repository contains dataset, ontology and a python script for ontology population using textrank and word2vec algorithms.

## Requirements

- Python 3.x
- NLTK
- Gensim
- NetworkX
- scikit-learn
- SpaCy

## Dataset

The dataset used for testing the script is located in the dataset directory. It consists of texts collected from a variety of sources, including Wikipedia, online articles, specialized websites, and forums where domain experts and mentors contribute valuable knowledge. Additionally, we have included online lecture slides from educational institutions, selected for their relevance to algorithms and data structures discipline. This ensures that the dataset covers both foundational and advanced concepts comprehensively. Each file in this directory has a .txt extension. The dataset comprises diverse textual documents, which are processed to extract important phrases.

## Ontology
The ontology used is based on the core ontology ONTO-TDM (Ontology for Teaching Domaon Modeling) specifically designed to represent and describe a discipline, focusing on its
application in learning-by-doing scenarios. The primary aim of this ontology is to provide a comprehensive framework that captures the essential elements of a discipline, including its
concepts, relationships, functions, and rules.
The resulting ontology can be found in the ontology directory.

## Code
The main Python script for text extraction is named text_extraction.py. It performs the following steps:

Preprocesses text by tokenizing, cleaning, and lemmatizing.
Trains a Word2Vec model on the text corpus.
Computes a similarity matrix between phrases using the trained Word2Vec model.
Converts the similarity matrix into a graph and applies TextRank to rank phrases.
Extracts and saves important phrases to a CSV file named phrases.csv.

The machine learning model training and prediction is named machine-learning-model.py, where a random forest classifier is trained over annotated data and words vector features.

The extracted and annotated data are migrated to the ontology using OwlReady2 api and the python script for that is named owlready2.py.

## Notes
Ensure that stop words.txt is in the same directory as your script or provide the correct path to it. it can be located in the directory as stopwords.txt.

Adjust the similarity threshold in the matrix_to_graph function if needed.



