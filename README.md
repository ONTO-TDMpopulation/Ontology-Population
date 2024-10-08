# Ontology-Population

This repository contains dataset, ontology and python scripts for ONTO-TDM (Ontology for Teaching Domain Modeling) population.

## Requirements

- Python 3.x
- NLTK
- Gensim
- NetworkX
- scikit-learn
- SpaCy
- OwlReady2 api

## Dataset

The dataset used for testing the script is located in the dataset directory "https://github.com/ONTO-TDMpopulation/Ontology-Population/tree/main/Dataset". It consists of texts collected from a variety of sources, including Wikipedia, online articles, specialized websites related to algorithms and data structures discipline. The dataset covers both foundational and advanced concepts comprehensively from this domain. Each file in this directory has a .txt extension. The dataset comprises diverse textual documents, which are processed to extract important phrases.

## Ontology
The ontology used is based on the core ontology ONTO-TDM specifically designed to represent and describe a discipline, focusing on its
application in learning-by-doing scenarios. The primary aim of this ontology is to provide a comprehensive framework that captures the essential elements of a discipline, including its
concepts, relationships, functions, and rules.
The resulting ontology can be found in the ontology directory "https://github.com/ONTO-TDMpopulation/Ontology-Population/tree/main/Ontology".

## Code
1. Text Extraction

The main Python script for text extraction is named text_extraction.py "https://github.com/ONTO-TDMpopulation/Ontology-Population/blob/main/text-extraction.py". It performs the following steps:

- Preprocesses text by tokenizing, cleaning, and lemmatizing.
- Trains a Word2Vec model on the text corpus.
- Computes a similarity matrix between phrases using the trained Word2Vec model.
- Converts the similarity matrix into a graph and applies TextRank to rank phrases.
- Extracts and saves important phrases to a CSV file.

2. Machine Learning Model
   
The machine learning model training and prediction are implemented in machine-learning-model.py "https://github.com/ONTO-TDMpopulation/Ontology-Population/blob/main/machinelearningmodel.py". This script trains a random forest classifier using annotated data and word vector features extracted from the dataset.

4. Ontology Population with Instances
   
The extracted and annotated data are migrated to the ontology using the OwlReady2 API. This process is managed by the owlready2.py script "https://github.com/ONTO-TDMpopulation/Ontology-Population/blob/main/Owlready2.py", which ensures that the extracted and annotated words and phrases are accurately integrated into the ontology, respecting the different classes and relationships defined in ONTO-TDM.





