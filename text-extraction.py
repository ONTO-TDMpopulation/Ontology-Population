import os
import csv
import re
import nltk
import gensim
import numpy as np
import networkx as nx
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.en.stop_words import STOP_WORDS
import spacy

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the Spacy NLP model
nlp = spacy.load('en_core_web_sm')

# Load custom stop words
with open('stop words.txt', 'r') as f:
    custom_stop_words = f.read().splitlines()

# Add custom stop words to Spacy's default stop words
for word in custom_stop_words:
    lexeme = nlp.vocab[word]
    lexeme.is_stop = True

# Preprocess text: tokenization, cleaning, lemmatization, and stop word removal
def preprocess_text(text):
    doc = nlp(text)
    phrases = set()  # Use a set to avoid duplicates
    for chunk in doc.noun_chunks:
        # Normalize phrase with lemmatization and remove hyphens
        phrase = ' '.join(token.lemma_.lower().replace('-', '') for token in chunk if token.lemma_ not in STOP_WORDS and not token.is_digit)
        # Ensure 'data' remains unchanged
        phrase = re.sub(r'\bdatum\b', 'data', phrase)
        # Clean special characters
        phrase = re.sub(r'[^\w\s]', '', phrase)
        if phrase:
            phrases.add(phrase)
    print(f"Processed phrases: {list(phrases)}")  # Debugging line
    return list(phrases)

# Train the Word2Vec model
def train_word2vec_model(sentences):
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    model = gensim.models.Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)
    return model

# Compute similarity matrix from phrases
def create_similarity_matrix(phrases, model):
    phrase_vectors = []
    for phrase in phrases:
        phrase_vector = np.zeros((model.vector_size,))
        tokens = nltk.word_tokenize(phrase)
        num_tokens = len(tokens)
        for token in tokens:
            if token in model.wv:
                phrase_vector += model.wv[token]
        if num_tokens > 0:
            phrase_vector /= num_tokens
        phrase_vectors.append(phrase_vector)
    
    similarity_matrix = cosine_similarity(phrase_vectors)
    print(f"Similarity matrix: {similarity_matrix}")  # Debugging line
    return similarity_matrix

# Convert similarity matrix to a graph
def matrix_to_graph(similarity_matrix, phrases):
    graph = nx.Graph()
    num_phrases = len(phrases)
    
    # Add nodes
    for i, phrase in enumerate(phrases):
        graph.add_node(i, label=phrase)
    
    # Add edges based on similarity matrix
    for i in range(num_phrases):
        for j in range(i + 1, num_phrases):
            if similarity_matrix[i, j] > 0.5:  # Similarity threshold
                graph.add_edge(i, j, weight=similarity_matrix[i, j])
    
    return graph

# Apply TextRank to extract important phrases
def extract_important_phrases(text, model):
    phrases = preprocess_text(text)
    if len(phrases) == 0:
        return []  # Return an empty list if no phrases were extracted
    
    similarity_matrix = create_similarity_matrix(phrases, model)
    graph = matrix_to_graph(similarity_matrix, phrases)
    pagerank_scores = nx.pagerank(graph)
    ranked_phrases = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Ranked phrases: {ranked_phrases}")  # Debugging line
    return [(phrases[i], score) for i, score in ranked_phrases]

# Main process
def process_corpus(corpus_dir):
    all_sentences = []
    for filename in os.listdir(corpus_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(corpus_dir, filename), 'r') as f:
                text = f.read()
                sentences = preprocess_text(text)
                all_sentences.extend(sentences)
    
    if len(all_sentences) == 0:
        print("No sentences found in the corpus.")
        return

    # Train the Word2Vec model
    word2vec_model = train_word2vec_model(all_sentences)

    # Extract important phrases from each file
    phrases_by_file = {}
    for filename in os.listdir(corpus_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(corpus_dir, filename), 'r') as f:
                text = f.read()
                ranked_phrases = extract_important_phrases(text, word2vec_model)
                phrases_by_file[filename] = [(phrase, score) for phrase, score in ranked_phrases]

    # Save important phrases to a CSV file, removing duplicates and excluding the vector_representation column
    with open('phrases.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['File_name', 'Phrase'])
        unique_phrases = set()  # Use a set to avoid duplicates
        for file_name, phrases in phrases_by_file.items():
            for phrase, score in phrases:
                # Clean the phrase to remove unwanted characters
                clean_phrase = re.sub(r'[^\w\s]', '', phrase)
                if clean_phrase not in unique_phrases:  # Check to avoid duplicates
                    unique_phrases.add(clean_phrase)
                    writer.writerow([file_name, clean_phrase])

    # Confirmation of writing
    print("Data written to phrases.csv")

# Run the corpus processing
process_corpus('corpus')
