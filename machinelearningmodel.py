from pathlib import Path

from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas
from sklearn.pipeline import Pipeline


# Generate word2vec model from free text.
def generate_word2vec_model(source_text) -> Word2Vec:
    """
    Generate a word2vec model from a folder containing text data.
    :param source_text: Path to a folder containing text files.
    :return: A word2vec model
    """
    sentences = PathLineSentences(source_text)
    model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)
    # Uncomment to see the words with the vectors in the console:
    # for key in model.wv.key_to_index.keys():
    #    print(f"{key}\t->\t{model.wv[key]}")
    return model


# Include the vectors to the annotated dataset
def include_vectors_in_annotated_dataset(word2vec_model: Word2Vec, annotated_data_path: Path, output_file_path: Path):
    """
    Reads a CSV file including a first column indicating a word (or noun phrase), and a second column indicating
    the class associated to that word. Then It generates a new CSV file with a first column called 'word', several
    columns named 'f0', 'f1', 'f2', ..., 'fn' with each feature of the vector associated to the word, and a final column
    named 'word_class', which is the class associated to the word.
    :param word2vec_model: The word2vec model used for extracting the vector of the words from the input CSV file
    :param annotated_data_path: Path to the input CSV file.
    :param output_file_path: Path to the output CSV file.
    :return: The output is written in output_file_path.
    """
    with open(annotated_data_path, 'r') as annotated_data, open(output_file_path, 'w+') as output_file:
        line_number = 0

        for line in annotated_data.readlines():
            line_number = line_number + 1
            if line_number == 1:
                # Print header
                header = "word," + (",".join(['f' + str(n) for n in range(0, word2vec_model.vector_size)])) + ",word_class"
                output_file.write(header + "\n")
            else:
                # Read the word and the word_class from your annotated file
                word = line.strip().split(',')[0]
                word_class = line.strip().split(',')[1]

                # Search the corresponding vector in the model
                # If the word is in the dictionary, return it directly
                if word2vec_model.wv.has_index_for(word):
                    vector = word2vec_model.wv[word]
                # If not, it is possible that it is a multiword, so get the mean vector of each word separately
                else:
                    vector = word2vec_model.wv.get_mean_vector(word.split(" "))

                # print the word, the vector, and the word class into the output file
                vector_str = ",".join([str(x) for x in vector])
                output_file.write(f"{word},{vector_str},{word_class}\n")


# Train a machine learning model from the annotated dataset
def train_ml_model(input_data: Path) -> Pipeline:
    """
    Train a machine learning model for word classification. It prints in the console a report about the precision
    reached.
    :param input_data: A path to a CSV file including a column named 'word', several columns named 'f0', 'f1', 'f2',
    ..., 'fn' indicating each component of the vector of the corresponding word, and a column called 'word_class',
    indicating the ontology class in which the word should be classified.
    :return: A pipeline that can be used to predict the class of a vector.
    """
    dataframe = pandas.read_csv(input_data)
    x_train, x_test, y_train, y_test = train_test_split(
        dataframe.drop(['word', 'word_class'], axis='columns'),
        dataframe['word_class'],
        train_size=0.8,
        random_state=1234,
        shuffle=True
    )

    pipe = Pipeline([('model', RandomForestClassifier())])
    pipe.fit(X=x_train, y=y_train)
    predictions = pipe.predict(X=x_test)

    print(
        classification_report(
            y_true=y_test,
            y_pred=predictions
        )
    )
    return pipe


# Predict new words
def predict(word, word2vec_model: Word2Vec, ml_model: Pipeline) -> list:
    """
    Predict the class of a word.
    :param word: The word to predict. It should exist in the word2vec model vocabulary.
    :param word2vec_model: The word2vec model to retrieve the vector of the word.
    :param ml_model: The machine learning model that receives the word vector and make the prediction.
    :return: A list of predicted classes for the word.
    """
    if word2vec_model.wv.has_index_for(word):
        word_vector = word2vec_model.wv[word]
    else:
        word_vector = word2vec_model.wv.get_mean_vector(word.split(" "))

    # To make the prediction, the ml model expects a pandas dataframe. This dataframe will contain a single row with
    # the vector of the word, and a header indicating the name of the features generated by word2vec, which are
    # 'f0', 'f1', 'f2', ..., 'fn'
    input_dataframe = pandas.DataFrame(data=[word_vector], columns=['f' + str(n) for n in range(0, word2vec_model.vector_size)])
    return ml_model.predict(input_dataframe)


if __name__ == '__main__':
    # -----PARAMETERS-----

    # Folder with the free text files.
    free_text_folder = Path("resources/free_text/")
    # CSV file with your annotated data
    annotated_data_path = Path("resources/annotated_data.csv")
    # CSV file that will be generated including the vectors in your annotated data.
    annotated_data_vectors_path = Path("resources/annotated_data_vectors.csv")

    # -----STEPS-----

    # 1. Generate word2vec model from free text.
    word2vec_model = generate_word2vec_model(free_text_folder)

    # 2. Generate annotated dataset, including the word, the vector and the class
    include_vectors_in_annotated_dataset(word2vec_model, annotated_data_path, annotated_data_vectors_path)

    # 3. Train a machine learning model from the annotated data
    ml_model = train_ml_model(annotated_data_vectors_path)

    # 4. Make predictions on new words (new words should be in the vocabulary of the word2vec model because you need
    # to get their vectors.)
    word = 'algorithm'
    prediction = predict(word, word2vec_model, ml_model)
    print(f"The word '{word}' is predicted as {prediction}.")

    word = 'tree'
    prediction = predict(word, word2vec_model, ml_model)
    print(f"The word '{word}' is predicted as {prediction}.")

    word = 'Recursion'
    prediction = predict(word, word2vec_model, ml_model)
    print(f"The word '{word}' is predicted as {prediction}.")

    word = 'Travelling Salesman Problem'
    prediction = predict(word, word2vec_model, ml_model)
    print(f"The word '{word}' is predicted as {prediction}.")
