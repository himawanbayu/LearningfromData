import random as python_random
import json
import argparse
import numpy as np
from keras.src.models import Sequential
from keras.src.layers import Dense, Embedding, LSTM, Bidirectional
from keras.src.initializers import Constant
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelBinarizer
from keras.src.optimizers import SGD, Adam, RMSprop
from keras.src.layers import TextVectorization
from keras import callbacks
import tensorflow as tf
import re
import emoji
from wordsegment import load, segment
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sklearn.preprocessing import OneHotEncoder

np.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)

load()
tt = TweetTokenizer()
stop_words = set(stopwords.words('english'))

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--train_file", default='train.tsv', type=str,
                        help="Input file to learn from (default train.tsv)")
    parser.add_argument("-d", "--dev_file", type=str, default='dev.tsv',
                        help="Separate dev set to read in (default dev.tsv)")
    parser.add_argument("-t", "--test_file", type=str,
                        help="If added, use trained model to predict on test set")
    parser.add_argument("-e", "--embeddings", default='glove.6B.100d.txt', type=str,
                        help="Embedding file we are using (default glove.6B.100d.txt)")
    args = parser.parse_args()
    return args

def convert_emojis_to_words(text):
    return emoji.demojize(text, delimiters=(" ", " "))

def preprocess_text(text):
    text = re.sub(r'@[\w]+', '', text)
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = convert_emojis_to_words(text)
    text = re.sub(r'#', '', text)

    parts = text.split()
    processed_tokens = []
    
    for part in parts:
        segmented = segment(part) if part.isalpha() and len(part) > 6 else [part]
        processed_tokens.extend(segmented)

    tokens = [token.lower() for token in processed_tokens if token.lower() != 'url']
    processed_text = ' '.join(tokens)
    processed_text = re.sub(r'(\W\s?)+', ' ', processed_text).strip()
    return processed_text


def read_corpus(corpus_file):
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as in_file:
        for line in in_file:
            # Split tweet and label based on the last tab character
            tweet, label = line.rsplit('\t', 1)
            tweet = preprocess_text(tweet.strip())
            documents.append(tweet)
            labels.append(label.strip())
    return documents, labels

def read_embeddings(embeddings_file):
    '''Read in word embeddings from a GloVe .txt file and save as a dictionary'''
    embeddings = {}
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings


def get_emb_matrix(voc, emb):
    num_tokens = len(voc) + 2
    word_index = dict(zip(voc, range(len(voc))))
    embedding_dim = len(emb["the"])
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = emb.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def create_model(Y_train, emb_matrix):
    learning_rate = 0.001
    # loss_function = 'categorical_crossentropy'
    loss_function = 'BinaryCrossentropy'
    optim = RMSprop(learning_rate=learning_rate)
    embedding_dim = len(emb_matrix[0])
    num_tokens = len(emb_matrix)
    # num_labels = len(set(Y_train))
    
    model = Sequential()
    model.add(Embedding(num_tokens, embedding_dim, embeddings_initializer=Constant(emb_matrix), trainable=True))
    model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)))
    model.add(Bidirectional(LSTM(32, return_sequences=False, dropout=0.5, recurrent_dropout=0.5)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(units=2, activation="softmax"))
    
    model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])
    return model

def train_model(model, X_train, Y_train, X_dev, Y_dev, encoder):
    batch_size = 32
    epochs = 50
    callback = callbacks.EarlyStopping(monitor='val_loss', patience=3)
    
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_dev, Y_dev), callbacks=[callback], verbose=1)
    test_set_predict(model, X_dev, Y_dev, "dev", encoder)
    return model

def test_set_predict(model, X_test, Y_test, ident, encoder):
    Y_pred = model.predict(X_test)
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    
    print('Accuracy on {} set: {:.3f}'.format(ident, accuracy_score(Y_test, Y_pred)))
    print(f'Classification Report for {ident} set:\n{classification_report(Y_test, Y_pred, target_names=encoder.classes_)}')

def main():
    args = create_arg_parser()

    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)
    embeddings = read_embeddings(args.embeddings)

    vectorizer = TextVectorization(standardize=None, output_sequence_length=50)
    text_ds = tf.data.Dataset.from_tensor_slices(X_train + X_dev)
    vectorizer.adapt(text_ds)
    
    voc = vectorizer.get_vocabulary()
    emb_matrix = get_emb_matrix(voc, embeddings)

    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)
    Y_dev_bin = encoder.transform(Y_dev)

    # Ensure labels are one-hot encoded
    Y_train_bin = np.eye(2)[Y_train_bin.ravel()]  # Convert to shape (num_samples, 2)
    Y_dev_bin = np.eye(2)[Y_dev_bin.ravel()]

    model = create_model(Y_train, emb_matrix)

    X_train_vect = vectorizer(np.array([[s] for s in X_train])).numpy()
    X_dev_vect = vectorizer(np.array([[s] for s in X_dev])).numpy()

    model = train_model(model, X_train_vect, Y_train_bin, X_dev_vect, Y_dev_bin, encoder)

    if args.test_file:
        X_test, Y_test = read_corpus(args.test_file)
        Y_test_bin = encoder.transform(Y_test)
        X_test_vect = vectorizer(np.array([[s] for s in X_test])).numpy()
        test_set_predict(model, X_test_vect, Y_test_bin, "test", encoder)

if __name__ == '__main__':
    main()
