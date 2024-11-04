from keras.callbacks import LearningRateScheduler
from keras.optimizers.schedules import PolynomialDecay
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM
from keras.src.losses import CategoricalCrossentropy, BinaryCrossentropy
from keras.src.optimizers import Adam, RMSprop
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelBinarizer
from transformers.optimization_tf import WarmUp
import tensorflow as tf
import emoji
from wordsegment import load, segment

from nltk.tokenize import TweetTokenizer as tt
from nltk.corpus import stopwords


import argparse
import re

load()
stop_words = set(stopwords.words('english'))

def create_arg_parser():
    """
    Create argument parser with all necessary arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--train_file", default='train.tsv', type=str,
                        help="Input file to learn from (default train.tsv)")
    parser.add_argument("-d", "--dev_file", type=str, default='dev.tsv',
                        help="Separate dev set to read in (default dev.tsv)")
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

def train_and_evaluate(X_train, Y_train, X_dev, Y_dev, max_seq_len, learning_rate, batch_size, epochs, use_scheduler):
    """Train a BERT model using the given parameters and evaluate on the dev set."""
    encoder = LabelBinarizer()
    Y_train_bin = tf.keras.utils.to_categorical(encoder.fit_transform(Y_train))
    Y_dev_bin = tf.keras.utils.to_categorical(encoder.transform(Y_dev))

    lm = "cardiffnlp/twitter-roberta-base-offensive"
    tokenizer = AutoTokenizer.from_pretrained(lm)
    tokens_train = tokenizer(X_train, padding=True, max_length=max_seq_len, truncation=True, return_tensors="tf")
    tokens_dev = tokenizer(X_dev, padding=True, max_length=max_seq_len, truncation=True, return_tensors="tf")

    model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=2)

    # Configure optimizer and compile the model
    if use_scheduler:
        linear_decay = PolynomialDecay(
            initial_learning_rate=learning_rate,
            end_learning_rate=0,
            decay_steps=epochs * (len(X_train) // batch_size)
        )
        warmup_schedule = WarmUp(
            warmup_learning_rate=0,
            after_warmup_lr_sched=linear_decay,
            warmup_steps=int(0.1 * epochs * (len(X_train) // batch_size))
        )
        optimizer = Adam(learning_rate=warmup_schedule)
    else:
        optimizer = Adam(learning_rate=learning_rate)


    model.compile(
        optimizer=optimizer,
        loss=BinaryCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    class EpochLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f"Epoch {epoch + 1} completed with accuracy: {logs['accuracy']}, validation accuracy: {logs['val_accuracy']} \n")

    model.fit(
        tokens_train.data,
        Y_train_bin,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(tokens_dev.data, Y_dev_bin),
        callbacks=[EpochLogger()]
    )

    Y_pred_logits = model.predict(tokens_dev.data).logits
    Y_pred_classes = Y_pred_logits.argmax(axis=1)

    val_accuracy = accuracy_score(Y_dev_bin.argmax(axis=1), Y_pred_classes)

    print(f"Validation Accuracy: {val_accuracy}")
    print(classification_report(Y_dev_bin.argmax(axis=1), Y_pred_classes))

    return val_accuracy


def main():
    """Use the train and dev corpora to train and evaluate a BERT model."""
    parser = create_arg_parser()
    args = parser
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)

    train_and_evaluate(X_train, Y_train, X_dev, Y_dev, 512, 1e-5, 16, 2, False)


if __name__ == '__main__':
    main()
