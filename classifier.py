import argparse
import re

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import emoji
from wordsegment import load, segment

from feature_prep import test_vec_parameters, test_combining_vecs, test_preprocessing

load()
tt = TweetTokenizer()
stop_words = set(stopwords.words('english'))

def create_arg_parser():
    """
    Create argument parser with all necessary arguments.
    To see all arguments run the script with the -h flag.

    :return: The arguments for the current run
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_file", default='train.tsv', type=str,
                        help="Train file to learn from (default train.txt)")
    parser.add_argument("-d", "--dev_file", default='dev.tsv', type=str,
                        help="Dev file to evaluate on (default dev.txt)")
    parser.add_argument('-features', '--test_features', action='store_true',
                        help='Test the best way to prepare the features.')
    parser.add_argument("-c", "--classifier", default='nb',
                        help="Classifier to use (default Naive Bayes)")
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

def identity(inp):
    """
    Dummy function that just returns the input
    """
    return inp


def custom_preprocessor(tokens):
    """
    Custom preprocessor to remove non-alphabetic tokens.
    The RegEx pattern matches alphabetic strings while allowing optional apostrophes (n't, don't, etc.).
    """
    pattern = re.compile(r"^[a-zA-Z]+(?:'\w+)?$")

    return [token for token in tokens if pattern.match(token)]


def get_default_vectorizer():
    """
    Returns the vectorizer setup which was found most effective during feature testing.

    :return: The default vectorizer
    """
    return TfidfVectorizer(
        max_df=0.9,
        min_df=2,
        ngram_range=(1, 2),
        max_features=10000,
        preprocessor=custom_preprocessor,
        # stop_words='english',
        tokenizer=identity,
        token_pattern=None
    )


def main():
    """
    Main function to run the script.
    """
    args = create_arg_parser()

    X_train, Y_train = read_corpus(args.train_file)
    X_test, Y_test = read_corpus(args.dev_file)

    # Run the feature preprocessing
    if args.test_features:
        test_vec_parameters(
            X_train,
            Y_train,
            X_test,
            Y_test,
            [custom_preprocessor, identity],
            identity)
        test_combining_vecs(X_train, Y_train, X_test, Y_test, custom_preprocessor, identity)
        test_preprocessing(X_train, Y_train, X_test, Y_test, identity)

        exit()

    vec = get_default_vectorizer()

    # Choose the classifier
    param_dist = {}
    match args.classifier:
        case 'nb':
            classifier = Pipeline([('vec', vec), ('cls', MultinomialNB())])
            param_dist = {
                'cls__alpha': [2],
                'cls__fit_prior': [True, False]
            }
        case 'svm':
            classifier = Pipeline([('vec', vec), ('cls', SVC(probability=True, kernel='linear'))])
        case 'knn':
            classifier = Pipeline(
                [('vec', vec), ('cls', KNeighborsClassifier(n_neighbors=11, weights='distance', metric='euclidean'))])
        case 'dt':
            classifier = Pipeline([('vec', vec), ('cls', DecisionTreeClassifier())])
            param_dist = {
                'cls__max_depth': [25],
                'cls__criterion': ['gini'],
                'cls__min_samples_leaf': [4],
                'cls__min_samples_split': [2],
                'cls__max_features': [None]
            }
         case 'rf':
             classifier = Pipeline(
                 [('vec', vec), ('cls', RandomForestClassifier(n_estimators=500, max_depth=40, min_samples_leaf=2))])
        case 'all':
            classifier = Pipeline([('vec', vec), ('cls', VotingClassifier(voting='soft', estimators=[
                ('nb', MultinomialNB()),
                ('svm', SVC(probability=True, kernel='linear')),
                ('knn', KNeighborsClassifier(n_neighbors=11, weights='distance', metric='euclidean')),
                ('rf', RandomForestClassifier(n_estimators=500, max_depth=40, min_samples_leaf=2))
            ]))])
        case _:
            raise ValueError(f"Invalid classifier: {args.classifier}")

    param_search = GridSearchCV(classifier, param_grid=param_dist, cv=5, n_jobs=-1, verbose=2)
    param_search.fit(X_train, Y_train)
    print("\nBest parameters set found on training set:")
    print(param_search.best_params_)
    print("\nMaximum accuracy found on dev set:")
    print(param_search.best_score_)
    Y_pred = param_search.predict(X_test)

    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(Y_test, Y_pred))


if __name__ == "__main__":
    main()



 