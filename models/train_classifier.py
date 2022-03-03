import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys

import nltk
nltk.download('stopwords')
nltk.download('wordnet') # download for lemmatization
text = nltk.download('punkt')

from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import re
import pickle

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
#from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,mean_absolute_error
from sklearn.metrics import classification_report, precision_score, recall_score
from sklearn import metrics


def load_data(database_filepath):
    '''
    reads data from SQL database and split it into features and labels for furter model prediction
    :param database_filepath:
    :return: features (X), labels multioutput (y) and names of categories for an output (categories)
    '''

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM DisasterResponse", engine)
    X = df['message'].values
    y = df.iloc[:, 4:].values
    categories = list(df.columns[4:])
    return X, y, categories


def tokenize(text):
    '''
    clean, tokenize, lemmatize , removing stop words in input text and store into tokens
    :param text:
    :return: tokens
    '''
    # normalize case and remove punctuation
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    '''
    - builds a pipeline that processes text and then performs multi-output classification on the 36 categories in the dataset.
    -  GridSearchCV is used to find the best parameters for the model.
    - pipeline - vectorize , then applies tfidf transformer and then using ADABoostClassifier for
    multioutcome prediction
     
    :return: model
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',
         MultiOutputClassifier(AdaBoostClassifier(learning_rate=0.5,
           n_estimators=80)))
    ])

    parameters = {
                'clf__estimator__n_estimators': [50, 80],
                'clf__estimator__learning_rate': [0.01, 0.1, 0.5]
            }

    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=2, n_jobs=1, verbose=2, return_train_score=False)

    return cv


def evaluate_model(model, X_test, y_test, categories):
    '''
    takes model results and evaluates its performance : accuracy,precision and recall
    :param model:
    :param X_test:
    :param y_test:
    :param categories:
    :return:
    '''
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)



def save_model(model, classifier_filepath):
    '''
    saving model into pickle file by given user filepath
    :param model:
    :param classifier_filepath:
    :return:
    '''
    pickle.dump(model, open(classifier_filepath, 'wb'))


def main():
    '''
    1) takes in database and model classifier paths
    2) loading data and split it into training and test sets
    3) creates and evaluates model
    4)  saves model into pickle file

    :return:
    '''
    if len(sys.argv) == 3:   #sys. argv is the list of commandline arguments passed to the Python program.
        # argv represents all the items that come along via the command line input
        database_filepath, classifier_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format('data/database_filepath'))

        X, y, categories = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, categories)

        print('Saving model...\n    MODEL: {}'.format(classifier_filepath))
        save_model(model, classifier_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
