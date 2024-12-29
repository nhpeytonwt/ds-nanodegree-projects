import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle

# ------------------------------------------------------------------------------------

def load_data(database_filepath):
    """ Load data from SQLite database.

    Args:
        database_filepath (str): Filepath for input SQLite database.

    Returns:
        X (ser): Data for independent variables.
        y (dataframe): Data for target variable.
        category_names (list): Category names for evaluation.
    """

    # Connect to SQL database and load data
    engine = create_engine(f'sqlite:///{database_filepath}')
    with engine.connect() as conn:
        query = text("SELECT * FROM disaster_response")
        df = pd.read_sql(query, con=conn)
    
    # Define X and y vars
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns.tolist()

    return X, Y, category_names

# ------------------------------------------------------------------------------------

def tokenize(text):
    """ Normalize, tokenize, remove stopwords, and lemmatize input text.

    Args:
        text (str): Text data.

    Returns:
        lemmatized (list): Cleaned tokens.
    """
    # Normalize: Remove all non-numeric characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize text
    tokenized = word_tokenize(text)

    # Remove stop words
    tokenized = [w for w in tokenized if w not in stopwords.words("english")]

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w) for w in tokenized]

    return lemmatized

# ------------------------------------------------------------------------------------

def build_model():
    """Build a pipeline including vectorization, tfidf, and predictive model that uses gridsearch for crossval

    Returns:
        cv (model): Fully set-up model.
    """
    # Set up pipeline (vectorize -> tfidf -> random forest)
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
    ])

    # Set up parameter grid
    parameters = {
        'tfidf__use_idf': [True, False], # Whether or not to use tfidf
        'clf__n_estimators': [50, 100]   # Number of trees in forest
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

# ------------------------------------------------------------------------------------

def evaluate_model(model, X_test, Y_test, category_names):
    """ Run model and print out results of classification report.

    Args:
        model (model): Trained model.
        X_test (ser): Test data for independent vars.
        y_test (dataframe): Test data for target var.
        category_names (list): List of category names for evaluation.
    """
    Y_pred = model.predict(X_test)

    for i, column in enumerate(category_names):
        print(f'Results for {column}')
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))

# ------------------------------------------------------------------------------------

def save_model(model, model_filepath):
    """ Saves out model pickle.

    Args:
        model (model): Trained model.
        model_filepath (str): Locatiojn to save pickle.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model.best_estimator_, file)
    print(f"Trained model saved as '{model_filepath}'")

# ------------------------------------------------------------------------------------

def main():
    """
    Main function to run the pipelin eand save results.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

# ------------------------------------------------------------------------------------