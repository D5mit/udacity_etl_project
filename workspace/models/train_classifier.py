import sys
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sqlalchemy import create_engine

import re
import pickle
import pandas as pd


def load_data(database_filepath):
    """
    Load data from database

    Parameters:
    database_filepath (string): the path of the database file
    """

    # df = pd.read_sql_table('MessageClass', 'sqlite:///DisasterResponse2.db')
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('MessageClass', engine)

    # get text (X) and labels of data (y)
    X = df.iloc[:, 1:2].values[:, 0]
    y = df.iloc[:, 4:].values

    #save the categories names
    category_names = df.columns[4:40].values
    return X, y, category_names



def replace_text_regex(text, iregex, iplaceholder):
    """
    replace texts based on regular expresion

    Parameters:
    text (string): Text that will be used to replace
    iregex (string): The regular expresion
    iplaceholder (string): If the regular expresion is found, the text is replaced with the placeholder texts

    Returns:
    the modified text
    """

    # remove URLs and replace with "urlplaceholder"
    url_regex = iregex

    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)

    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, iplaceholder)

    return text


def tokenize(text):
    """
    Takes text and clean it
    - Remove urls and replace with "urlplaceholder"
    - Remove twitter tages and replace with "twitterplaceholder"
    - Replace the String based on the pattern -> replace number with string
    - Lemmatize text
    - Return clean_tokens

    Parameters:
    clean_tokens (list): text as a list

    Returns:
    the modified text
    """

    # remove URLs and replace with "urlplaceholder"
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = replace_text_regex(text, url_regex, 'urlplaceholder')

    # remove www.URLs and replace with "urlplaceholder"
    url_regex = 'www.?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = replace_text_regex(text, url_regex, 'urlplaceholder')

    # remove twitter tages and replace with "twitterplaceholder"
    url_regex = '//t.co?/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = replace_text_regex(text, url_regex, 'twitterplaceholder')

    # Replace the String based on the pattern -> replace number with string
    text = re.sub('[-+]?[0-9]+', 'inumber', text).upper()

    # Replace the String based on the pattern
    text = text.replace('+', '')

    # Replace the String based on the pattern
    text = text.replace('.', '')

    # Replace the String based on the pattern
    text = text.replace("'", '')

    # Replace the String based on the pattern
    text = text.replace("!", '')

    # Replace the String based on the pattern
    text = text.replace("#", '')

    # Replace the String based on the pattern
    text = text.replace("(", '')

    # Replace the String based on the pattern
    text = text.replace(")", '')

    # Replace the String based on the pattern
    text = text.replace("*", '')

    # Replace the String based on the pattern
    text = text.replace("~", '')

    # Replace the String based on the pattern
    text = text.replace("<", '')

    # Replace the String based on the pattern
    text = text.replace(">", '')

    # Replace the String based on the pattern
    text = text.replace("@", '')

    # Replace the String based on the pattern
    text = text.replace('`', '')

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')


    clean_tokens = []
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    creates a pipeline to create a classifier
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [50],
        #                          'clf__estimator__max_features': max_features
        #                          'clf__estimator__max_depth': max_depth,
        'clf__estimator__min_samples_split': [3],
        'clf__estimator__min_samples_leaf': [3],
        #                          'clf__estimator__bootstrap': bootstrap
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=5)
    return cv

def evaluate_model(model, X_test, y_test, category_names, idetails=False):
    """
    Takes the predicted value and compare it to the test value. (y_pred vs y_test)
    - for each column use the classification_report function to calculate the 'macro avg': f1 score, recall and precision
    - Get the average of all the  f1 score, recall and precisions

    Parameters:
    model: the model taht will be evaluated
    x_test (numpy.ndarray): x values (test input)
    y_test (numpy.ndarray): y test values (test labeled data)
    idetails=False (boolean): output details

    Returns:
    prints the: f1 score, recall and precision
    """

    print(' - Predict...')
    y_pred = model.predict(X_test)

    print(' - Evaluate...')
    icolumns = category_names
    counter = 0                 # counts the number of F1 scores
    total_f1 = 0                # calculates the sum of all F1 scores
    average_f1 = 0              # average F1 for over all columns
    total_precision = 0
    total_recall = 0
    average_precision = 0
    average_recall = 0

    for column in icolumns:

        # get F1 scores
        report = classification_report(y_test[counter], y_pred[counter], output_dict=True)

        # use macro see blog:
        # https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9
        # Use F1
        macro_precision = report['macro avg']['precision']
        macro_recall = report['macro avg']['recall']
        macro_f1 = report['macro avg']['f1-score']

        # print output details
        if idetails == True:
            print('')
            print(column)
            print('F1 Score:', macro_f1)

        total_f1 = total_f1 + macro_f1
        total_precision = total_precision + macro_precision
        total_recall = total_recall + macro_recall
        counter = counter + 1

    average_f1 = total_f1 / counter
    print('f1 score: ', average_f1)

    average_recall = total_recall / counter
    print('Recall score: ', average_recall)

    average_precision = total_precision / counter
    print('Precision score: ', average_precision)


def save_model(model, model_filepath):
    """
    Save the model to a pkl file
    Parameters
    ----------
    model : str, mandatory
        model to save

    model_filepath : str, mandatory
        path to where the model must be saved  """

    # Save to file in the current working directory
    pkl_filename = model_filepath
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

def main():
    """
    Main function of the program that will do the following
        Load data...
        Build model
        Train model
        Evaluate model
        Save model...
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        if database_filepath == 'default.db':   #make it easier to load
            database_filepath = 'sqlite:////Users/d5mit/PycharmProjects/Udacity_ETL/Project/workspace/data/DisasterResponse.db'

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names, True)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py sqlite:////Users/d5mit/PycharmProjects/udacity_ETL_project/Project/workspace/data/DisasterResponse.db classifier.pkl')



if __name__ == '__main__':
    main()