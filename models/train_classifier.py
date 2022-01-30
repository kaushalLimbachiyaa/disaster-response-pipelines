import sys
import re
import pickle
import pandas as pd

from sqlalchemy import create_engine

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath):
    '''

    This function loads data from Sqlite Database

    Args :
    database_filepath - str - Database Filename

    Returns :
    X - DataFrame - Containig features
    Y - DataFrame - Containing all the categories 
    category_names - list - List of all category names

    '''
    # load data from database
    engine_url = 'sqlite:///' + database_filepath
    engine = create_engine(engine_url)
    df = pd.read_sql_table('categories',con=engine)

    X = df.message
    Y = df[df.columns[5:]]

    # get names of all 36 categories
    category_names = Y.columns

    return X, Y, category_names

def tokenize(text):
    '''

    This fuction will perform following operation on given text:

    -> Detect URLs in text and replace it with 'urlplaceholder'
    -> Normalize Text - convert text to lowecase and remove punctualization
    -> Tokenize Text - tokenize each text and remove stopwords from text
    -> Lemmatize Text - Lemmatize each text and remove leading and trainling space from text
    
    Args :
    text - str - a text to be tokenized
    
    Returns:
    text - str - Tokenized , Normalized and Lemmatized text
    
    '''

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # get list of all urls using regex
    detected_urls = re.findall(pattern=url_regex,string=text)
    
    # replace each url in text string with placeholder
    # Normalize text
    for url in detected_urls:
        text = re.sub(pattern=url_regex, repl='urlplaceholder', string=str(text))
        text = re.sub(r"[^a-zA-z0-9]"," ",text.lower())
    
    # tokenize text filtering stopwords
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words("english")]

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:

        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''

    This function will build a machine pipeline and define the paramerter to be tuned.
    
    This machine pipeline takes in the message column as input and outputs classification results
    on the other 36 categories in the dataset (multiple target variables).
    
    Pipeline has CountVectorizer and TfidfTransformer as Tranformers and RandomForestClassifier as a classfier.
    
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',MultiOutputClassifier(estimator = RandomForestClassifier()))
    ])
    

    parameters = {
        'clf__estimator__min_samples_split' : [2,4],
        'clf__estimator__min_samples_leaf' : [1,2],
        'clf__estimator__n_estimators': [5]
    }


    
    
    gridsearch_cv = GridSearchCV(pipeline,param_grid=parameters, cv=2, n_jobs=3, verbose=3)

    return gridsearch_cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''

    This fuction will do following operations:
    -> Predict on test data
    -> Return classification report with f1 score, precision and recall for each category of the dataset.

    '''
    Y_test = pd.DataFrame(data=Y_test, columns=category_names)

    # predict on the test data
    y_predcv = model.predict(X_test)

    # convert to dataframe
    y_predcv = pd.DataFrame(data=y_predcv, columns=category_names)

    return classification_report(Y_test, y_predcv)
   
def save_model(model, model_filepath):
    '''
    
    This fuction will export the model as pickle file to be used for making predictions later

    '''
    with open(model_filepath, 'wb') as model_file:
        pickle.dump(model,model_file)


def main():
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