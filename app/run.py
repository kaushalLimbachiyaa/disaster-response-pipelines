import json
import plotly
import pandas as pd
import re
import nltk 

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

# def tokenize(text):
#     tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()

#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#         clean_tokens.append(clean_tok)

#     return clean_tokens

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


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('categories', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    def get_counts_names(text):
        '''
        This fuction will return counts and column names for text
        
        Args:
        text - str - Value of genre/category
        
        Returns:
        counts - list -List of Category Counts
        names - list - List of Category names
        
        '''
        #get counts and names for gener-wise categories
        if text in ('social','direct','news'):
            counts = [ df[(df[col]==1) & (df['genre']==text) ]['message'].count() for col in df.columns[5:] ] 
            names = [col for col in df.columns[5:]]
            return counts , names
        
        #get counts and names for water/ air-related / weather_related category
        else:
            counts = df[df[text] ==1].groupby('genre').count()['message']
            names = list(counts.index)
            return counts , names
            
    #prepare graph 1  
    social_genre_counts , social_genre_names = get_counts_names('social')
    direct_genre_counts ,direct_genre_names = get_counts_names('direct')
    news_genre_counts , news_genre_names = get_counts_names('news')

    #prepare graph 2
    water_counts, water_names = get_counts_names('water')
    aid_related_counts, aid_related_names = get_counts_names('aid_related')
    weather_related_counts, weather_related_names = get_counts_names('weather_related')

    # create visuals
    graphs = [
        #graph 1
        {
            'data': [
                Scatter(
                    x=news_genre_names,
                    y=news_genre_counts,
                    name = 'News Genre Counts'
                ),
                Scatter(
                    x=direct_genre_names,
                    y=direct_genre_counts,
                    name = 'Direct Genre Counts'
                ),
                Scatter(
                    x=social_genre_names,
                    y=social_genre_counts,
                    name = 'Social Genre Counts'
                )
            ],

            'layout': {
                'title': 'Distribution of Genre across Categories',
                'yaxis': {
                    'title': "Genre Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        #graph 2
        {
            'data': [
                Bar(
                    x=water_names,
                    y=water_counts,
                    name = 'water',
                    yaxis = water_names
                ),
                Bar(
                    x=aid_related_names,
                    y=aid_related_counts,
                    name = 'aid_related'
                ),
                Bar(
                    x=weather_related_names,
                    y=weather_related_counts,
                    name = 'weather_related'
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[5:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()