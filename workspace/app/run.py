import json
import plotly
import pandas as pd

# matplotlib library
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# from sklearn.externals import joblib
from sqlalchemy import create_engine
import joblib

from wordcloud import WordCloud
import re
from nltk.corpus import stopwords

app = Flask(__name__)


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


# load data
# db_path = 'sqlite:////Users/d5mit/PycharmProjects/udacity_etl_project/workspace/data/DisasterResponse.db'
db_path = 'sqlite:///../data/DisasterResponse.db'
engine = create_engine(db_path)


df = pd.read_sql_table('MessageClass', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals on genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # extract data on request vs offers
    df_req_off = df.iloc[:, 5:7]            #request and offers
    df_req_off_sum = df_req_off.sum(axis=0, skipna=True)
    itype = df_req_off_sum.index
    inumber = df_req_off_sum.values

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of type of disasters',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=itype,
                    y=inumber
                )
            ],

            'layout': {
                'title': 'Distribution of Requests vs Offers',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Request and offers"
                }
            }
        },
    ]

    # add wordcloud
    x_words = df['message'][df['related'] > 0]

    plt.subplots(figsize=(8, 8))

    wordcloud = WordCloud(
        background_color='white'
        # width=512,
        # height=384
    ).generate(' '.join(x_words))
    plt.imshow(wordcloud)  # image show
    plt.axis('off')  # to off the axis of x and y
    plt.savefig('../app/static/Plotly-World_Cloud.png')

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
    classification_results = dict(zip(df.columns[4:], classification_labels))

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