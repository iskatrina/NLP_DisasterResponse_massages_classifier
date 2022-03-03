import json
import plotly
import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

import plotly.graph_objects as go
from plotly.tools import make_subplots

app = Flask(__name__)


def tokenize(text):
    '''

    :param text:
    :return:
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


#getting data fro visualization
df_distribution = pd.DataFrame(df.iloc[:,4:])
categories_names = pd.DataFrame(df_distribution.mean().sort_values(ascending=False)).index
categories_data = (df_distribution.mean().sort_values(ascending=False).values).round(2)

# index webpage displays cool visuals and receives user input text for model
# index.html is a front-end representation
@app.route('/')
@app.route('/index')
def index():

    #Visualization 1 , default.
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
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
        },

        # Visualisation 2: Distribution of massages by Categories
        {
            'data': [
                Bar(
                    y=categories_data,
                    x=categories_names,

                )
            ],

            'layout': {
                'title': 'Distribution of Messages by Categories',
                'yaxis': {
                    'title': "Share"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 30,
                    'dtick': 1
                }
            }
        }
    ]




    # encode plotly graphs in JSON
    # create id-s for each figure and then convert them (ids and graphs)  to json files

    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    # pass data(which are within graphs already) from backend to frontend through the variable inside render_template()
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
# @app.route('/go', methods=["POST", "GET"])
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