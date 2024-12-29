import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine, text


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///./data/DisasterResponse.db')
with engine.connect() as conn:
        query = text("SELECT * FROM disaster_response")
        df = pd.read_sql(query, con=conn)

# load model
model = joblib.load("./models/classifier.pkl")


# index webpage displays visuals and receives user input text
@app.route('/')
@app.route('/index')
def index():

    # Example visuals (adjust as needed)
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_counts = df.drop(['id', 'message', 'original', 'genre'], axis=1).sum().sort_values(ascending=False)
    category_names = list(category_counts.index)

    graphs = [
        {
            'data': [ # Pie chart showing message sources (genres)
                {
                    'type': 'pie',
                    'labels': genre_names,
                    'values': genre_counts,
                    'hole': 0.2,  
                    'marker': {
                        'colors': ['orange', 'green', 'purple']
                    }
                }
            ],
            'layout': {
                'title': 'Distribution of Message Source',
            }
        },
        {       # Bar chart showing the top 10 categories by message count
            'data': [
                Bar(
                    # Show the top 10 categories
                    x=category_names[:10], 
                    y=category_counts[:10],
                    marker={'color': 'purple'}  
                )
            ],
            'layout': {
                'title': 'Top 10 Categories by Message Count',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Category"}
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