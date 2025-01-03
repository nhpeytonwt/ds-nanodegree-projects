# Disaster Response Pipeline Project

## Instructions:

### Running files:
[GitHub Project Repository](https://github.com/nhpeytonwt/ds-nanodegree-projects/tree/main/Project-2-Disaster-Pipeline)

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves out results in pickle
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001/ 

## Required libraries
- sys
- numpy
- pandas
- sqlalchemy
- re
- nltk
- sklearn
- pickle
- json
- plotly
- flask
- joblib

## Qualitative Overview: CRISP-DM

### Businses Understanding
- During a disaster, time is of the essence. Our goal is to classify disaster-related messages into one of 36 sub-categories (e.g., food, water, shelter etc.) so that they can be routed to the appropriate authorities.

### Data Understanding
- The data consists of a dataset of messages and a dataset of categories.
- Messages contains disaster messages translated to English along with their source while categories attributes messages to a set of 36 possible categories.
- The datasets are merged on the "id" column.

### Data Preparation
The ETL pipeline performs the following tasks:
- Splits categories into separate columns
- Converts the new columns to binary format
- Drop duplicate columns

### Modeling
The ML pipeline is used to classify messages:
- Tokenize, vectorize, remove stop words, and lemmatize text.
- Classify via Random Forest trained with a grid search.

### Evaluation
Model results are evaluated on the basis of:
- Precision: Percentage of correct true positive predictions.
- Recall: Percentage of true positives correctly predicted.
- F1-Score: Weighted avearge of precision nad recall.

### Deployment
Trained modle is deployed via a Flask web app, which includes:
- Ability for users to classify new disaster responses.
- Charts visualizing disaster data.

## Acknowledgements
- Thanks to **Udacity** for providing the dataset and project framework.
- Additional thanks to the authors of any open-source libraries used in the project.