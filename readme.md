# Disaster Response Pipeline Project

# Overview

This project is part of Udacity Data Science project in collaboration with Figure Eight. 
During desasters there are a lot of messages simultaneously submitted of various needs, which might be timeconsuming to handle given the urgency of situation.
Purpose of this project to  build pipeline/model to classify each message by category of need (food , water, military, medical, child help , electricity ...) and send those in respective disaster relief agency for further faster help.
Project uses Natural Language Processing model and Classification model, to categorize these events.
The project includes a web app where an emergency worker can input a new message and get classification results in several categories.

# Content 

The project is divided into three components:

- ETL Pipeline: To load datasets, clean the data and store in one  SQLite database
- ML Pipeline: To build a text processing and machine learning pipeline, train a model to classify text message in categories
- Flask Web App: To show model results in real time


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/

# Results

1. Input a message to get a result

2. See a result as number of highlighted categories message might belong to

3. Training Dataset overview
    - Messages overview by genre
    - Distribution of messages by Categories

