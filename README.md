# Disaster Response Pipeline Project

The disaster response pipeline project is has 3 mai1n parts: 
1. The Jupyter Notebooks that were used in order to explore the data. They can be found under the folder "Notebooks"
It was also used to create initial versions of the python code for the pipeline.
2. The workspace with the training loading and training python files.  
They can be found under: workspace/data for the loading code and under "workspace/models" for the training and creation of the model.
3. The web app that makes use of the model is found ander: "workspace/app"    

# Imports 
1. pytest
2. import sys
3. import pandas as pd
4. from sqlalchemy import create_engine

# Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
