# Disaster Response Pipeline Project

This project includes a web app where an emergency worker can input 
a new message and get classification results in several categories.
The web app displays visualizations of the data. In order to build 
the app a basic data pipelines and machine learning pipeline was
 developed.

The disaster response pipeline project is has 3 mai1n parts: 
1. The Jupyter Notebooks that were used in order to explore the data. They can be found under the folder "Notebooks"
It was also used to create initial versions of the python code for the pipeline.
2. The workspace with the training loading and training python files.  
They can be found under: workspace/data for the loading code and under "workspace/models" for the training and creation of the model.
3. The web app that makes use of the model is found ander: "workspace/app"    

# Imports (packages to install via pip):
- pytest
- sys
- pandas
- sqlalchemy
- sklearn
- nltk
- re
- pickle
- json
- plotly
- matplotlib
- wordcloud
- flask
- joblib

# Overview of project:
### Notebooks:
- ETL Pipeline Preparation.ipynb # notebook to explore and create ETL
- ML Pipeline Preparation.ipynb # notebook to create ML pipeline
- ML Pipeline Preparation-visualization.ipynb # notebook to visualize data
### Workspace:
#### app
- run.py # Flask file that runs app
##### /static
- Plotly-world_Cloud.png # created wordcloud
##### /templates
 - master.html # main page of web app
 - go.html # classification result page of web app

#### data
- disaster_categories.csv # data to process
- disaster_messages.csv # data to process
- process_data.py # process the files to create a db
- Disasterresponse.db # database to save clean data to
- test_process_data.py # use pytest to unit test the data load program

#### models
- train_classifier.py # train the model
- classifier.pkl # saved model

README.md

# Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
        note: pytest was used in order to automate the testing for the process data script.
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py ../data/DisasterResponse.db models/classifier.pkl`
    - The model that was used in order to train the pipeline on:
        - CountVectorizer(tokenizer=tokenize))
        - TfidfTransformer())
        - MultiOutputClassifier(RandomForestClassifier())
        
        With the following parameters:
        - clf__estimator__n_estimators = 50
        - clf__estimator__min_samples_split = 3
        - clf__estimator__min_samples_leaf = 3


2. Run the following command in the app's directory to run your web app.
    `python run.py`
    
3. A working version of the application can be found on:
    - www.d5mit.co.za  (go to the Disaster Response link on the main page).
    - The website is hosted on AWS EC2.


