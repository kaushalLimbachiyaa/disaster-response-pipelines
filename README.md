# Disaster Response Pipeline Project


### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

Follwing are the nesesary libraries to run the code.
### Libraries used:
    1. pandas==1.2.2
    2. Matplotlib==3.3.4
    
The code should run with no issues using Python versions 3.9.6.

## Project Motivation<a name="motivation"></a>

I have analysed pre-labelled tweets and text messages from real life disaters provided by <a name="appen" href="https://appen.com/"> appen</a>, former Figure Eight Platform.
During natural disasters, there are millions of tweets and text messages sent to disaster response organizations either via . Disaster response organizations have to to filter and pull out the most important messages from this huge amount of communications a and redirect specific requests or indications to the proper organization that takes care of medical aid, water, logistics ecc. Every second is vital in this kind of situations, so handling the message correctly is the key

## File Descriptions <a name="files"></a>

Gender-Pay-Gap-In-India.ipynb : This notebooks available here to showcase work related to the above questions. You may downlaod the notebook and run on your local machine(Jupyter Notebook must be installed to do so). The notebooks is exploratory in searching through the data related to the questions showcased by the notebook title. Markdown cells were used to assist in walking through the thought process for individual steps.  


## Results<a name="results"></a>

The main findings of the code can be found at the post available [here](https://medium.com/@kaushal370/gender-pay-gap-in-it-sector-india-b1f09ed332a).

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Stack Overflow for the data.  You can find the Licensing for the data and other descriptive information at the Stack Overflow link available [here](https://insights.stackoverflow.com/survey). I have chosen survey Result Data for year 2021.



### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

![alt text](WebApp_TrainingDataset.png)
