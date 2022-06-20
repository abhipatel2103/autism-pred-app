# Autism web app deployed on Heroku

This deployed web app is live at .

This web app predict whether person has Autism Disorder as a function of their input parameters

### The web app was built on python using the following libraries:

* streamlit
* pandas
* numpy
* scikit-learn
* pickle
* matplotlib and seaborn

## Tried and compared performance of the following Classification Models:
1) Logistic Regression
2) Support Vector Machine (SVC)
3) Decision Tree Classifier
4) Random Forest Classifier
5) Extra Tree Classifier


## File Structure

1) `train.csv` = Dataset provided to train ML model for Autism Disorder Prediction

2) `dataExploration.ipnyb` = Performed EDA and basic data exploration. Also, infered required pre-processing step to be implemented
3) `dataPreprocess.ipnyb` = Implemented pre-processing steps by creating data Pipeline, ColumnTransformers and FunctionTransformers. 
                            Saved transformed data into CSV `preprocessed_data.csv`.
4) `model-training.ipnyb` = Built ML models by feeding pre-processes data and compared performance. Saved best model to make predcitions on User input
5) `autism-app.py` = Created Webapp using **Streamlit**. Accepted user input and made prediction using already saved Pipeline and Model. 

6) `requirements.txt` = Package installation requirements to run *stramlit-app.py* file on server.
7) `Procfile` = Commands to execute that run webapp on server
8) `setup.sh` = Server side configuration




