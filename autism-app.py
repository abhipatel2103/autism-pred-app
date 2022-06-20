
#Importing Required libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import dataPreprocess  #importing dataPreprocess.py module so that functions and classes can be available here


#Creating Heading of Webpage
st.write("""
    ### Autism Prediction App

    This App predicts ***Autism*** disorder

    Data obtained from the [Kaggle Competition](https://www.kaggle.com/competitions/autismdiagnosis/data)
    ***
    """
)

#diplaing image on webpage
img = Image.open('autism.jpg')
with st.container():
    st.image(img)

#Creating Heading of Sidebar
st.sidebar.header('User Input Features')
#'gender', 'ethnicity', 'jaundice', 'austim', 'contry_of_res',\n 'used_app_before', 'relation'



def user_input_features():
    '''
    Function to take input from users and convert all inputs into Dataframe
    '''
    try:
        def create_radio(col_name):
            '''Create Radio buttons'''
            return st.sidebar.radio(col_name, (0,1), horizontal=True)

        a1_score = create_radio('A1_score')
        a2_score = create_radio('A2_Score')
        a3_score = create_radio('A3_Score')
        a4_score = create_radio('A4_Score')
        a5_score = create_radio('A5_Score')
        a6_score = create_radio('A6_Score')
        a7_score = create_radio('A7_Score')
        a8_score = create_radio('A8_Score')
        a9_score = create_radio('A9_Score')
        a10_score = create_radio('A10_Score')

        gender = st.sidebar.selectbox('gender', ('m','f')),
        jaundice = st.sidebar.selectbox('jaundice', ('no','yes'))
        austim = st.sidebar.selectbox('autism(Family)', ('no','yes'))
        used_app_before = st.sidebar.selectbox('used_app_before', ('no','yes'))
        relation = st.sidebar.selectbox('relation', ('Self','Relative','Parent','Others', 'Health care professional'))
        
        ethnicity = st.sidebar.selectbox('ethnicity', ('White-European','Middle Eastern ','Pasifika','Black','Others','Hispanic', 'Asian','Turkish',
            'South Asian','Latino'))

        contry_of_res = st.sidebar.selectbox('contry_of_res', ('Austria', 'India', 'United States', 'South Africa', 'Jordan',
           'United Kingdom', 'Brazil', 'New Zealand', 'Canada', 'Kazakhstan',
           'United Arab Emirates', 'Australia', 'Others', 'France',
           'Malaysia', 'Netherlands', 'Afghanistan', 'Italy', 'Bahamas',
           'Ireland', 'Sri Lanka', 'Russia', 'Spain', 'Iran'))

        age = st.sidebar.slider('Age',1,100,30)
        result = st.sidebar.slider('Result', 0.0,15.0,10.0)

        data = {
                'A1_Score': a1_score, 'A2_Score': a2_score, 'A3_Score': a3_score, 'A4_Score': a4_score, 'A5_Score': a5_score,
                'A6_Score': a6_score, 'A7_Score': a7_score, 'A8_Score': a8_score, 'A9_Score': a9_score, 'A10_Score': a10_score,
                'gender': gender, 'jaundice': jaundice,'austim':austim, 'used_app_before': used_app_before,
                'relation': relation, 'ethnicity': ethnicity, 'contry_of_res': contry_of_res,
                'age': age, 'result': result

        }

        features = pd.DataFrame(data, index=[0])
        return features
    except Exception as err:
        print('Issues while taking user input in autism-app.py file')
        print(err)

st.write('**Entered User Input Features:**')
input_df = user_input_features()
st.write(input_df)


def preprocess_input(df):
    '''
    Loads Pipeline created in Preprocessing steps to transform user input in desired format; so it can be feeded to ML model
    '''
    try:

        pipeline = 'autism_pipeline.pkl'
        loaded_pipeline = pickle.load(open(pipeline,'rb')) #Loading saved preprocessing Pipeline

        df_trans = loaded_pipeline.transform(df)
        df_trans = pd.DataFrame(df_trans.toarray())  #Converting results to dataframe
        
        #Convert features name from numeric format to string format to avoid warning from ML model
        col_str_name = [str(col) for col in df_trans.columns]
        df_trans.columns = col_str_name
        
        return df_trans

    except Exception as err:
        print('Issue while transforming user input using Pipeline in preprocess_input function of autism-app.py file')
        print(err)


new_input_df = preprocess_input(input_df) #calling function on dataframe contianing user input to apply transformation and saving it to new dataframe 

def make_prediction(df):
    '''
    Load ML model amd make perdiction on user input
    '''
    try:
        model = 'autism_model.pkl'
        loaded_model = pickle.load(open(model,'rb')) #Loading saved model
        
        return loaded_model.predict(df), loaded_model.predict_proba(df)

    except Exception as err:
        print('Issue while making Prediction of user input in make_prediction function of autism-app.py file')
        print(err)
        

pred, pred_proba= make_prediction(new_input_df)

if pred == 0:
    st.write(""" Predction for Autism Disorder: <span style="background:yellow;color:black;"> No </span> """, unsafe_allow_html=True)
else:
    st.write(""" Predction for Autism Disorder: <code style="background:yellow;color:black"> Yes </code> """, unsafe_allow_html=True)


st.write('**Prediction Probability**')
st.write(pred_proba)

