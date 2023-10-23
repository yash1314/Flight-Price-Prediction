import pandas as pd
import numpy as np
import streamlit as st
import datetime 
import src.utils as utils
from src.pipelines.prediction_pipeline import PredictPipeline
import time

predict = PredictPipeline()

st.title('Flight Fare Price Prediction.')
st.markdown("### Predict ticket price for you flight journey.")
st.markdown('Enter all the details and then press predict to get price estimation.')

cols1, cols2 , cols3= st.columns(3)

with cols1:
    st.markdown("### Arival Airport")
    Source = st.selectbox(label= 'From: ', options= ['New Delhi', 'Banglore', 'Mumbai', 'Chennai'])

with cols3:
    st.markdown("### Departure Airport")
    Destination = st.selectbox(label='To', options= ['Cochin', 'Banglore', 'New Delhi', 'Hyderabad'])

st.markdown('')

with cols1:
    st.markdown("### Boarding Date")
    date = date = st.date_input(label='Enter boarding date',value=datetime.datetime.now(), format = "DD/MM/YYYY" )

with cols3:
    st.markdown("### Select Airline")
    airline = st.selectbox(label= 'Airlines: ', options= [ 'Air India',
    'GoAir',
    'IndiGo',
    'Jet Airways',
    'Jet Airways Business',
    'Multiple carriers',
    'Multiple carriers Premium economy',
    'SpiceJet',
    'Trujet',
    'Vistara',
    'Vistara Premium economy'])


with cols1:
    st.markdown("### Select Flight Duration")
    duration = st.selectbox(label= 'Duration of flight: ', options= ['Short', 'Medium', 'Long'])

with cols3:
    st.markdown("### Select Number Stops")
    total_stops = st.selectbox(label = 'Number of Stops', options= ['non-stop', '1 stop', '2 stops', '3 stops', '4 stops'])


st.markdown('')

if st.button('Predict'):
    
    input_data = utils.preprocess(Airline= airline, Date_of_Journey=date, Source= Source,
                                    Destination= Destination, Duration=duration, Total_Stops=total_stops)
    
    result = np.round(predict.predict(features = input_data))

    bar = st.progress(50)
    time.sleep(1)
    bar.progress(100)

    st.header(f"The Predicted Price of Fight ticket is ₹ {result}")