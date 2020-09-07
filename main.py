# Description: This is a web app that diagnoses CKD using Python and Machine Learning

# Import the libraries
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import requests
from io import BytesIO
import s3fs
import streamlit as st

# Create a title and sub-title
st.write("""
# Chronic Kidney Disease Detection
Determine if someone has Chronic Kidney Disease (CKD) using Python and ML!
""")

# Open and display thumbnail image
response = requests.get('https://chronic-kidney-disease-assets.s3-us-west-1.amazonaws.com/ckdImage.png')
image = Image.open(BytesIO(response.content))
st.image(image, caption='ML', use_column_width=True)

# Get the data
df = pd.read_csv('s3://chronic-kidney-disease-assets/chronickidneyfinal.csv')

# Set a sub header
st.subheader('Data Information:')

# Show the data as a table
st.dataframe(df)

# Show statistics on the data
st.write(df.describe())

# Show the data as a chart
chart = st.bar_chart(df)

# Create the feature and target data set
X = df.drop(['classification'], 1)
Y = np.array(df['classification'])

# Split the data into 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

# Get the feature input from the user
def get_user_input():
    sg = st.sidebar.slider('sg', 1.0, 2.0, 1.5)
    al = st.sidebar.slider('al', 0.0, 5.0, 2.5)
    sc = st.sidebar.slider('sc', 0.0, 100.0, 50.0)
    hemo = st.sidebar.slider('hemo', 0.0, 25.0, 12.5)
    pcv = st.sidebar.slider('pcv', 0.0, 60.0, 30.0)
    wc = st.sidebar.slider('wc', 2000.0, 30000.0, 14000.0)
    rc = st.sidebar.slider('rc', 0.0, 10.0, 5.0)
    htn = st.sidebar.slider('htn', 0.0, 1.0, 0.0)

    # Store a dictionary into a variable
    user_data = {
        'sg' : sg,
        'al' : al,
        'sc' : sc,
        'hemo' : hemo,
        'pcv' : pcv,
        'wc' : wc,
        'rc' : rc,
        'htn' : htn,
    }

    # Transform the data into a data frame
    features = pd.DataFrame(user_data, index = [0])
    return features

# Store the user input into a variable
user_input = get_user_input()

# Set a sub header and display the users' input
st.subheader('User Input:')
st.write(user_input)

# Create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

# Show the models' metrics
st.subheader('Model Test Accuracy Score:')
st.write( str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100) + '%')

# Store the models' predictions in a variable
prediction = RandomForestClassifier.predict(user_input)

# Set a sub header and display the classification
st.subheader('Classification: ')
st.write(prediction)

diagnosis_certainty = str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100) + '%'

if prediction == 1:
    diagnosis_statement = "There is a {} chance you have Chronic Kidney Disease. God bless you <3".format(diagnosis_certainty)
elif prediction == 0:
    diagnosis_statement = "There is a {} chance you do not have Chronic Kidney Disease. God bless you <3".format(diagnosis_certainty)

st.write(diagnosis_statement)