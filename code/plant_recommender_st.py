# This version of the program circumvents sqlite3 to avoid the issues I've been having while trying to publish this app on Streamlit Cloud, and saves time by not having to query the database with each user input.

# Imports
import streamlit as st
import numpy as np
import pandas as pd
#import sqlite3

# Clustering and other modeling Imports
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import SpectralClustering

# set random seed
np.random.seed(42)

st.title('Plant Recommender App')
st.write("""
Created by Joseph Hicks \n
08/18/2021\n
Last Updated 02/28/2022
""")
st.markdown('[GitHub](https://github.com/josephh867)')

# Import data

# Read in the data into a dataframe from raw github csv
df = pd.read_csv('https://raw.githubusercontent.com/josephh867/plant-recommender/main/datasets/cleaned-data.csv')


# Sorting and storing desired features
features = ['id', 'Scientific_Name_x', 'Category', 'Family', 'Growth_Habit', 'Native_Status',
           'Active_Growth_Period', 'Fall_Conspicuous', 'Fire_Resistance', 'Flower_Color',
           'Flower_Conspicuous', 'Fruit_Conspicuous', 'Growth_Rate', 'Lifespan', 'Toxicity',
           'Drought_Tolerance', 'Hedge_Tolerance', 'Moisture_Use', 'pH_Minimum', 'pH_Maximum',
           'Salinity_Tolerance', 'Shade_Tolerance', 'Temperature_Minimum_F', 'Bloom_Period'
           ]

categorical_features = ['Category', 'Family', 'Growth_Habit', 'Native_Status',
                        'Active_Growth_Period', 'Fall_Conspicuous', 'Flower_Color',
                        'Flower_Conspicuous', 'Fruit_Conspicuous', 'Bloom_Period', 'Fire_Resistance'
                        ]

ordinal_features = ['Toxicity', 'Drought_Tolerance', 'Hedge_Tolerance',
                   'Moisture_Use', 'Salinity_Tolerance', 'Shade_Tolerance', 'Growth_Rate', 'Lifespan'
                   ]

other_features = ['id', 'Scientific_Name_x', 'pH_Minimum', 'pH_Maximum',
                 'Temperature_Minimum_F'
                 ]


# Instaniating the SpectralClustering model
spec = SpectralClustering(gamma=0.5, n_clusters=5, n_init=5, n_jobs=-1)

# Recommender Submission Form
st.write("""Fill out the form below and hit SUBMIT to make a query. Please note that you do not have to fill out every field.
""")
with st.form('recommender'):
    # Create an empty dictionary to gather values for the dummy entry
    dummy = {}
    dummy['id'] = 42
    dummy['Scientific_Name_x'] = 'sample'

    # Inputs for simple user-chosen features
    dummy['Lifespan'] = st.selectbox('Choose a desired plant lifespan',
                                     ('Short', 'Moderate', 'Long'), key='a')
    dummy['Drought_Tolerance'] = st.selectbox('Choose your desired drought tolerance',
                                     ('Low', 'Medium', 'High'), key='b')
    dummy['Moisture'] = st.selectbox('Enter the expected moisture use for your plants',
                                     ('Low', 'Medium', 'High'), key='c')
    dummy['Hedge_Tolerance'] = st.selectbox('How tolerant of pruning will your plants need to be?',
                                     ('Low', 'Medium', 'High'), key='d')
    dummy['Shade_Tolerance'] = st.selectbox('How shade tolerant will your plants need to be?',
                                     ('Intolerant', 'Intermediate', 'Tolerant'), key='e')
    dummy['Salinity_Tolerance'] = st.selectbox('How salt tolerant will your plants need to be?',
                                     ('None', 'Low', 'Medium', 'High'), key='f')
    dummy['Growth_Rate'] = st.selectbox('Select Desired Growth Rate',
                                        ('Slow', 'Moderate', 'Rapid'), key='g')
    dummy['Temperature_Minimum_F'] = st.number_input('Enter the coldest expected temperature for your area in F', key='h')
    dummy['pH_Minimum'] = st.number_input('Enter the minimum soil pH', key='i')
    dummy['pH_Maximum'] = st.number_input('Enter the maximum soil pH', key='j')
    dummy['Flower_Conspicuous'] = st.radio('Would you prefer a plant with showy flowers?',
                                           ('Yes', 'No'), key='k')
    dummy['Flower_Color'] = st.selectbox('What flower color would you prefer?',
                                         ('Yellow', 'Red', 'Purple', 'Brown', 'Blue', 'Green', 'White', 'Orange'), key='l')
    dummy['Fall_Conspicuous'] = st.radio('Would you prefer a plant with bright autumn colors?',
                                           ('Yes', 'No'), key='m')
    dummy['Fire_Resistance'] = st.radio('Would you prefer a plant with fire resistance?',
                                           ('Yes', 'No'), key='n')
    dummy['Fruit_Conspicuous'] = st.radio('Would you prefer a plant with conspicuous and/or edible fruit?',
                                           ('Yes', 'No'), key='o')
    neighbors = st.slider('Select the number of results you would like to recieve',
                          value=5,
                          min_value=1,
                          max_value=50,
                          key = 'slider'
                          )

    # Submit the form and start the modeling process
    submitted = st.form_submit_button(label='Submit')
    if submitted:
        # Fill in the other columns with dummy values if they are not specified
        for col in features:
            if col not in dummy.keys():
                dummy[col] = ''

        # Append the dummy entry to the overall dataset
        df_d = pd.DataFrame(dummy, index=[0])
        data = df.append(df_d)

        # Drop the label columns for now, recombine with the data later
        labels = data[['id', 'Scientific_Name_x']]
        data.drop(columns=['id', 'Scientific_Name_x'], inplace=True)

        # Perform data cleaning
        # Map the ordinal features
        ord_dict = {
           # Toxicity
           'None':0, 'Slight':1, 'Moderate':2, 'Severe':3,

           # Tolerances other than shade tolerance
           'None':0, 'Low':1, 'Medium':2, 'High':3,

           # Shade Tolerance
           'Intolerant':0, 'Intermediate':1, 'Tolerant':2,

           # Unique keys for Growth Rate and Lifespan
           'Slow':1, 'Short':1, 'Long':3, 'Rapid':3
        }

        def ordinal_mapper(df, ord_features):

            # Fill empty strings with -1
            df[ord_features] = df[ord_features].replace('', -1)

            # Get values from the dictionary using the get method
            df = df.applymap(lambda x: ord_dict.get(x,x))
            return df

        data = ordinal_mapper(data, ordinal_features)

        data.replace('', np.nan, inplace=True)

        # Dummify the nominal features
        data = pd.get_dummies(data, columns=categorical_features, drop_first=True, dummy_na=True)

        # Now scale the data and perform clustering
        data.fillna(0, inplace=True)
        sc = StandardScaler()
        data_sc = sc.fit_transform(data)
        spec.fit_predict(data_sc)

        # Store cluster predictions and dummy entry cluster
        data['cluster'] = spec.labels_
        out_cluster = spec.labels_[-1]

        # Recombine the data with the label features
        data = pd.concat([data, labels], axis=1)

        # Filter down to the dummy entry and its nearest neighbors
        output = data.loc[data['cluster'] == out_cluster]

        # Sample from the filtered dataset
        results = output[['id', 'Scientific_Name_x']].sample(neighbors)

        # Display results
        results
