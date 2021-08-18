# Imports
import streamlit as st
import numpy as np
import pandas as pd

st.title('Plant Recommender App')
st.write("""
Created by Joseph Hicks
08/18/2021
""")
st.markdown('[GitHub](https://github.com/josephh867)')



# Ask user if they would like to run the cleaning steps to initialize
st.write("""To run this app locally, you will need to perform data cleaning first so the model will run correctly.
To do this, hit the button below:
""")
clean = st.button('Perform Data Cleaning')

if clean:
    # Grab user input on what the file will be called
    output_name = st.text_input('Enter desired filename for the cleaned data.')

    # Read in the data into a dataframe
    con = sqlite3.connect('../datasets/usdadb_new.sqlite3')
    df = pd.read_sql_query('''
                           SELECT *
                           FROM usda
                           WHERE Temperature_Minimum_F IS NOT ''
                           ''', con
                           )

    # Close the connection
    con.close()

    # Sorting and storing desired features
    features = ['id', 'Scientific_Name_x', 'Category', 'Family', 'Growth_Habit', 'Native_Status',
               'Active_Growth_Period', 'Fall_Conspicuous', 'Fire_Resistance', 'Flower_Color',
               'Flower_Conspicuous', 'Fruit_Conspicuous', 'Growth_Rate', 'Lifespan', 'Toxicity',
               'Drought_Tolerance', 'Hedge_Tolerance', 'Moisture_Use', 'pH_Minimum', 'pH_Maximum',
               'Salinity_Tolerance', 'Shade_Tolerance', 'Temperature_Minimum_F', 'Bloom_Period'
               ]

    df = df[features]

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

        # Fill empty strings with 0
        df[ord_features] = df[ord_features].replace('', -1)

        # Get values from the dictionary using the get method
        df = df.applymap(lambda x: ord_dict.get(x,x))
        return df

    df = ordinal_mapper(df, ordinal_features)

    # Dummify the nominal features
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True, dummy_na=True)

    # Save the dataframe to csv with specified filename
    df.to_csv('../datasets/'+output_name+'.csv',index=False)




# User inputs via a form
with st.form('plant_form'):
    st.write("Inside the form")
    slider_val = st.slider("Form slider")
    checkbox_val = st.checkbox("Form checkbox")

    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("slider", slider_val, "checkbox", checkbox_val)
