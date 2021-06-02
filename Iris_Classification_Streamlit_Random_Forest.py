#!/usr/bin/env python
# coding: utf-8

# ### Iris Flower Dataset Classification with Streamlit and Random Forest

# In[1]:


# https://www.kaggle.com/arshid/iris-flower-dataset
# https://www.youtube.com/watch?v=JwSS70SZdyM&t=780s


# In[3]:


import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Iris Flower Dataset Prediction Application by using Random FOrest
This app predicts the **Iris flower** type!
""")

st.sidebar.header('Customer Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('Customer Input parameters')
st.write(df)

# Loading the IRIS dataset
iris = datasets.load_iris()
X = iris.data
Y = iris.target


# Training by using Random Forst
clf = RandomForestClassifier()
clf.fit(X, Y)


# Predicting
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

# Print
st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

# Print The prediction 
st.subheader('Prediction')
st.write(iris.target_names[prediction])
#st.write(prediction)

# Print The prediction Probability
st.subheader('Prediction Probability')
st.write(prediction_proba)


# In[ ]:




