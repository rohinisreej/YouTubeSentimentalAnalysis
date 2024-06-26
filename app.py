import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import string
from string import punctuation
import nltk
import re

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Streamlit setup
st.title('Automated Comment harvesting and Sentiment Analysis')
st.write('Public opinion.')

# Load the dataset
uploaded_file = st.file_uploader(r"D:\Rohini\VscodeStreamlit\bb1.csv", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head())

    # Sentiment Analysis
    nltk.download('vader_lexicon')
    sentiments = SentimentIntensityAnalyzer()
    data["Replies"] = data["Replies"].apply(lambda x: str(x) if not isinstance(x, str) else x)
    data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["Replies"]]
    data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["Replies"]]
    data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["Replies"]]
    data['Compound'] = [sentiments.polarity_scores(i)["compound"] for i in data["Replies"]]
    score = data["Compound"].values
    sentiment = []
    for i in score:
        if i >= 0.05:
            sentiment.append('Positive')
        elif i <= -0.05:
            sentiment.append('Negative')
        else:
            sentiment.append('Neutral')
    data["Sentiment"] = sentiment

    # Machine Learning Model
    X_train, X_test, y_train, y_test = train_test_split(data['Replies'], data['Sentiment'], test_size=0.2, random_state=42)

    # Use TfidfVectorizer or CountVectorizer for text data
    vectorizer = TfidfVectorizer()  # You can also try CountVectorizer
    model = make_pipeline(vectorizer, MultinomialNB())
    model.fit(X_train, y_train)
    predicted_sentiments = model.predict(X_test)

    accuracy = accuracy_score(y_test, predicted_sentiments)
    st.write(f"Accuracy: {accuracy:.2f}")

    data['Predicted_Sentiment'] = model.predict(data['Replies'])
    st.write(data[['Replies', 'Sentiment', 'Predicted_Sentiment']])

    # Plotting bar chart and pie chart
    st.subheader('Sentiment Analysis Results')

    sentiment_counts = data['Predicted_Sentiment'].value_counts()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    sentiment_counts.plot(kind='bar', color=['pink', 'green', 'yellow'], ax=axes[0])
    axes[0].set_title('Predicted Sentiments Bar Chart')
    axes[0].set_xlabel('Sentiment')
    axes[0].set_ylabel('Count')

    # Pie chart
    sentiment_counts.plot(kind='pie', autopct='%1.1f%%', colors=['pink', 'green', 'yellow'], ax=axes[1])
    axes[1].set_title('Predicted Sentiments Pie Chart')

    st.pyplot(fig)

# To run the streamlit app, save this script as app.py and run `streamlit run app.py` in your terminal
