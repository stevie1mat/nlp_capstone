import streamlit as st
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

# Function to preprocess the text
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = text.lower().split()
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Function to load and preprocess documents from a selected category
def load_data(category):
    folder_path = category
    documents = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r') as file:
                documents.append(preprocess(file.read()))
    return ' '.join(documents)

# Streamlit app
st.title("BBC News Category Word Cloud")

# Input field for category selection
category = st.selectbox("Select a category:", ["business", "politics", "sports", "tech"])

# Load and preprocess data based on the selected category
if category:
    text_data = load_data(category)

    # Generate and display word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    st.pyplot(plt)
