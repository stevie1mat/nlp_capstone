import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

# Path to the folder containing text files
folder_path = 'business'

# Load all text files from the folder
documents = []
for filename in sorted(os.listdir(folder_path)):
    if filename.endswith('.txt'):
        with open(os.path.join(folder_path, filename), 'r') as file:
            documents.append(file.read())

# Preprocess the text
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = text.lower().split()
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

preprocessed_docs = [preprocess(doc) for doc in documents]

# Vectorize using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(preprocessed_docs)

# Apply LDA for topic modeling
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(tfidf)

# Function to display topics
def display_topics(model, feature_names, no_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        topics[f"Topic {topic_idx+1}"] = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
    return topics

# Get topic names
topics = display_topics(lda, tfidf_vectorizer.get_feature_names_out(), 10)

# Streamlit app
st.title("BBC News Topic Modeling - Business Section")

st.write("### Extracted Topics:")
for topic, words in topics.items():
    st.write(f"**{topic}:** {', '.join(words)}")
