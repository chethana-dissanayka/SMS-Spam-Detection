import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english')]
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text]
    return " ".join(text)

# Load model and vectorizer
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)

except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()

st.title("SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    try:
        # Preprocess input message
        transformed_sms = transform_text(input_sms)
        # Vectorize using loaded TF-IDF vectorizer
        vector_input = tfidf.transform([transformed_sms])
        # Predict using loaded model
        result = model.predict(vector_input)[0]

        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")

    except Exception as e:
        st.error(f"Prediction error: {e}")
