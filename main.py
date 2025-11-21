import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Step 1: 
# Load IMDB dataset word index

st.title("IMDB Movie Review Sentiment Analysis")

st.info("Loading word index and model... This may take a few seconds.")

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Step 2:
#  Load the pre-trained model safely

try:
    model = load_model('simplernn_imdb_model.h5', compile=False)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Step 3:
#  Functions for preprocessing and decoding
def decode_review(encoded_review):
    words = [reverse_word_index.get(i - 3, '?') for i in encoded_review]
    return ' '.join(words)

def preprocess_text(text):
    words = text.lower().split()
    encoded = [word_index.get(word, 2) + 3 for word in words]  # 2 = unknown word, shift by 3
    padded_review = sequence.pad_sequences([encoded], maxlen=500)
    return padded_review

# Step 4: Prediction function
def predict_sentiment(text):
    preprocessed_input = preprocess_text(text)
    try:
        prediction = model.predict(preprocessed_input)
        sentiment = 'Positive' if prediction[0][0] >= 0.5 else 'Negative'
        confidence = prediction[0][0]
    except Exception as e:
        sentiment = "Error"
        confidence = 0
        st.error(f"Prediction failed: {e}")
    return sentiment, confidence

# Step 5: Streamlit user input
user_input = st.text_area("Enter your movie review here:")

if st.button("Predict Sentiment"):
    if user_input.strip() != "":
        sentiment, confidence = predict_sentiment(user_input)
        st.write(f"Predicted Sentiment: {sentiment})")
        ## change line for prediction score
        st.write(f"Prediction Score: {confidence:.4f}")
    else:
        st.warning("Please enter a movie review to analyze.")
