# Streamlit app (streamlit_app.py)
import streamlit as st
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# Load the trained model
model = load_model('emotion_detection_model.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define maxlen (same as used during training)
maxlen = 79

# Define emotion labels and corresponding emojis
emotion_labels = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']
emotion_emojis = {
    'Sadness': '😢',
    'Joy': '😊',
    'Love': '❤️',
    'Anger': '😠',
    'Fear': '😨',
    'Surprise': '😲'
}

# Function to preprocess input text
def preprocess_input(text, tokenizer, maxlen):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=maxlen)
    return padded_sequence

# Streamlit app
st.title("Emotion Detection")

# Text input
user_input = st.text_area("Enter a sentence to detect emotion:")

# Predict button
if st.button("Detect Emotion"):
    if user_input:
        # Preprocess the input
        processed_input = preprocess_input(user_input, tokenizer, maxlen)
        
        # Make prediction
        prediction = model.predict(processed_input)
        predicted_class = np.argmax(prediction, axis=1)[0]
        detected_emotion = emotion_labels[predicted_class]
        
        # Display the prediction
        st.write(f"The detected emotion is: {detected_emotion}")
        st.markdown(f"<h1 style='text-align: center; font-size: 100px;'>{emotion_emojis[detected_emotion]}</h1>", unsafe_allow_html=True)
    else:
        st.write("Please enter a sentence.")
