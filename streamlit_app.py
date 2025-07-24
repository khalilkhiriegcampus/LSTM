import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load trained model
model = tf.keras.models.load_model('gru_model.h5')

# Get total words and max sequence length
total_words = len(tokenizer.word_index) + 1
max_sequence_len = model.input_shape[1] + 1

# Prediction function
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return "?"

# Streamlit UI
st.title("Next Word Prediction App (Hamlet)")

user_input = st.text_input("Enter a phrase:", "Do's not divide the Sunday from the")

if st.button("Predict Next Word"):
    next_word = predict_next_word(model, tokenizer, user_input, max_sequence_len)
    st.write(f"**Next word prediction:** `{next_word}`")