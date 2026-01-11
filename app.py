import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("next_word_lstm_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Function to predict next word
def predict_next_word(model, tokenizer, seed_text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted_probs, axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return ""

# Function to generate multiple words
def generate_text(model, tokenizer, seed_text, max_sequence_len, next_words=10):
    output_text = seed_text
    for _ in range(next_words):
        next_word = predict_next_word(model, tokenizer, output_text, max_sequence_len)
        if next_word == "":
            break
        output_text += " " + next_word
    return output_text

# Streamlit UI
st.title("Next Word Prediction with LSTM")
st.write("Type a sentence and let the model predict the next word or generate text.")

seed_text = st.text_input("Enter seed text:", "I love deep learning")

next_words = st.slider("Number of words to generate:", min_value=1, max_value=50, value=5)

if st.button("Generate"):
    max_sequence_len = model.input_shape[1] + 1
    generated_text = generate_text(model, tokenizer, seed_text, max_sequence_len, next_words)
    st.write("### Generated Text:")
    st.success(generated_text)