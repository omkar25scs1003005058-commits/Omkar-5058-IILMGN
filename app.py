import streamlit as st
import pickle
import string
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---------- NLTK SETUP (same idea as before) ----------
def setup_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

setup_nltk()

ps = PorterStemmer()

# cache stopwords ONCE (instead of calling every time)
stop_words = set(stopwords.words('english'))


# ---------- TEXT PREPROCESSING (same logic as your original) ----------
def transform_text(text):
    # lower-case
    text = text.lower()

    # simple tokenization (regex)
    text = re.findall(r'\b\w+\b', text)

    y = []
    # keep only alpha-numeric tokens
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]  # copy

    # remove stopwords and punctuation
    y.clear()
    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    text = y[:]  # copy

    # stemming
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# ---------- LOAD MODEL & VECTORIZER ----------
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))


# ---------- STREAMLIT UI ----------
st.title("Email Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    if not input_sms.strip():
        st.warning("Please enter a message first.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)

        # (Optional: debug – see processed text)
        # st.write("Transformed:", transformed_sms)

        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict
        result = model.predict(vector_input)[0]

        # (Optional: debug – see probabilities if supported)
        # if hasattr(model, "predict_proba"):
        #     proba = model.predict_proba(vector_input)[0]
        #     st.write(f"Ham prob: {proba[0]:.2f}, Spam prob: {proba[1]:.2f}")

        # 4. Show result
        if result == 0:
            st.error("This message looks like **SPAM**.")
        else:
            st.success("This message looks **NOT SPAM**.")

