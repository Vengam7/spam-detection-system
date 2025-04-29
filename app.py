import streamlit as st
from joblib import load

# Load saved model and vectorizer
model = load('spam_classifier_model.joblib')
vectorizer = load('tfidf_vectorizer.joblib')

# Set up Streamlit UI
st.set_page_config(page_title="Spam Message Detector")
st.title("📩 Spam Message Detector")

# User input
msg = st.text_area("Enter a message:")

# Prediction
if st.button("Check Message"):
    msg_clean = msg.lower()
    msg_vector = vectorizer.transform([msg_clean])
    prediction = model.predict(msg_vector)

    if prediction[0] == 1:
        st.error("🚫 This is SPAM!")
    else:
        st.success("✅ This is NOT spam.")
