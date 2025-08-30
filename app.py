import streamlit as st
import joblib
import os

# =========================
# Load Model & Vectorizer
# =========================
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "linear_svc_model.pkl")
    vectorizer_path = os.path.join("models", "tfidf_vectorizer.pkl")

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error(" Model or vectorizer file not found! Please train the model first.")
        return None, None

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

model, vectorizer = load_model()


# Streamlit UI

st.set_page_config(page_title=" Spam Detector", page_icon="📩", layout="centered")

st.title(" Spam Email Detector")
st.write("Paste an email below, and I'll predict whether it's **Ham (Not Spam)** or **Spam**.")

# Text area for email input
email_text = st.text_area(" Enter your email text here:", height=200)

# Prediction button
if st.button(" Predict"):
    if not email_text.strip():
        st.warning(" Please enter some text first.")
    elif model is None or vectorizer is None:
        st.error(" Model not loaded. Please train and save the model before running the app.")
    else:
        # Transform and predict
        input_vector = vectorizer.transform([email_text])
        prediction = model.predict(input_vector)[0]

        # Display result
        if prediction == 0:
            st.success(" This email is **Ham (Not Spam)**")
        else:
            st.error(" This email is **Spam**")

        # Confidence score (if available)
        if hasattr(model, "decision_function"):
            score = model.decision_function(input_vector)[0]
            st.info(f" Confidence Score: `{score:.4f}`")
