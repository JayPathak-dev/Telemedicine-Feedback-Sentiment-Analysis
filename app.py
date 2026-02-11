import streamlit as st
import numpy as np
import pickle
import re
import pandas as pd
import matplotlib.pyplot as plt
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Hospital Sentiment Intelligence",
    page_icon="🏥",
    layout="wide"
)

# ==========================================
# PREMIUM CSS STYLING
# ==========================================
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #4f46e5, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
    margin-bottom: 30px;
}

.metric-card {
    padding: 20px;
    border-radius: 20px;
    background: linear-gradient(145deg, #111827, #1f2937);
    box-shadow: 0 4px 30px rgba(0,0,0,0.4);
    text-align: center;
}

.section-header {
    font-size: 24px;
    font-weight: 600;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏥 Hospital Sentiment Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Emotion Analytics Dashboard</div>', unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_artifacts():
    model = load_model("sentiment_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_artifacts()

# ==========================================
# NLP SETUP
# ==========================================
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
MAX_SEQUENCE_LENGTH = 120

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(word)
                     for word in words if word not in stop_words]
    return " ".join(cleaned_words)

# ==========================================
# SENTIMENT SCORE (-1 to +1)
# ==========================================
def predict_score(text):
    cleaned = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH,
                           padding='post', truncating='post')
    prediction = model.predict(padded, verbose=0)[0][0]
    scaled_score = (prediction * 2) - 1
    return round(float(scaled_score), 3)

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("⚙️ Mode Selection")
mode = st.sidebar.radio("", ["Single Analysis", "Bulk Dashboard"])

# ==========================================
# SINGLE ANALYSIS
# ==========================================
if mode == "Single Analysis":

    st.markdown('<div class="section-header">📝 Analyze Single Comment</div>', unsafe_allow_html=True)

    user_input = st.text_area("Enter hospital feedback:", height=150)

    if st.button("Analyze Sentiment 🚀"):

        if user_input.strip() == "":
            st.warning("Please enter a comment.")
        else:
            score = predict_score(user_input)

            col1, col2, col3 = st.columns([1,2,1])

            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Sentiment Score (-1 → +1)", score)
                st.markdown('</div>', unsafe_allow_html=True)

            if score > 0.3:
                sentiment = "Positive 😊"
                color = "green"
            elif score < -0.3:
                sentiment = "Negative 😞"
                color = "red"
            else:
                sentiment = "Neutral 😐"
                color = "orange"

            st.markdown(f"### Overall Sentiment: :{color}[{sentiment}]")

            # Custom Score Bar
            fig, ax = plt.subplots(figsize=(8,1.5))
            ax.barh([""], [score])
            ax.set_xlim(-1, 1)
            ax.axvline(0)
            ax.set_yticks([])
            st.pyplot(fig)

# ==========================================
# BULK DASHBOARD
# ==========================================
else:

    st.markdown('<div class="section-header">📊 Bulk Sentiment Dashboard</div>', unsafe_allow_html=True)

    bulk_input = st.text_area(
        "Enter multiple comments (one per line):",
        height=200
    )

    if st.button("Run Bulk Analysis 🚀"):

        comments = [c for c in bulk_input.split("\n") if c.strip() != ""]

        if len(comments) == 0:
            st.warning("Please enter valid comments.")
        else:

            results = []

            for comment in comments:
                score = predict_score(comment)

                if score > 0.3:
                    sentiment = "Positive"
                elif score < -0.3:
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"

                results.append({
                    "Comment": comment,
                    "Score": score,
                    "Sentiment": sentiment
                })

            df_results = pd.DataFrame(results)

            # KPI Metrics
            positive = len(df_results[df_results["Sentiment"] == "Positive"])
            neutral = len(df_results[df_results["Sentiment"] == "Neutral"])
            negative = len(df_results[df_results["Sentiment"] == "Negative"])

            col1, col2, col3 = st.columns(3)

            col1.metric("Positive", positive)
            col2.metric("Neutral", neutral)
            col3.metric("Negative", negative)

            st.markdown("### 📋 Detailed Results")
            st.dataframe(df_results, use_container_width=True)

            # Pie Chart
            fig, ax = plt.subplots()
            ax.pie(
                [positive, neutral, negative],
                labels=["Positive", "Neutral", "Negative"],
                autopct="%1.1f%%"
            )
            st.pyplot(fig)

            # Download Option
            csv = df_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="sentiment_results.csv",
                mime="text/csv"
            )
