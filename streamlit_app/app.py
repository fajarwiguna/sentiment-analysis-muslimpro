import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import plotly.express as px
from datetime import datetime

# ===== BASE PATH SETUP =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
NLTK_DIR = os.path.join(BASE_DIR, "nltk_data")

# ===== NLTK SETUP =====
nltk.data.path.append(NLTK_DIR)
try:
    stop_words = set(stopwords.words('indonesian'))
except LookupError:
    nltk.download('stopwords', download_dir=NLTK_DIR)
    stop_words = set(stopwords.words('indonesian'))

# ===== CLEANING FUNCTION =====
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

# ===== LOAD MODEL + TOKENIZER =====
@st.cache_resource
def load_model_and_tokenizer():
    model_path = os.path.join(MODEL_DIR, 'lstm_model.h5')
    tokenizer_path = os.path.join(MODEL_DIR, 'tokenizer.pkl')
    label_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')

    model = tf.keras.models.load_model(model_path)

    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open(label_path, 'rb') as handle:
        le = pickle.load(handle)

    return model, tokenizer, le

# ===== SESSION STATE =====
for key in ['review', 'history', 'feedback', 'reason']:
    if key not in st.session_state:
        st.session_state[key] = "" if key != "history" else []

# ===== SAMPLE REVIEWS =====
sample_reviews = [
    "Aplikasi ini sangat membantu dalam mendalami Islam",
    "Sering crash saat buka doa harian",
    "Fitur adzan sangat akurat dan bermanfaat",
    "Antarmukanya kurang ramah pengguna",
    "Bagus, tapi terlalu banyak iklan"
]

# ===== SIDEBAR =====
st.sidebar.title("⚙️ Informasi")

st.sidebar.markdown("### 📌 Tentang Aplikasi")
st.sidebar.info(
    "Aplikasi ini mendeteksi sentimen dari ulasan pengguna aplikasi MuslimPro "
    "menggunakan model Machine Learning berbasis LSTM. "
    "Output berupa kategori sentimen: **positif**, **netral**, atau **negatif**."
)

st.sidebar.markdown("---")  # Garis horizontal pemisah

st.sidebar.markdown("### 🧠 Model yang Digunakan")
st.sidebar.markdown("""
- Arsitektur: **LSTM (Long Short-Term Memory)**
- Bahasa: **Indonesia**
- Framework: **TensorFlow, Keras**
""")

# ===== CSS =====
st.markdown("""
    <style>
    .stTextArea textarea { border: 2px solid #007bff; border-radius: 8px; }
    .stButton>button { border-radius: 8px; font-weight: bold; }
    .custom-sample-button button {
        background-color: #6c757d;
        color: white;
        border-radius: 8px;
        font-weight: bold;
    }
    .custom-sample-button button:hover {
        background-color: #5a6268;
    }
    .prediction-box { padding: 10px; border-radius: 8px; margin-top: 10px; font-size: 18px; }
    .positive { background-color: #d4edda; color: #155724; }
    .neutral { background-color: #fff3cd; color: #856404; }
    .negative { background-color: #f8d7da; color: #721c24; }
    </style>
""", unsafe_allow_html=True)

# ===== JUDUL & INPUT =====
st.title("🧠 Analisis Sentimen Ulasan MuslimPro")
st.markdown("Masukkan ulasan aplikasi MuslimPro untuk mengetahui apakah ulasannya **positif**, **netral**, atau **negatif**.")

st.markdown("### ✍️ Masukkan atau pilih ulasan")
col1, col2 = st.columns([4, 1])
with col1:
    review = st.text_area("Ulasan Anda:", value=st.session_state.review, height=100, max_chars=500, key="review_area")
    st.session_state.review = review
with col2:
    st.markdown("### &nbsp;")
    st.markdown('<div class="custom-sample-button">', unsafe_allow_html=True)
    if st.button("🎲 Contoh Ulasan", use_container_width=True):
        st.session_state.review = np.random.choice(sample_reviews)
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ===== LOAD MODEL =====
if not os.path.exists(MODEL_DIR):
    st.error("Folder 'model' tidak ditemukan!")
else:
    model, tokenizer, le = load_model_and_tokenizer()

    # ===== PREDIKSI =====
    if st.button("🎯 Jalankan Prediksi"):
        review = st.session_state.review.strip()
        if not review:
            st.error("Ulasan tidak boleh kosong!")
        elif len(review) < 10:
            st.error("Ulasan terlalu pendek! Minimal 10 karakter.")
        else:
            cleaned_review = clean_text(review)
            seq = tokenizer.texts_to_sequences([cleaned_review])
            padded = pad_sequences(seq, padding='post', maxlen=100)

            with st.spinner("Memproses..."):
                prediction = model.predict(padded)
                predicted_class = le.inverse_transform([np.argmax(prediction)])[0]
                confidence = float(np.max(prediction))

            st.session_state.last_result = {
                "review": review,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.feedback = None
            st.session_state.reason = ""
            st.rerun()

# ===== TAMPILKAN HASIL =====
if 'last_result' in st.session_state:
    result = st.session_state.last_result
    emoji_map = {"positif": "😊", "netral": "😐", "negatif": "😠"}
    class_style = {"positif": "positive", "netral": "neutral", "negatif": "negative"}
    st.markdown(
        f"<div class='prediction-box {class_style[result['predicted_class']]}'>"
        f"<b>Sentimen:</b> {emoji_map[result['predicted_class']]} {result['predicted_class'].capitalize()}</div>",
        unsafe_allow_html=True
    )
    st.write(f"**Tingkat Keyakinan:** {result['confidence']:.2%}")
    st.write(f"**Ulasan Anda:** {result['review']}")

    # ===== FEEDBACK =====
    st.markdown("**Apakah prediksi ini benar?**")
    fb_col1, fb_col2 = st.columns(2)
    with fb_col1:
        if st.button("👍 Benar"):
            st.session_state.feedback = "Benar"
    with fb_col2:
        if st.button("👎 Salah"):
            st.session_state.feedback = "Salah"
            st.rerun()

    save_feedback = False
    if st.session_state.feedback == "Salah":
        st.session_state.reason = st.text_input("📝 Alasan Anda (opsional):", key="reason_input")
        if st.button("📤 Kirim Feedback"):
            save_feedback = True
    elif st.session_state.feedback == "Benar":
        save_feedback = True

    if save_feedback:
        feedback_data = {
            "Ulasan": result["review"],
            "Prediksi": result["predicted_class"],
            "Feedback": st.session_state.feedback,
            "Alasan": st.session_state.reason,
            "Waktu": result["timestamp"]
        }
        feedback_df = pd.DataFrame([feedback_data])
        feedback_file = os.path.join(BASE_DIR, "feedback.csv")
        if os.path.exists(feedback_file):
            feedback_df.to_csv(feedback_file, mode='a', header=False, index=False)
        else:
            feedback_df.to_csv(feedback_file, index=False)

        st.success("✅ Feedback disimpan!")
        del st.session_state.last_result
        st.session_state.feedback = None
        st.session_state.reason = ""

