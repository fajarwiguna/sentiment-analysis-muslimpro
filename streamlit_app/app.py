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
from datetime import datetime
import plotly.express as px

# ===== BASE PATH =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
NLTK_DIR = os.path.join(BASE_DIR, "nltk_data")

# ===== NLTK SETUP =====
nltk.data.path.append(NLTK_DIR)
try:
    stop_words = set(stopwords.words("indonesian"))
except LookupError:
    nltk.download("stopwords", download_dir=NLTK_DIR)
    stop_words = set(stopwords.words("indonesian"))

# ===== CLEANING FUNCTION =====
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

# ===== LOAD MODEL & TOKENIZER =====
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "lstm_model.h5"))
    with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    return model, tokenizer, le

# ===== SESSION STATE INIT =====
for key in ["review", "history", "feedback", "reason"]:
    if key not in st.session_state:
        st.session_state[key] = "" if key != "history" else []

# ===== SIDEBAR =====
st.sidebar.title("⚙️ Informasi")
st.sidebar.markdown("### 📌 Tentang Aplikasi")
st.sidebar.info(
    "Aplikasi ini mendeteksi sentimen dari ulasan pengguna MuslimPro "
    "menggunakan model **LSTM**. Keluaran berupa: **positif**, **netral**, atau **negatif**."
)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Model")
st.sidebar.markdown("- Arsitektur: LSTM\n- Bahasa: Indonesia\n- Framework: TensorFlow")

# ===== CSS =====
st.markdown("""
<style>
    .stTextArea textarea { border: 2px solid #007bff; border-radius: 8px; }
    .stButton>button { border-radius: 8px; font-weight: bold; }
    .prediction-box { padding: 10px; border-radius: 8px; margin-top: 10px; font-size: 18px; }
    .positive { background-color: #d4edda; color: #155724; }
    .neutral { background-color: #fff3cd; color: #856404; }
    .negative { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# ===== LOAD MODEL =====
if not os.path.exists(MODEL_DIR):
    st.error("Folder 'model' tidak ditemukan!")
else:
    model, tokenizer, le = load_model_and_tokenizer()

    # ===== TABS =====
    tab1, tab2 = st.tabs(["🔍 Analisis Ulasan", "📊 Visualisasi & Upload"])

    # === TAB 1: Prediksi Satu Ulasan ===
    with tab1:
        st.title("🧠 Analisis Sentimen MuslimPro")
        st.markdown("Masukkan ulasan untuk mengetahui sentimen-nya.")

        col1, col2 = st.columns([4, 1])
        sample_reviews = [
            "Aplikasi ini sangat membantu dalam mendalami Islam",
            "Sering crash saat buka doa harian",
            "Fitur adzan sangat akurat dan bermanfaat",
            "Antarmukanya kurang ramah pengguna",
            "Bagus, tapi terlalu banyak iklan"
        ]

        with col1:
            review = st.text_area("Ulasan Anda:", value=st.session_state.review, height=100, max_chars=500, key="review_area")
            st.session_state.review = review
        with col2:
            if st.button("🎲 Contoh Ulasan"):
                st.session_state.review = np.random.choice(sample_reviews)
                st.rerun()

        if st.button("🎯 Jalankan Prediksi"):
            review = st.session_state.review.strip()
            if not review:
                st.error("Ulasan tidak boleh kosong!")
            elif len(review) < 10:
                st.error("Ulasan terlalu pendek!")
            else:
                cleaned = clean_text(review)
                seq = tokenizer.texts_to_sequences([cleaned])
                padded = pad_sequences(seq, padding="post", maxlen=100)

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

        # ===== HASIL =====
        if "last_result" in st.session_state:
            result = st.session_state.last_result
            emoji_map = {"positif": "😊", "netral": "😐", "negatif": "😠"}
            class_style = {"positif": "positive", "netral": "neutral", "negatif": "negative"}
            st.markdown(
                f"<div class='prediction-box {class_style[result['predicted_class']]}'>"
                f"<b>Sentimen:</b> {emoji_map[result['predicted_class']]} {result['predicted_class'].capitalize()}</div>",
                unsafe_allow_html=True
            )
            st.write(f"**Tingkat Keyakinan:** {result['confidence']:.2%}")
            st.write(f"**Ulasan:** {result['review']}")

            # Feedback
            st.markdown("**Apakah prediksi ini benar?**")
            colfb1, colfb2 = st.columns(2)
            with colfb1:
                if st.button("👍 Benar"):
                    st.session_state.feedback = "Benar"
            with colfb2:
                if st.button("👎 Salah"):
                    st.session_state.feedback = "Salah"
                    st.rerun()

            # Save feedback
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
                feedback_path = os.path.join(BASE_DIR, "feedback.csv")
                if os.path.exists(feedback_path):
                    feedback_df.to_csv(feedback_path, mode="a", header=False, index=False)
                else:
                    feedback_df.to_csv(feedback_path, index=False)
                st.success("✅ Feedback disimpan!")
                del st.session_state.last_result
                st.session_state.feedback = None
                st.session_state.reason = ""

    # === TAB 2: Visualisasi + Upload CSV ===
    with tab2:
        st.subheader("📊 Contoh Distribusi Sentimen")
        distribusi = {"Positif": 60, "Netral": 25, "Negatif": 15}
        fig = px.pie(
            values=list(distribusi.values()),
            names=list(distribusi.keys()),
            title="Distribusi Sentimen (Dummy Data)",
            color_discrete_map={"Positif": "#28a745", "Netral": "#ffc107", "Negatif": "#dc3545"}
        )
        fig.update_layout(width=500, height=400)
        st.plotly_chart(fig)

        st.markdown("### 📁 Coba Upload File CSV")
        uploaded_file = st.file_uploader("Upload file CSV dengan kolom 'review'", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if "review" not in df.columns:
                st.error("CSV harus memiliki kolom 'review'.")
            else:
                df["clean"] = df["review"].astype(str).apply(clean_text)
                seqs = tokenizer.texts_to_sequences(df["clean"])
                padded = pad_sequences(seqs, padding='post', maxlen=100)
                preds = model.predict(padded)
                df["sentimen"] = le.inverse_transform(np.argmax(preds, axis=1))
                st.dataframe(df[["review", "sentimen"]])
