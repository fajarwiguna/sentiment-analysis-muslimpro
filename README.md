# 🧠 Analisis Sentimen Ulasan MuslimPro

Aplikasi web interaktif untuk menganalisis sentimen dari ulasan pengguna aplikasi **MuslimPro** menggunakan model machine learning berbasis **LSTM**.

---

## 📚 Deskripsi Singkat

Proyek ini terdiri dari dua bagian utama:

1. **Machine Learning Pipeline (Jupyter Notebook)** – untuk scraping, preprocessing, dan pelatihan model.
2. **Streamlit Web App** – untuk menguji model dalam antarmuka pengguna interaktif.

---

## 🗂️ Struktur Direktori

```
sentiment-analysis-muslimpro/
├── data/                      ← 📂 Dataset hasil scraping
│   └── muslimpro_reviews.csv
├── notebooks/
│   ├── 01_scraping_reviews.ipynb       ← 💬 Scraping ulasan MuslimPro
│   ├── 02_cleaning_preprocessing.ipynb ← 🧹 Preprocessing (opsional)
│   ├── 03_training_LSTM.ipynb          ← 🧠 Pelatihan model LSTM
├── streamlit_app/
│   ├── app.py                          ← 🖥️ Aplikasi Streamlit utama
│   ├── model/
│   │   ├── lstm_model.h5               ← 📦 Model LSTM terlatih
│   │   ├── tokenizer.pkl               ← ✂️ Tokenizer untuk teks
│   │   └── label_encoder.pkl           ← 🔤 Label encoder untuk sentimen
├── requirements.txt                    ← 📦 Dependensi Python
└── README.md                           ← 📄 Dokumentasi proyek ini
```

---

## 🚀 Fitur Aplikasi

- 🎯 Prediksi sentimen dari ulasan MuslimPro (Positif, Netral, Negatif)
- ✍️ Input ulasan manual atau acak
- 📜 Riwayat prediksi ditampilkan secara real-time
- 👍👎 Feedback pengguna (dengan opsi alasan)
- 📊 Visualisasi distribusi sentimen (contoh data)
- 📥 Feedback tersimpan ke `feedback.csv`

---

## 📦 Instalasi

1. **Clone repo ini**:
   ```bash
   git clone https://github.com/username/sentiment-analysis-muslimpro.git
   cd sentiment-muslimpro-ml
   ```

2. **Buat environment & install dependencies**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # atau .venv\Scripts\activate di Windows
   pip install -r requirements.txt
   ```

---

## ▶️ Menjalankan Aplikasi

```bash
cd streamlit_app
streamlit run app.py
```

> Pastikan folder `model/` berisi file:
>
> * `lstm_model.h5`
> * `tokenizer.pkl`
> * `label_encoder.pkl`

---

## 📈 Melatih Ulang Model (Opsional)

Buka dan jalankan Jupyter Notebook berikut:

* `notebooks/01_scraping_reviews.ipynb` – scraping ulasan dari Google Play
* `notebooks/02_cleaning_preprocessing.ipynb` – preprocessing teks
* `notebooks/03_training_LSTM.ipynb` – pelatihan model LSTM & ekspor model

---

## 📋 Requirements

Daftar dependensi tersedia di `requirements.txt`, contoh:

```txt
streamlit
tensorflow
nltk
scikit-learn
pandas
numpy
plotly
google-play-scraper
```

---


## 📄 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).