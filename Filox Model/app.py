# ==============================================================================
# app.py - Aplikasi Streamlit untuk Deteksi Hoaks
# ==============================================================================

import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import numpy as np
import os
import re

# --- Konfigurasi Halaman dan Judul ---
st.set_page_config(
    page_title="🚨 FILOX | Detektor Hoaks",
    page_icon="🚨",
    layout="wide"
)

# --- Path ke Folder Model ---
# Pastikan folder ini berada di direktori yang sama dengan app.py
MODEL_DIR = "./hoax_classifier_final"

# ==============================================================================
# FUNGSI-FUNGSI UTAMA (Dengan Caching)
# ==============================================================================

@st.cache_resource
def load_model_and_tokenizer(model_dir):
    """
    Memuat model dan tokenizer.
    @st.cache_resource memastikan fungsi ini hanya dijalankan sekali,
    sehingga model tidak di-load berulang kali.
    """
    if not os.path.isdir(model_dir):
        st.error(f"Error: Direktori model tidak ditemukan di '{model_dir}'.")
        st.error("Pastikan folder 'hoax_classifier_final' yang sudah di-download dari Colab berada di lokasi yang sama dengan file app.py ini.")
        return None, None
    
    try:
        # Memuat tokenizer dan model dari folder yang telah disimpan
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = TFAutoModelForSequenceClassification.from_pretrained(model_dir)
        print("Model dan tokenizer berhasil dimuat.")
        return model, tokenizer
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None, None

def predict(text, model, tokenizer):
    """
    Melakukan prediksi pada teks input.
    """
    # 1. Preprocessing sederhana pada teks input
    text = str(text).lower().strip()
    text = re.sub(r'http\S+|www\S+', '', text, flags=re.MULTILINE) # Hapus URL

    # 2. Tokenisasi teks
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=128)
    
    # 3. Dapatkan prediksi dari model
    logits = model(inputs).logits
    
    # 4. Ubah logits menjadi probabilitas
    probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]
    
    # 5. Tentukan hasil prediksi
    predicted_class_id = np.argmax(probabilities)
    confidence = probabilities[predicted_class_id]
    
    # Asumsi dari training: Label 0 = Valid, Label 1 = Hoaks
    label = "Hoaks" if predicted_class_id == 1 else "Berita Valid"
    
    return label, confidence

# ==============================================================================
# TAMPILAN ANTARMUKA STREAMLIT (UI)
# ==============================================================================

# Judul Utama Aplikasi
st.title("🚨 FILOX: Filter Pendeteksi Hoaks")
st.markdown("Aplikasi ini memanfaatkan model AI (IndoBERT) untuk membantu menganalisis potensi berita hoaks dalam Bahasa Indonesia.")
st.markdown("---")

# Memuat model saat aplikasi pertama kali dijalankan
# Pesan loading akan muncul di sini
with st.spinner("Mempersiapkan model, mohon tunggu..."):
    model, tokenizer = load_model_and_tokenizer(MODEL_DIR)

# Hanya tampilkan UI utama jika model berhasil dimuat
if model and tokenizer:
    st.success("Model siap digunakan!")
    
    st.header("Analisis Teks Berita Anda")
    
    # Area untuk input teks dari pengguna
    user_input = st.text_area(
        "Masukkan atau salin (paste) teks berita yang ingin diperiksa di bawah ini:",
        height=250,
        placeholder="Contoh: Beredar informasi di grup WhatsApp bahwa pemerintah akan memberikan bantuan tunai sebesar 5 juta rupiah..."
    )

    # Tombol untuk memicu analisis
    if st.button("🔍 Analisis Sekarang", type="primary"):
        if user_input.strip():
            # Tampilkan spinner selama proses prediksi
            with st.spinner("🤖 Menganalisis..."):
                prediction, confidence = predict(user_input, model, tokenizer)
            
            st.subheader("Hasil Analisis:")
            
            # Tampilkan hasil dengan format yang berbeda tergantung prediksi
            if prediction == "Hoaks":
                st.error(f"**Kesimpulan: Terdeteksi sebagai {prediction}**")
                st.progress(float(confidence), text=f"Tingkat Keyakinan: {confidence:.2%}")
                st.warning("Harap berhati-hati dan selalu verifikasi informasi dari sumber yang kredibel sebelum menyebarkannya.", icon="⚠️")
            else: # Jika Berita Valid
                st.success(f"**Kesimpulan: Terdeteksi sebagai {prediction}**")
                st.progress(float(confidence), text=f"Tingkat Keyakinan: {confidence:.2%}")
                st.info("Meskipun terdeteksi sebagai berita valid, tetaplah bijak dan kritis dalam menerima informasi.", icon="💡")
        else:
            # Peringatan jika pengguna tidak memasukkan teks
            st.warning("Mohon masukkan teks terlebih dahulu untuk dianalisis.")

# Footer
st.markdown("---")
st.markdown("Dibuat untuk Proyek Capstone | Model: IndoBERT | Framework: Streamlit & TensorFlow")
