import streamlit as st
from model import IndoBERTInference

# --- Konfigurasi Halaman (Harus menjadi perintah Streamlit pertama) ---
# Set page configuration
st.set_page_config(
    page_title="Filox: Filter Berita Hoaks",
    page_icon="🛡️",
    layout="centered"
)

# Gunakan st.cache_resource untuk memuat model hanya sekali
@st.cache_resource
def load_model():
    """Fungsi untuk memuat model dan menyimpannya di cache."""
    return IndoBERTInference()

# Muat kelas inferensi (model akan dimuat di sini saat pertama kali dijalankan)
inferencer = load_model()

# --- Antarmuka Pengguna Streamlit ---

# Judul dan deskripsi
st.title("🛡️ Filox: Filter Berita Hoaks")
st.markdown("Selamat datang di Filox! Aplikasi ini membantu Anda mendeteksi berita hoaks menggunakan model IndoBERT. Cukup masukkan teks berita di bawah ini dan klik tombol **Analisis**.")

# Contoh teks
st.subheader("Contoh Teks untuk Dicoba:")
example_texts = {
    "Berita Bohong(Hoax)": "Vaksin COVID-19 berbahaya dan dapat mengubah DNA manusia secara permanen.",
    "Berita Asli (Fakta)": "juru bicara klaim prabowo percaya maju pilpres restu jokowi juru bicara menteri pertahanan prabowo subianto dahnil anzar simanjuntak prabowo percaya maju calon presiden capres pilpres restu presiden joko widodo jokowi dahnil restu suara dukungan prabowo bertambah restu restu jokowi prabowo semangat pemilih prabowo bertambah dahnil acara political show cnn indonesia tv senin malam dahnil prabowo salah tokoh memiliki adab capres prabowo izin jokowi melenggang kontestasi politik prabowo agenda politik lakukan mengganggu kinerja tugastugasnya menteri pertahanan kepemimpinan jokowi prabowo beliau jokowi silahkan mengizinkan prabowo proses kontestasi dahnil gerindra mendukung sepenuhnya pencapresan prabowo suara grass root gerindra sambungnya jokowi memperkenalkan tokoh berpotensi capres cawapres pilpres peringatan ulang partai persatuan pembangunan ppp jumat salah tokoh jokowi prabowo jokowi menyinggung kans prabowo calon presiden pilpres mengungkit reputasi kemenangan prabowo pilpres kali pilpres menang mohon maaf prabowo jatahnya prabowo jokowi puncak peringatan ulang perindo jakarta senin lnadal",
}

# Gunakan kolom untuk tata letak contoh yang lebih bersih
cols = st.columns(len(example_texts))
for i, (key, value) in enumerate(example_texts.items()):
    if cols[i].button(key, key=f"example_{i}"):
        st.session_state.text_input = value


# Area teks untuk input pengguna
news_text = st.text_area(
    "Masukkan teks berita di sini:",
    height=200,
    key="text_input",
    placeholder="Ketik atau tempel artikel berita yang ingin Anda periksa..."
)

# Tombol Analisis
if st.button("Analisis Teks", use_container_width=True, type="primary"):
    if news_text and inferencer.model:
        with st.spinner("Menganalisis teks... Harap tunggu sebentar."):
            result = inferencer.predict(news_text)
            prediction = result['prediction']
            confidence = result['confidence']

            # Tampilkan hasil
            st.subheader("Hasil Analisis")

            if prediction == 'HOAX':
                st.error(f"**Prediksi:** {prediction}")
                confidence_score = confidence * 100
            else:
                st.success(f"**Prediksi:** {prediction}")
                confidence_score = (1 - confidence) * 100

            st.write(f"**Tingkat Keyakinan:** {confidence_score:.2f}%")

            # Bilah progres untuk keyakinan
            if prediction == 'HOAX':
                st.progress(confidence)
            else:
                st.progress(1 - confidence)
                
            with st.expander("Lihat Detail Prediksi"):
                st.write(result)

    elif not news_text:
        st.warning("Mohon masukkan teks berita terlebih dahulu.")
    else:
        st.error("Model tidak dapat dimuat. Silakan periksa log server.")

# Footer
st.markdown("---")
st.markdown("Dibuat untuk Proyek Capstone oleh Tim Filox.")