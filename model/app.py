import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ==========================================
# 1. Konfigurasi Halaman & Judul
# ==========================================
st.set_page_config(page_title="Iris Predictor App", layout="centered")
st.title("🌸 Aplikasi Prediksi Bunga Iris (Versi Pickle)")
st.write("Aplikasi ini menggunakan file model .pkl yang disimpan dengan library Pickle.")

# ==========================================
# 2. Load Kedua Model Menggunakan Pickle
# ==========================================
@st.cache_resource # Menjaga agar file tidak dibaca ulang setiap ada interaksi di web
def load_models_pickle():
    # Membuka file model numpy
    with open("model_numpy.pkl", "rb") as f_num:
        model_num = pickle.load(f_num)
        
    # Membuka file model pandas
    with open("model_pandas.pkl", "rb") as f_pan:
        model_pan = pickle.load(f_pan)
        
    return model_num, model_pan

try:
    model_numpy, model_pandas = load_models_pickle()
except Exception as e:
    st.error(f"Gagal memuat file .pkl. Pastikan file berada di folder yang sama. Error: {e}")

# Mapping hasil prediksi angka ke nama spesies asli bunga Iris
target_names = ['Setosa', 'Versicolor', 'Virginica']

# ==========================================
# 3. Sidebar: Pilihan Tipe Data Model
# ==========================================
st.sidebar.header("Pengaturan Model")
pilihan_model = st.sidebar.radio(
    "Pilih Model yang Ingin Digunakan:",
    ("Model NumPy Array", "Model Pandas DataFrame")
)

# ==========================================
# 4. Input Nilai Fitur dari Pengguna (UI)
# ==========================================
st.subheader("Masukkan Karakteristik Bunga:")

# Membuat layout 2 kolom untuk input angka agar lebih rapi
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)

with col2:
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

# ==========================================
# 5. Proses Prediksi Saat Tombol Ditekan
# ==========================================
if st.button("Prediksi Spesies Bunga"):
    
    if pilihan_model == "Model NumPy Array":
        # Jalur NumPy: Mengubah input menjadi matriks 2D array [[sl, sw, pl, pw]]
        data_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediksi = model_numpy.predict(data_input)
        
    else:
        # Jalur Pandas: Mengubah input menjadi DataFrame dengan nama kolom yang sesuai saat training
        # Sesuaikan string nama kolom di bawah dengan nama fitur asli dataset kamu jika berbeda
        nama_kolom = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        data_input = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=nama_kolom)
        prediksi = model_pandas.predict(data_input)
        
    # Menampilkan hasil akhir ke layar aplikasi
    hasil_spesies = target_names[int(prediksi[0])]
    st.success(f"🎉 Hasil Prediksi Menggunakan **{pilihan_model}**: Bunga termasuk spesies **{hasil_spesies}**")

    # Bagian edukasi: Melihat tipe data yang dikirim ke masing-masing model
    with st.expander("Lihat Detail Struktur Data Input"):
        st.write(f"Tipe data yang dikirim: `{type(data_input)}`")
        if pilihan_model == "Model NumPy Array":
            st.write(data_input)
        else:
            st.dataframe(data_input)