import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Judul Aplikasi
st.title('Prediksi Kemacetan Lalu Lintas Zurich')
st.write('Aplikasi ini memprediksi status lalu lintas (macet/lancar) berdasarkan nilai arus lalu lintas (flow).')

# Fungsi untuk memuat dan memproses data
@st.cache_resource
def train_model():
    # 1. Memuat Data
    df = pd.read_csv("C:/Users/PC/Downloads/zurich.csv")

    # 2. Pembersihan Data Awal
    df = df.drop(columns=['speed'], errors='ignore')
    df = df.drop_duplicates()
    df = df[(df['flow'] >= 0) & (df['occ'] >= 0)]

    # 3. Penanganan Outlier (Capping) untuk 'flow'
    # Ini penting agar input dari user juga bisa di-cap dengan batas yang sama
    q1_flow = df['flow'].quantile(0.25)
    q3_flow = df['flow'].quantile(0.75)
    iqr_flow = q3_flow - q1_flow
    lower_bound_flow = q1_flow - 1.5 * iqr_flow
    upper_bound_flow = q3_flow + 1.5 * iqr_flow
    df['flow_capped'] = df['flow'].clip(lower=lower_bound_flow, upper=upper_bound_flow)

    # 4. Penanganan Outlier (Capping) untuk 'occ' untuk membuat target
    q1_occ = df['occ'].quantile(0.25)
    q3_occ = df['occ'].quantile(0.75)
    iqr_occ = q3_occ - q1_occ
    lower_bound_occ = q1_occ - 1.5 * iqr_occ
    upper_bound_occ = q3_occ + 1.5 * iqr_occ
    df['occ_capped'] = df['occ'].clip(lower=lower_bound_occ, upper=upper_bound_occ)

    # 5. Membuat Variabel Target
    df['traffic_status'] = np.where(df['occ_capped'] > 0.1, 'macet', 'lancar')

    # 6. Menyiapkan Data untuk Model
    X = df[['flow_capped']]
    y = df['traffic_status']

    # 7. Melatih Model Random Forest
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    return model, lower_bound_flow, upper_bound_flow

# Melatih model (hanya berjalan sekali saat aplikasi pertama kali dimuat)
model, lower_bound, upper_bound = train_model()

# --- Penjelasan Model ---
st.header('Model yang Digunakan: Random Forest')
with st.expander("Lihat Detail Model dan Perbandingannya"):
    st.write("""
    Model **Random Forest** dipilih untuk melakukan prediksi ini. Alasan utamanya adalah karena model ini menunjukkan performa yang seimbang dan lebih unggul dalam F1-Score untuk mendeteksi kondisi 'lancar' dibandingkan model K-Nearest Neighbors (KNN) selama tahap analisis. 
    
    Random Forest juga lebih tahan terhadap outlier dan tidak memerlukan penskalaan data, menjadikannya pilihan yang andal untuk kasus penggunaan ini.
    """)

    # Data untuk tabel perbandingan
    comparison_data = {
        'Metric': ['Accuracy', 'F1-Score (Macet)', 'F1-Score (Lancar)'],
        'RandomForest': [0.776297, 0.467544, 0.858404],
        'K-Nearest Neighbors': [0.759889, 0.470061, 0.844780]
    }
    comparison_df = pd.DataFrame(comparison_data).set_index('Metric')
    
    st.subheader('Perbandingan Performa Model')
    st.table(comparison_df.style.format("{:.4f}"))

# --- Input dari User ---
st.header('Masukkan Data untuk Prediksi')
st.info(f"""
**Apa itu Flow?**
`Flow` (Arus Lalu Lintas) adalah jumlah kendaraan yang melewati suatu titik dalam satu interval waktu.

**Rentang Input:**
Model ini menerima nilai `flow` antara **0** hingga sekitar **{upper_bound:.0f}**. Nilai di luar rentang ini akan secara otomatis disesuaikan agar sesuai dengan data yang digunakan untuk melatih model.
""")
input_flow = st.number_input('Arus Lalu Lintas (flow)', min_value=0.0, step=10.0, format="%.1f")

# Tombol Prediksi
if st.button('Prediksi Status Lalu Lintas'):
    # Pastikan input dari user juga di-cap dengan batas yang sama seperti data training
    input_flow_capped = np.clip(input_flow, lower_bound, upper_bound)
    
    # Buat dataframe dari input untuk prediksi
    input_df = pd.DataFrame([[input_flow_capped]], columns=['flow_capped'])
    
    # Lakukan prediksi
    prediction = model.predict(input_df)[0]
    
    # Tampilkan hasil dengan format baru
    st.header('Hasil Prediksi')
    if prediction == 'macet':
        st.markdown(f"""<div style="padding: 15px; border-radius: 10px; background-color: #FFD2D2;">
        <h2 style="color: #D8000C;">🔴 Status Lalu Lintas: MACET</h2>
        <p><strong>Rekomendasi:</strong> Lalu lintas sangat padat. Disarankan untuk mencari rute alternatif atau menunda perjalanan Anda jika memungkinkan.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div style="padding: 15px; border-radius: 10px; background-color: #D9EAD3;">
        <h2 style="color: #4F8A10;">🟢 Status Lalu Lintas: LANCAR</h2>
        <p><strong>Rekomendasi:</strong> Kondisi lalu lintas saat ini lancar. Selamat menikmati perjalanan Anda!</p>
        </div>""", unsafe_allow_html=True)
