import pandas as pd
import numpy as np
import os
import joblib
import streamlit as st

# Judul Aplikasi
st.title('Prediksi Kemacetan Lalu Lintas Zurich')
st.write('Aplikasi ini memprediksi status lalu lintas (macet/lancar) berdasarkan nilai arus lalu lintas (flow).')

# ===== LOAD MODEL SAJA (TANPA CSV) =====
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model.pkl")
    return joblib.load(model_path)


model, lower_bound, upper_bound = load_model()

# --- Penjelasan Model ---
st.header('Model yang Digunakan: Random Forest')
with st.expander("Lihat Detail Model dan Perbandingannya"):
    st.write("""
    Model **Random Forest** dipilih untuk melakukan prediksi ini. Alasan utamanya adalah karena model ini menunjukkan performa yang seimbang dan lebih unggul dalam F1-Score untuk mendeteksi kondisi 'lancar' dibandingkan model K-Nearest Neighbors (KNN) selama tahap analisis. 
    
    Random Forest juga lebih tahan terhadap outlier dan tidak memerlukan penskalaan data, menjadikannya pilihan yang andal untuk kasus penggunaan ini.
    """)

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
Model ini menerima nilai `flow` antara **0** hingga sekitar **{upper_bound:.0f}**. 
Nilai di luar rentang ini akan secara otomatis disesuaikan agar sesuai dengan data training.
""")

input_flow = st.number_input(
    'Arus Lalu Lintas (flow)',
    min_value=0.0,
    step=10.0,
    format="%.1f"
)

# Tombol Prediksi
if st.button('Prediksi Status Lalu Lintas'):
    input_flow_capped = np.clip(input_flow, lower_bound, upper_bound)
    input_df = pd.DataFrame([[input_flow_capped]], columns=['flow_capped'])
    prediction = model.predict(input_df)[0]

    st.header('Hasil Prediksi')
    if prediction == 'macet':
        st.markdown(
            """<div style="padding: 15px; border-radius: 10px; background-color: #FFD2D2;">
            <h2 style="color: #D8000C;">🔴 Status Lalu Lintas: MACET</h2>
            <p><strong>Rekomendasi:</strong> Lalu lintas sangat padat. Disarankan untuk mencari rute alternatif atau menunda perjalanan Anda jika memungkinkan.</p>
            </div>""",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """<div style="padding: 15px; border-radius: 10px; background-color: #D9EAD3;">
            <h2 style="color: #4F8A10;">🟢 Status Lalu Lintas: LANCAR</h2>
            <p><strong>Rekomendasi:</strong> Kondisi lalu lintas saat ini lancar. Selamat menikmati perjalanan Anda!</p>
            </div>""",
            unsafe_allow_html=True
        )

