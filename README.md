# Aplikasi Prediksi Kemacetan Lalu Lintas Zurich

Aplikasi web berbasis **Streamlit** yang digunakan untuk memprediksi status lalu lintas (**Macet** atau **Lancar**) berdasarkan nilai arus lalu lintas (*traffic flow*).  
Aplikasi ini merupakan hasil penerapan model *machine learning* yang telah dilatih sebelumnya menggunakan dataset lalu lintas Zurich.

---

## Tujuan Aplikasi

Aplikasi ini bertujuan untuk:
- Menyediakan simulasi prediksi kondisi lalu lintas secara sederhana
- Menunjukkan implementasi *machine learning model* dalam bentuk aplikasi web
- Mengubah hasil analisis data menjadi sistem yang dapat digunakan oleh pengguna

---

##  Fitur

- **Prediksi Status Lalu Lintas**  
  Pengguna dapat memasukkan nilai `flow` untuk memperoleh prediksi kondisi lalu lintas.

- **Antarmuka Sederhana & Interaktif**  
  Dibangun menggunakan Streamlit agar mudah digunakan dan dipahami.

- **Rekomendasi Praktis**  
  Aplikasi memberikan saran singkat berdasarkan hasil prediksi (*macet* atau *lancar*).

---

##  Penjelasan Model

Model prediksi yang digunakan adalah **Random Forest Classifier**.  
Proses *training model* dilakukan **secara offline** menggunakan dataset berukuran besar, sehingga **dataset tidak dimuat langsung saat aplikasi dijalankan**.

Hasil training disimpan dalam file **`model.pkl`**, yang kemudian dimuat oleh aplikasi Streamlit untuk melakukan prediksi.  
Pendekatan ini dipilih agar:
- Aplikasi lebih ringan dan cepat
- Proses deployment lebih stabil
- Tidak bergantung pada dataset berukuran besar saat runtime

---

## 📂 Struktur Proyek
TUBES_DS/
├── app.py # File utama aplikasi Streamlit
├── model.pkl # Model machine learning hasil training
├── requirements.txt # Daftar dependensi Python
├── Zurich_DSA.ipynb # Notebook eksplorasi dan training data
├── zurich.py # Script training model (offline)
└── README.md


## Cara Menjalankan

1.  **Clone Repositori**
    ```bash
    git clone <URL_REPOSITORI_ANDA>
    cd <NAMA_FOLDER_PROYEK>
    ```

2.  **Instal Dependensi**
    Pastikan memiliki Python. Kemudian, instal semua pustaka yang diperlukan menggunakan file `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Jalankan Aplikasi Streamlit**
    Jalankan perintah berikut di terminal:
    ```bash
    streamlit run app.py
    ```

4.  **Buka di Browser**
    Aplikasi akan terbuka di browser, biasanya di url `http://localhost:8501`.
