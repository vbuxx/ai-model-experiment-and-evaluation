# AI Model Experiment & Evaluation

Project ini membandingkan model klasifikasi teks klasik dengan Large Language Model (LLM) berbasis Gemini API untuk mengklasifikasikan sentiment ulasan pelanggan e-commerce.

## 1. Problem statement

### Objective

Membuat eksperimen awal untuk menentukan pendekatan yang paling sesuai untuk fitur otomatisasi klasifikasi sentiment pada halaman produk e-commerce. Setiap ulasan diklasifikasikan menjadi `positif` atau `negatif`.

### Target/label

- `positif`: pelanggan puas, produk sesuai, atau pengalaman belanja baik.
- `negatif`: pelanggan kecewa, menemukan masalah, atau mengalami pengalaman belanja buruk.

### Batasan dan asumsi

- Dataset yang digunakan adalah dataset dari brief assignment dan tidak diubah.
- Perbandingan memakai test set yang sama untuk Classic ML dan Gemini API.
- Eksperimen hanya mencakup klasifikasi biner dan bahasa ulasan yang tersedia di dataset, terutama Bahasa Indonesia.
- Model klasik tidak memakai kolom `product_name`; fitur yang digunakan hanya `review_text`.
- Inference Gemini memerlukan `GEMINI_API_KEY`. Jika key belum tersedia, script tetap bisa dijalankan untuk memvalidasi pendekatan Classic ML, tetapi metrik Gemini belum dihitung.
- Hasil Gemini dapat berubah jika model/API yang dipakai berubah. Karena itu nama model, prompt, dan parameter generasi dicatat di kode.

## 2. Dataset

Dataset asli disimpan di [data/customer_reviews_sentiment.csv](data/customer_reviews_sentiment.csv). Sumbernya adalah spreadsheet resmi yang diberikan di brief assignment:

<https://docs.google.com/spreadsheets/d/1ry3h8o_MxeKR0bDolcZ8Au9giMgZl7Q0cyfjdhj1XLE/edit?usp=sharing>

Karakteristik dataset:

- 200 ulasan pelanggan.
- Kolom: `review_id`, `product_name`, `review_text`, dan `sentiment`.
- Target: `sentiment` dengan dua kelas, `positif` dan `negatif`.
- Pembagian data: 80% training set dan 20% test set, stratified, `random_state=42`.

## 3. Pendekatan eksperimen

### Classic ML

Pipeline yang digunakan:

1. `TfidfVectorizer` dengan unigram dan bigram.
2. `LogisticRegression` dengan `max_iter=1000`.

Model hanya dilatih pada training set. Prediksi dan evaluasi dilakukan pada test set.

### Gemini API

Gemini menerima seluruh ulasan test dalam satu batch request dengan prompt few-shot sederhana. Prompt memberi definisi label, dua contoh, dan instruksi agar model mengembalikan JSON berisi satu label untuk setiap nomor ulasan. Batch dipakai agar request lebih hemat dan tidak mudah terkena rate limit.

Parameter utama:

- Model default: `gemini-3.5-flash-lite` dan bisa diubah melalui `GEMINI_MODEL`.
- `thinking_level="low"` untuk Gemini 3.x, karena klasifikasi ini sederhana dan tidak membutuhkan reasoning panjang.
- `max_output_tokens=512` untuk memberi ruang bagi proses thinking dan seluruh JSON label test set.
- Untuk model Gemini versi lama, kode otomatis memakai `temperature=0.0` dan `max_output_tokens=10`.

Implementasi memakai Google GenAI SDK baru (`google-genai`) dengan pola `client.models.generate_content(...)`.

## 4. Hasil evaluasi

Jalankan eksperimen terlebih dahulu agar file hasil berikut dibuat:

- [results/predictions.csv](results/predictions.csv): label aktual, prediksi Classic ML, prediksi Gemini, dan raw response Gemini.
- [results/model_comparison.csv](results/model_comparison.csv): tabel metrik.
- [documentation/model_comparison_summary.png](documentation/model_comparison_summary.png): confusion matrix.

Hasil eksperimen pada 40 baris test set:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Classic ML | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Gemini API | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Confusion matrix kedua pendekatan juga sama:

| Actual \\ Predicted | negatif | positif |
|---|---:|---:|
| negatif | 18 | 0 |
| positif | 0 | 22 |

Pada dataset dan split ini, kedua pendekatan tidak menghasilkan false positive maupun false negative. Hasil ini menunjukkan performa yang sama pada eksperimen awal, tetapi belum cukup untuk menyimpulkan bahwa keduanya akan sama baiknya pada data produksi yang lebih beragam.

Interpretasi metrik:

- Accuracy mengukur proporsi semua prediksi yang benar.
- Precision mengukur ketepatan ketika model memprediksi kelas positif.
- Recall mengukur kemampuan menemukan seluruh contoh positif.
- F1-score merangkum keseimbangan precision dan recall.
- Confusion matrix membantu melihat false positive dan false negative, bukan hanya skor agregat.

## 5. Analisis trade-off dan limitation

### Trade-off

- **Effort implementasi:** Classic ML membutuhkan preprocessing, training, dan pemeliharaan model. Gemini tidak membutuhkan proses training, tetapi tetap membutuhkan desain prompt, integrasi API, dan pengelolaan error.
- **Kecepatan inference:** Classic ML berjalan lokal setelah model dilatih sehingga latency dan throughput lebih mudah dikontrol. Gemini membutuhkan network request ke API; implementasi ini mengurangi overhead dengan mengirim test set dalam satu batch.
- **Biaya:** Classic ML umumnya tidak memiliki biaya per request setelah infrastruktur tersedia. Gemini memiliki potensi biaya berdasarkan penggunaan token/API dan membutuhkan pengaturan rate limit.
- **Kualitas:** Pada eksperimen ini kedua pendekatan memiliki metrik yang sama-sama 1.0000. Dataset kecil dan pola kalimat yang cukup jelas membuat hasil ini belum cukup untuk menggeneralisasi ke seluruh variasi ulasan produksi.

### Limitation

- Dataset hanya berisi 200 baris, sehingga skor test set sensitif terhadap beberapa prediksi.
- Label hanya biner; ulasan netral, campuran, sarkasme, atau multi-aspek belum dipisahkan.
- Model Classic ML belum melakukan tuning hyperparameter atau normalisasi Bahasa Indonesia yang lebih mendalam.
- Gemini dapat menghasilkan format output yang tidak sesuai instruksi atau gagal karena masalah jaringan, rate limit, dan perubahan model. Parser memberi status prediksi tidak valid agar tidak diam-diam dianggap benar.
- Eksperimen API mengukur satu konfigurasi prompt dan satu model; hasil dapat berubah pada model, prompt, atau waktu yang berbeda.

## 6. Rekomendasi technical approach

Pada eksperimen ini performa Classic ML dan Gemini sama-sama sempurna pada test set, sehingga tidak ada keunggulan metrik yang membenarkan tambahan biaya dan latency API. Untuk baseline produksi dengan volume ulasan tinggi dan label yang relatif stabil, rekomendasi awalnya adalah **Classic ML** karena inference lokal, latency lebih terprediksi, dan biaya per prediksi rendah. Gemini lebih menarik sebagai baseline pembanding, fallback untuk kasus ambigu, atau alat untuk membantu membuat data berlabel.

Jika hasil eksperimen menunjukkan Gemini jauh lebih baik pada recall/F1 dan volume request masih kecil, pendekatan hybrid dapat dipertimbangkan: Classic ML menangani prediksi umum, sedangkan Gemini hanya menerima kasus dengan confidence rendah. Sebelum keputusan final, model terpilih perlu diuji pada dataset yang lebih besar, monitoring drift, dan evaluasi biaya/latency yang lebih realistis.

## 7. Cara menjalankan

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Jalankan validasi Classic ML saja:

```bash
RUN_GEMINI=false python run_experiment.py
```

Jalankan kedua pendekatan untuk memenuhi eksperimen assignment:

```bash
export GEMINI_API_KEY="isi_api_key_di_environment"
export RUN_GEMINI=true
python run_experiment.py
```

Model Gemini dapat diganti tanpa mengubah kode:

```bash
export GEMINI_MODEL="gemini-3.5-flash-lite"
```

Buka notebook secara interaktif:

```bash
jupyter notebook notebook/experiment_notebook.ipynb
```

Notebook dan script menggunakan split, prompt, dan fungsi evaluasi yang sama. Jangan menulis API key langsung ke file atau commit API key ke GitHub.

## 8. Referensi teknis

- [Google Gemini API — Generating content](https://ai.google.dev/api/generate-content)
- [Google GenAI SDK migration guide](https://ai.google.dev/gemini-api/docs/migrate)
