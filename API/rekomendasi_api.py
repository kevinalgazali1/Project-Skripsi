from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mysql.connector
import pandas as pd
import numpy as np
import re

app = FastAPI()

# --- koneksi ke database MySQL ---
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",  # ubah sesuai konfigurasi MySQL kamu
        database="testing"
    )

# --- fungsi bantu untuk membersihkan teks ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.get("/rekomendasi/{user_id}")
def rekomendasi(user_id: int):
    db = get_connection()
    cursor = db.cursor(dictionary=True)

    # Ambil data profil user
    cursor.execute("SELECT * FROM alumni_siswa_profiles WHERE user_id = %s", (user_id,))
    user_profile = cursor.fetchone()
    if not user_profile:
        return {"message": "User tidak ditemukan"}

    for key, value in user_profile.items():
        if value is None:
            user_profile[key] = ""

    # Ambil semua loker yang masih buka
    cursor.execute("SELECT * FROM lokers WHERE status = 'buka'")
    jobs = cursor.fetchall()
    if not jobs:
        return {"message": "Tidak ada loker yang tersedia"}

    df = pd.DataFrame(jobs).fillna("")

    # --- Bobot fitur user profile ---
    user_text = (
        (f"{user_profile['bidang_pekerjaan']} " * 3) +
        (f"{user_profile['jurusan_sekolah']} ") +
        (f"{user_profile['sertifikasi_terakhir']} ") +
        (f"{user_profile['skills']} " * 5)
    )
    user_text = clean_text(user_text)

    # --- Fitur tiap loker ---
    df['fitur_loker'] = (
        (df['posisi'].astype(str) + " ") +
        (df['pendidikan'].astype(str) + " ") +
        (df['deskripsi'].astype(str) + " ") +
        ((df['skills'].astype(str) + " ") * 5)
    )
    df['fitur_loker'] = df['fitur_loker'].apply(clean_text)

    # --- TF-IDF ---
    vectorizer = TfidfVectorizer(stop_words=None, min_df=1, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([user_text] + df['fitur_loker'].tolist())

    # --- cosine similarity ---
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # --- scaling agar hasil lebih lebar ---
    # normalisasi ke 0–1
    scaled = (cosine_similarities - np.min(cosine_similarities)) / (
        np.max(cosine_similarities) - np.min(cosine_similarities) + 1e-9
    )

    # log-scaling untuk efek dramatik
    df['similarity'] = np.log1p(scaled * 9) * 100  # hasil skala 0–100 lebih tinggi

    # --- tambah bonus poin jika kata bidang atau skill muncul eksplisit ---
    def bonus_score(row):
        score = 0
        if str(user_profile['bidang_pekerjaan']).lower() in row['fitur_loker']:
            score += 10
        for skill in str(user_profile['skills']).split(','):
            if skill.strip().lower() in row['fitur_loker']:
                score += 3
        return score

    df['bonus'] = df.apply(bonus_score, axis=1)
    df['final_score'] = df['similarity'] + df['bonus']

    min_score = df['final_score'].min()
    max_score = df['final_score'].max()

    df['relevansi_persen'] = (
        (df['final_score'] - min_score) / (max_score - min_score + 1e-9)
    ) * 100

    # --- Ambil top 10 ---
    top_jobs = df.sort_values(by='final_score', ascending=False).head(6)

    cursor.close()
    db.close()

    return {
        "user_id": user_id,
        "total_loker_ditemukan": len(df),
        "total_rekomendasi": len(top_jobs),
        "rekomendasi": top_jobs[
            ['id', 'nama_perusahaan', 'posisi', 'lokasi', 'pendidikan',
             'gambar', 'deskripsi', 'skills', 'similarity', 'bonus', 'final_score', 'relevansi_persen']
        ].to_dict(orient='records')
    }
