from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mysql.connector
import pandas as pd
import re


app = FastAPI()

# --- Koneksi ke database MySQL ---
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="testing2"
    )

# --- Fungsi pembersihan teks ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.get("/rekomendasi/{user_id}")
def rekomendasi(user_id: int):
    # 1. LOAD DATA FROM DATABASE
    db = get_connection()
    cursor = db.cursor(dictionary=True)
    
    # Ambil profil user
    cursor.execute("SELECT * FROM alumni_siswa_profiles WHERE user_id = %s", (user_id,))
    user_profile = cursor.fetchone()
    
    if not user_profile:
        cursor.close()
        db.close()
        return {"message": "User tidak ditemukan"}
    
    # Ambil semua loker yang buka
    cursor.execute("SELECT * FROM lokers WHERE status = 'buka'")
    jobs = cursor.fetchall()
    
    cursor.close()
    db.close()
    
    if not jobs:
        return {"message": "Tidak ada loker yang tersedia"}
    
    # Convert ke DataFrame
    df = pd.DataFrame(jobs).fillna("")
    
    # Handle null values di user profile
    for key, value in user_profile.items():
        if value is None:
            user_profile[key] = ""
    
    # 2. CLEAN TEXT (DENGAN PEMBOBOTAN)
    # Referensi: Alsaif & Hidri (2022) - "assigned higher weights to job skills"
    # Gabungkan fitur user dengan pembobotan berbeda
    user_text_raw = (
        f"{user_profile['bidang_pekerjaan']} " * 3 +      # Bobot 3x
        f"{user_profile['jurusan_sekolah']} " +           # Bobot 1x
        f"{user_profile['sertifikasi_terakhir']} " +      # Bobot 1x
        f"{user_profile['skills']} " * 5                  # Bobot 5x (prioritas tertinggi)
    )
    
    # Clean text user
    user_text_clean = clean_text(user_text_raw)
    
    # Gabungkan fitur loker dengan pembobotan yang sama
    df['fitur_loker_raw'] = (
        df['posisi'].astype(str) + " " +
        df['pendidikan'].astype(str) + " " +
        df['deskripsi'].astype(str) + " " +
        (df['skills'].astype(str) + " ") * 5               # Bobot 5x untuk skills
    )
    df['fitur_loker_clean'] = df['fitur_loker_raw'].apply(clean_text)
    
    # 3. CALCULATE TF-IDF MATRIX
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    tfidf_matrix = vectorizer.fit_transform([user_text_clean] + df['fitur_loker_clean'].tolist())
    
    # Ambil feature names dan TF-IDF values untuk user
    feature_names = vectorizer.get_feature_names_out()
    user_tfidf_vector = tfidf_matrix[0].toarray()[0]
    
    # Ambil top 10 terms dengan TF-IDF tertinggi untuk user
    top_indices = user_tfidf_vector.argsort()[-10:][::-1]
    user_top_terms = [
        {
            "term": feature_names[i],
            "tfidf_score": round(float(user_tfidf_vector[i]), 4)
        }
        for i in top_indices if user_tfidf_vector[i] > 0
    ]
    
    # 4. CALCULATE COSINE SIMILARITY
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    df['similarity_raw'] = similarities
    
    # 5. SCALE SIMILARITY SCORE
    # Konversi ke persentase (0-100)
    df['similarity_persen'] = df['similarity_raw'] * 100
    
    # 6. SELECT TOP 6 RECOMMENDATIONS
    top_jobs = df.nlargest(6, 'similarity_persen')
    
    # 7. LIST OF RECOMMENDATIONS dengan detail perhitungan
    rekomendasi_list = []
    for idx, row in top_jobs.iterrows():
        # Ambil TF-IDF vector untuk loker ini
        loker_idx = df.index.get_loc(idx) + 1
        loker_tfidf_vector = tfidf_matrix[loker_idx].toarray()[0]
        
        # Hitung dot product manual
        dot_product = sum(user_tfidf_vector * loker_tfidf_vector)
        user_magnitude = sum(user_tfidf_vector ** 2) ** 0.5
        loker_magnitude = sum(loker_tfidf_vector ** 2) ** 0.5
        cosine_sim_manual = dot_product / (user_magnitude * loker_magnitude) if (user_magnitude * loker_magnitude) > 0 else 0
        
        # Ambil top matching terms
        matching_terms = []
        for i, term in enumerate(feature_names):
            if user_tfidf_vector[i] > 0 and loker_tfidf_vector[i] > 0:
                matching_terms.append({
                    "term": term,
                    "user_tfidf": round(float(user_tfidf_vector[i]), 4),
                    "loker_tfidf": round(float(loker_tfidf_vector[i]), 4),
                    "contribution": round(float(user_tfidf_vector[i] * loker_tfidf_vector[i]), 4)
                })
        
        matching_terms = sorted(matching_terms, key=lambda x: x['contribution'], reverse=True)[:5]
        
        rekomendasi_list.append({
            "id": int(row['id']),
            "nama_perusahaan": row['nama_perusahaan'],
            "posisi": row['posisi'],
            "lokasi": row['lokasi'],
            "pendidikan": row['pendidikan'],
            "gambar": row['gambar'],
            "deskripsi": row['deskripsi'],
            "skills": row['skills'],
            "perhitungan": {
                "cosine_similarity_raw": round(float(row['similarity_raw']), 4),
                "similarity_persen": round(float(row['similarity_persen']), 2),
                "dot_product": round(float(dot_product), 4),
                "user_magnitude": round(float(user_magnitude), 4),
                "loker_magnitude": round(float(loker_magnitude), 4),
                "formula": f"cosine_similarity = {round(float(dot_product), 4)} / ({round(float(user_magnitude), 4)} * {round(float(loker_magnitude), 4)}) = {round(float(cosine_sim_manual), 4)}",
                "top_matching_terms": matching_terms
            }
        })
    
    # Summary statistik
    summary_stats = {
        "min_similarity": round(float(df['similarity_persen'].min()), 2),
        "max_similarity": round(float(df['similarity_persen'].max()), 2),
        "mean_similarity": round(float(df['similarity_persen'].mean()), 2),
        "median_similarity": round(float(df['similarity_persen'].median()), 2)
    }
    
    result = {
        "user_id": user_id,
        "user_profile": {
            "bidang_pekerjaan": user_profile['bidang_pekerjaan'],
            "jurusan_sekolah": user_profile['jurusan_sekolah'],
            "sertifikasi_terakhir": user_profile['sertifikasi_terakhir'],
            "skills": user_profile['skills']
        },
        "preprocessing": {
            "user_text_raw": user_text_raw[:200] + "..." if len(user_text_raw) > 200 else user_text_raw,
            "user_text_clean": user_text_clean[:200] + "..." if len(user_text_clean) > 200 else user_text_clean,
            "weighting_scheme": "Skills: 5x, Bidang Pekerjaan: 3x, Lainnya: 1x",
            "total_terms_in_vocabulary": len(feature_names),
            "user_top_tfidf_terms": user_top_terms
        },
        "tfidf_info": {
            "ngram_range": "(1, 2)",
            "min_df": 1,
            "total_features": len(feature_names),
            "user_vector_size": len(user_tfidf_vector),
            "non_zero_terms": int(sum(user_tfidf_vector > 0))
        },
        "summary_statistics": summary_stats,
        "total_loker_ditemukan": len(df),
        "total_rekomendasi": len(top_jobs),
        "rekomendasi": rekomendasi_list
    }
    
    return result
