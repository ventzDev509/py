import librosa
import requests
from io import BytesIO
from fastapi import FastAPI, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import sys
import io
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

app = FastAPI()

# ---------------------------------------------------------
# KONFIGIRASYON CORS
# ---------------------------------------------------------
origins = [
    "http://localhost:3000",
    "https://hmizikbackend-1.onrender.com",
    "https://hmizik.onrender.com", # Ajoute URL Frontend ou tou si sa nesesè
    "*" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BACKEND_URL = os.environ.get("BACKEND_URL", "https://hmizikbackend-1.onrender.com")
# BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:3000")


def fallback_to_content_based(current_user, tracks_df):
    """
    Si nou pa jwenn done collaborative, nou rekòmande 
    mizik ki popilè oswa nou pran yon echantiyon o aza.
    """
    # Nou triye pa plays pou n bay "Hits" yo kòm premye opsyon
    popular_tracks = tracks_df.sort_values(by='plays', ascending=False).head(20)
    return {"status": "success", "recommendations": popular_tracks['id'].tolist(), "method": "fallback"}
# ---------------------------------------------------------
# AI REKÒMANDASYON (Optimize)
# ---------------------------------------------------------
@app.post("/recommend/{track_id}")
async def get_recommendations(track_id: str, payload: list = Body(...)):
    try:
        df = pd.DataFrame(payload)
        
        if not os.path.exists('genre_encoder.pkl'):
             return {"status": "error", "message": "AI a bezwen antrene anvan."}
             
        le = joblib.load('genre_encoder.pkl')
        
        # 1. Netwaye done (ajoute 'rating' si NestJS voye l)
        for col in ['duration', 'bpm', 'plays', 'rating']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0 # Si pa gen feedback ankò, nou mete 0

        # 2. Encoding Genre
        known_genres = set(le.classes_)
        df['genre'] = df['genre'].apply(lambda x: x if x in known_genres else list(known_genres)[0])
        df['genre_encoded'] = le.transform(df['genre'].astype(str))
        
        target_rows = df[df['trackId'] == track_id]
        if target_rows.empty:
            return {"status": "success", "recommendations": [], "debug": "ID not in payload"}
        
        target_idx = target_rows.index[0]
        
        # 3. Kalkile Similarity de baz
        features = df[['genre_encoded', 'duration', 'bpm', 'plays']]
        # Nou normalize done yo pou plays ak bpm pa kraze lòt yo
        dist = cosine_similarity(features)
        
        # 4. APLIKE FEEDBACK (Pwa pèsonalize)
        # Nou pran nòt similarity a, epi nou miltipliye l pa feedback la
        # Yon rating 1 ap double chans li, yon -1 ap fè l tounen 0
        final_scores = dist[target_idx].copy()
        
        for i in range(len(final_scores)):
            rating = df.iloc[i].get('rating', 0)
            if rating == -1:
                final_scores[i] = final_scores[i] * 0.1  # Pini l fò (desann li 90%)
            elif rating == 1:
                final_scores[i] = final_scores[i] * 1.5  # Boost li 50%

        # 5. Pran pi bon yo apre pwa a fin aplike
        top_n = min(len(df), 6)
        similar_indices = final_scores.argsort()[-top_n:]
        
        recommended_indices = [i for i in similar_indices if i != target_idx][::-1]
        recommended_ids = df.iloc[recommended_indices]['trackId'].tolist()
        
        return {"status": "success", "recommendations": recommended_ids}
        
    except Exception as e:
        print(f"❌ Erè: {str(e)}", flush=True)
        return {"status": "error", "message": str(e)}


@app.post("/discovery-pro")
async def discovery_pro(payload: dict = Body(...)):
    try:
        current_user = payload['current_user_id']
        interactions = pd.DataFrame(payload['interactions'])
        tracks_df = pd.DataFrame(payload['all_tracks'])

        # 1. Tcheke si gen ase entèraksyon nan sistèm nan
        if interactions.empty or len(interactions['userId'].unique()) < 2:
            return fallback_to_content_based(current_user, tracks_df)

        # 2. Kreye User-Item Matrix
        matrix = interactions.pivot_table(index='userId', columns='trackId', aggfunc='size', fill_value=0)

        # 3. Si itilizatè a se yon nouvo moun (Cold Start)
        if current_user not in matrix.index:
            return fallback_to_content_based(current_user, tracks_df)

        # 4. Collaborative Filtering (User-Based)
        user_sim = cosine_similarity(matrix)
        user_sim_df = pd.DataFrame(user_sim, index=matrix.index, columns=matrix.index)

        # Jwenn 5 moun ki pi sanble ak li
        similar_users = user_sim_df[current_user].sort_values(ascending=False).iloc[1:6].index

        recommendations = []
        currentUserTracks = matrix.loc[current_user]

        for other_user in similar_users:
            otherUserTracks = matrix.loc[other_user]
            # Mizik lòt moun nan renmen (1) ke mwen poko tande (0)
            new_tracks = otherUserTracks[(otherUserTracks > 0) & (currentUserTracks == 0)].index.tolist()
            recommendations.extend(new_tracks)

        # Retire kopi epi pran 20 pi bon yo
        final_ids = list(dict.fromkeys(recommendations))[:20]

        # 5. Si nou toujou pa gen ase mizik, nou konplete l ak Hits
        if len(final_ids) < 10:
            hits = tracks_df.sort_values(by='plays', ascending=False).head(10)['id'].tolist()
            final_ids = list(dict.fromkeys(final_ids + hits))[:20]

        return {"status": "success", "recommendations": final_ids, "method": "collaborative"}

    except Exception as e:
        print(f"❌ Erè Python: {str(e)}")
        # Nan ka gwo erè, toujou retounen yon bagay pou app a pa bloke
        return {"status": "error", "recommendations": tracks_df.head(10)['id'].tolist()}   
@app.post("/discovery")
async def discovery_weekly(payload: dict = Body(...)):
    try:
        # 1. Done NestJS voye yo
        history_data = pd.DataFrame(payload['positive_history'])
        candidates_df = pd.DataFrame(payload['candidates'])
        
        if history_data.empty:
            # Si itilizatè a se yon nouvo moun, nou ba li mizik ki popilè senpleman
            return {"recommendations": candidates_df.head(20)['id'].tolist()}

        # Chaje encoder la pou genre
        le = joblib.load('genre_encoder.pkl')
        known_genres = set(le.classes_)

        # Fonksyon pou netwaye ak encode
        def prepare_df(df):
            df['genre'] = df['genre'].fillna('Konpa')
            df['genre'] = df['genre'].apply(lambda x: x if x in known_genres else list(known_genres)[0])
            df['genre_encoded'] = le.transform(df['genre'].astype(str))
            for col in ['bpm', 'duration']:
                 df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            return df

        history_df = prepare_df(history_data)
        candidates_df = prepare_df(candidates_df)

        # 2. Kalkile "Vektè Gou" Itilizatè a (Mwayèn karakteristik li renmen)
        user_profile_vector = history_df[['genre_encoded', 'bpm', 'duration']].mean().values.reshape(1, -1)

        # 3. Kalkile Similarity ant Gou li ak tout Nouvo Mizik yo
        candidate_features = candidates_df[['genre_encoded', 'bpm', 'duration']]
        scores = cosine_similarity(user_profile_vector, candidate_features)[0]

        # 4. Triye epi pran 20 pi bon yo
        candidates_df['score'] = scores
        best_discovery = candidates_df.sort_values(by='score', ascending=False).head(20)

        return {"status": "success", "recommendations": best_discovery['id'].tolist()}

    except Exception as e:
        return {"status": "error", "message": str(e)}
    
# ---------------------------------------------------------
# ANTRENMAN MODÈL LA
# ---------------------------------------------------------
@app.post("/train-recommendation")
async def train_model(payload: list = Body(...)):
    try:
        df = pd.DataFrame(payload)
        
        cols_to_fix = ['duration', 'bpm', 'plays', 'liked']
        for col in cols_to_fix:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        if 'genre' not in df.columns:
             return {"status": "error", "message": "Jaden 'genre' manke"}

        le = LabelEncoder()
        df['genre_encoded'] = le.fit_transform(df['genre'].astype(str))
        
        X = df[['genre_encoded', 'duration', 'bpm', 'plays']]
        y = df['liked']
        
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X, y)
        
        joblib.dump(clf, 'h_mizik_ai_model.pkl')
        joblib.dump(le, 'genre_encoder.pkl')
        
        print(f"✅ AI antrene: {len(df)} tracks.", flush=True)
        return {"status": "success", "count": len(df)}
    except Exception as e:
        print(f"❌ Erè AI Train: {str(e)}", flush=True)
        return {"status": "error", "message": str(e)}

# ---------------------------------------------------------
# ANALIZ BPM (Avèk ranje scalar)
# ---------------------------------------------------------
class TrackRequest(BaseModel):
    trackId: str
    audioUrl: str

def analyze_and_update(track_id, audio_url):
    try:
        print(f"DEBUG: Kòmanse analiz BPM pou {track_id}", flush=True)
        response = requests.get(audio_url, timeout=15)
        audio_data = BytesIO(response.content)
        
        y, sr = librosa.load(audio_data, duration=30)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Ranje scalar a
        if isinstance(tempo, (np.ndarray, list)):
            bpm_val = float(tempo[0])
        else:
            bpm_val = float(tempo)
            
        bpm = int(round(bpm_val))
        
        patch_url = f"{BACKEND_URL}/tracks/{track_id}/bpm"
        requests.patch(patch_url, json={"bpm": bpm})
        
        print(f"✅ SUCCESS: BPM {bpm} sove pou {track_id}", flush=True)
    except Exception as e:
        print(f"❌ ERROR BPM: {str(e)}", flush=True)

@app.post("/analyze-bpm")
async def handle_analyze(data: TrackRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(analyze_and_update, data.trackId, data.audioUrl)
    return {"status": "BPM analysis started"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)