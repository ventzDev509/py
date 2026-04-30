import librosa
import requests
from io import BytesIO
from fastapi import FastAPI, BackgroundTasks, Body
# Enpòte CORSMiddleware isit la
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

# Fòse UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

app = FastAPI()

# ---------------------------------------------------------
# KONFIGIRASYON CORS
# ---------------------------------------------------------
# Ou ka mete ["*"] pou pèmèt tout moun, oswa mete URL NestJS ou a sèlman pou plis sekirite
origins = [
    "http://localhost:3000",
    "https://hmizikbackend-1.onrender.com",
    "*" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Varyab anviwonman pou URL NestJS la
BACKEND_URL = os.environ.get("BACKEND_URL", "https://hmizikbackend-1.onrender.com")

# ---------------------------------------------------------
# AI REKÒMANDASYON
# ---------------------------------------------------------
@app.post("/recommend/{track_id}")
async def get_recommendations(track_id: str, payload: list = Body(...)):
    try:
        df = pd.DataFrame(payload)
        
        if not os.path.exists('genre_encoder.pkl'):
             return {"status": "error", "message": "Modèl la poko antrene. Tanpri kouri /train-recommendation anvan."}
             
        le = joblib.load('genre_encoder.pkl')
        
        for col in ['duration', 'bpm', 'plays']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        df['genre_encoded'] = le.transform(df['genre'].astype(str))
        features = df[['genre_encoded', 'duration', 'bpm', 'plays']]
        
        target_idx = df[df['trackId'] == track_id].index
        if target_idx.empty:
            return {"status": "error", "message": "Track pa jwenn nan lis la"}
        
        dist = cosine_similarity(features)
        similar_indices = dist[target_idx[0]].argsort()[-6:-1][::-1]
        
        recommended_ids = df.iloc[similar_indices]['trackId'].tolist()
        return {"status": "success", "recommendations": recommended_ids}
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
# ANALIZ BPM
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