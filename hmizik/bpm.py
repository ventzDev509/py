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

# Fòse UTF-8
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

# ---------------------------------------------------------
# AI REKÒMANDASYON (Optimize)
# ---------------------------------------------------------
@app.post("/recommend/{track_id}")
async def get_recommendations(track_id: str, payload: list = Body(...)):
    try:
        df = pd.DataFrame(payload)
        
        # 1. Tcheke fichye modèl yo
        if not os.path.exists('genre_encoder.pkl'):
             return {"status": "error", "message": "AI a bezwen antrene anvan."}
             
        le = joblib.load('genre_encoder.pkl')
        
        # 2. Netwaye done yo
        for col in ['duration', 'bpm', 'plays']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 3. Sekirite Genre (Evite erè si yon jan pa t nan antrenman an)
        known_genres = set(le.classes_)
        df['genre'] = df['genre'].apply(lambda x: x if x in known_genres else list(known_genres)[0])
        df['genre_encoded'] = le.transform(df['genre'].astype(str))
        
        # 4. Verifye si ID a egziste nan sa NestJS voye a
        target_rows = df[df['trackId'] == track_id]
        if target_rows.empty:
            # Si li pa jwenn li, li pap ka konpare anyen
            print(f"⚠️ ID {track_id} pa nan payload la", flush=True)
            return {"status": "success", "recommendations": [], "debug": "ID not in payload"}
        
        target_idx = target_rows.index[0]
        
        # 5. Kalkile Similarity
        features = df[['genre_encoded', 'duration', 'bpm', 'plays']]
        dist = cosine_similarity(features)
        
        # 6. Pran n_total rekòmandasyon (max 6)
        n_total = len(df)
        top_n = min(n_total, 6)
        
        # Jwenn n pi gwo nòt yo
        similar_indices = dist[target_idx].argsort()[-top_n:]
        # Retire tèt li (mizik k ap jwe a) epi ranvèse lis la
        recommended_indices = [i for i in similar_indices if i != target_idx][::-1]
        
        recommended_ids = df.iloc[recommended_indices]['trackId'].tolist()
        
        print(f"✅ OK! Rekòmande {len(recommended_ids)} mizik pou {track_id}", flush=True)
        return {"status": "success", "recommendations": recommended_ids}
        
    except Exception as e:
        print(f"❌ Erè: {str(e)}", flush=True)
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