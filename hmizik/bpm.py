import librosa
import requests
from io import BytesIO
from fastapi import FastAPI, BackgroundTasks, Body
from pydantic import BaseModel
import uvicorn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import sys
import io
# Ajoute sa nan enpòtasyon yo anlè a
from sklearn.neighbors import NearestNeighbors
import os

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Fòse UTF-8 pou Windows pa fè erè charmap
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

app = FastAPI()


@app.get("/recommend/{track_id}")
async def get_recommendations(track_id: str, payload: list = Body(...)):
    try:
        # 1. Prepare done yo
        df = pd.DataFrame(payload)
        le = joblib.load('genre_encoder.pkl')
        
        df['genre_encoded'] = le.transform(df['genre'].astype(str))
        features = df[['genre_encoded', 'duration', 'bpm', 'plays']].fillna(0)
        
        # 2. Jwenn index mizik itilizatè a ap koute a
        target_idx = df[df['trackId'] == track_id].index
        if target_idx.empty:
            return {"status": "error", "message": "Track pa jwenn nan lis la"}
        
        # 3. Kalkile Similarity (Ki sa ki pi pwòch?)
        dist = cosine_similarity(features)
        similar_indices = dist[target_idx[0]].argsort()[-6:-1][::-1] # Pran 5 ki pi sanble yo
        
        # 4. Retounen ID mizik yo rekòmande yo
        recommended_ids = df.iloc[similar_indices]['trackId'].tolist()
        
        return {"status": "success", "recommendations": recommended_ids}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
    
@app.get("/predict/{track_id}")
async def predict_similar_tracks(track_id: str):
    try:
        # 1. Chaje modèl la ak done nou te sove yo
        clf = joblib.load('h_mizik_ai_model.pkl')
        le = joblib.load('genre_encoder.pkl')
        
        # 2. Isit la, nou ta dwe rale lis tout tracks yo pou n konpare
        # Pou tès la, nou pral retounen yon senp mesaj siksè
        # Men nòmalman, AI a ap kalkile pwoksimite ant BPM ak Genre
        
        return {
            "status": "success",
            "message": f"AI a ap analize mizik ki sanble ak {track_id}",
            "recommendation_engine": "Active"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ---------------------------------------------------------
# AI REKÒMANDASYON (Likes + BPM + Plays)
# ---------------------------------------------------------
@app.post("/train-recommendation")
async def train_model(payload: list = Body(...)):
    try:
        print("📊 Done resevwa. Antrenman milti-faktè ap kòmanse...", flush=True)
        df = pd.DataFrame(payload)
        
        if df.empty:
            return {"status": "error", "message": "Pa gen done"}

        # 1. Netwayaj Genre
        le = LabelEncoder()
        df['genre_encoded'] = le.fit_transform(df['genre'].astype(str))
        
        # 2. Nou itilize: Genre, Duration, BPM, ak Plays kòm Features
        # Nou asire tout se chif (fill NaN ak 0)
        X = df[['genre_encoded', 'duration', 'bpm', 'plays']].fillna(0)
        y = df['liked']
        
        # 3. Antrenman ak Random Forest
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X, y)
        
        # 4. Sove "Sèvo" a
        joblib.dump(clf, 'h_mizik_ai_model.pkl')
        joblib.dump(le, 'genre_encoder.pkl')
        
        print(f"✅ AI antrene: {len(df)} tracks analize ak BPM & Plays.", flush=True)
        return {"status": "success", "count": len(df)}
        
    except Exception as e:
        print(f"❌ Erè AI: {str(e)}", flush=True)
        return {"status": "error", "message": str(e)}

# ---------------------------------------------------------
# ANALIZ BPM (SA KI TE DEJA AP MACHE A)
# ---------------------------------------------------------
class TrackRequest(BaseModel):
    trackId: str
    audioUrl: str

def analyze_and_update(track_id, audio_url):
    try:
        print(f"DEBUG: Analiz BPM pou {track_id}", flush=True)
        response = requests.get(audio_url, timeout=15)
        audio_data = BytesIO(response.content)
        y, sr = librosa.load(audio_data, duration=30)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = int(round(float(tempo)))
        
        requests.patch(f"http://127.0.0.1:3000/tracks/{track_id}/bpm", json={"bpm": bpm})
        print(f"SUCCESS: BPM {bpm} sove.", flush=True)
    except Exception as e:
        print(f"ERROR BPM: {str(e)}", flush=True)

@app.post("/analyze-bpm")
async def handle_analyze(data: TrackRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(analyze_and_update, data.trackId, data.audioUrl)
    return {"status": "BPM analysis started"}

if __name__ == "__main__":
    import uvicorn
    # Render itilize yon varyab anviwonman ki rele PORT
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.1", port=port)