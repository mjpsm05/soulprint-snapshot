# app.py
from fastapi import FastAPI, Request
import xgboost as xgb
import numpy as np
import os
from huggingface_hub import hf_hub_download

app = FastAPI()

# Lazy globals
embedder = None
model_cache = {}

ARCHETYPES = [
    "Griot", "Kinara", "Ubuntu", "Jali", "Sankofa", "Imani", "Maji",
    "Nzinga", "Bisa", "Zamani", "Tamu", "Shujaa", "Ayo", "Ujamaa", "Kuumba"
]

def get_embedder():
    """Load embedder lazily and cache it."""
    global embedder
    if embedder is None:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-mpnet-base-v2")
    return embedder

def load_model(archetype: str):
    """Download + load a model lazily at runtime."""
    if archetype not in model_cache:
        repo_id = f"mjpsm/{archetype}-xgb-model"
        filename = f"{archetype}_xgb_model.json"
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir="/tmp"   # use temp dir to fit Vercel’s limits
        )
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        model_cache[archetype] = model
    return model_cache[archetype]


@app.post("/soulprint-snapshot")
async def soulprint_snapshot(request: Request):
    body = await request.json()
    text_input = body.get("chatInput", "")
    if not text_input:
        return {"error": "Missing 'chatInput'."}

    embedder_model = get_embedder()
    embedding = embedder_model.encode([text_input])

    results = {}
    for archetype in ARCHETYPES:
        model = load_model(archetype)
        score = model.predict(np.array(embedding))[0]
        results[archetype] = round(float(score), 6)

    return results

