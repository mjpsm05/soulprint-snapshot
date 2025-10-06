# app.py
from fastapi import FastAPI, Request
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
import xgboost as xgb
import numpy as np
import os

app = FastAPI()

# Global embedder (loaded once)
embedder = SentenceTransformer("all-mpnet-base-v2")

# Archetype model list
ARCHETYPES = [
    "Griot", "Kinara", "Ubuntu", "Jali", "Sankofa", "Imani", "Maji",
    "Nzinga", "Bisa", "Zamani", "Tamu", "Shujaa", "Ayo", "Ujamaa", "Kuumba"
]

# Cache for loaded models
model_cache = {}

# Function to load models dynamically
def load_model(archetype: str):
    if archetype not in model_cache:
        repo_id = f"mjpsm/{archetype}-xgb-model"
        filename = f"{archetype}_xgb_model.json"
        model_path = hf_hub_download(repo_id=repo_id, filename=filename, token=os.getenv("HUGGINGFACE_TOKEN"))
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        model_cache[archetype] = model
    return model_cache[archetype]


@app.post("/soulprint-snapshot")
async def soulprint_snapshot(request: Request):
    body = await request.json()
    text_input = body.get("chatInput", "")
    
    if not text_input:
        return {"error": "Missing 'chatInput' in request body."}
    
    # Encode input
    embedding = embedder.encode([text_input])
    
    # Predict across archetypes
    predictions = {}
    for archetype in ARCHETYPES:
        model = load_model(archetype)
        score = model.predict(np.array(embedding))[0]
        predictions[archetype] = round(float(score), 6)
    
    return predictions
