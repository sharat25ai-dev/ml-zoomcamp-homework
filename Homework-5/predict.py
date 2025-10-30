import pickle

import uvicorn
from fastapi import FastAPI

from typing import Dict, Any

app = FastAPI(title="Lead Prediction API")

with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

@app.post('/predict')
def predict(data: Dict[str, Any]):
    lead = float(pipeline.predict_proba(data)[0, 1])
    return {
        "lead probability": lead,
        "lead status": bool(lead >= 0.5)
    }


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=9696)

