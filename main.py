import io
import os
import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

MODEL_ID = os.getenv("TB_MODEL_ID", "Owos/tb-classifier")
HF_TOKEN = os.getenv("HF_TOKEN", "")

app = FastAPI(title="IndraMed X-Ray AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}

@app.post("/api/analyze-xray")
async def analyze_xray(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "Image file required")
    
    raw = await image.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(413, "Maximum 10 MB limit")
    
    try:
        img = Image.open(io.BytesIO(raw))
        img.verify()
    except Exception:
        raise HTTPException(400, "Invalid image file")
    
    # Correct Hugging Face Serverless API URL
    url = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
    
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(url, headers=headers, data=raw)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f"Inference API failed ({response.status_code}): {response.text}"
                )
            results = response.json()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Inference request failed: {str(e)}")
            
    if not isinstance(results, list) or len(results) == 0:
        raise HTTPException(500, "Unexpected response format from Hugging Face")
        
    results = sorted(results, key=lambda x: float(x.get("score", 0)), reverse=True)
    top_result = results[0]
    
    return {
        "prediction": top_result.get("label"),
        "confidence": round(float(top_result.get("score", 0)), 4),
        "raw_results": results
    }
    
