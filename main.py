import io, os
import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

MODEL_ID = os.getenv("TB_MODEL_ID", "Owos/tb-classifier")
HF_TOKEN = os.getenv("HF_TOKEN", "")

app = FastAPI(title="IndraMed X-Ray AI API")
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
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
        raise HTTPException(413, "Maximum 10 MB limit exceeded")
    
    try: 
        img = Image.open(io.BytesIO(raw))
        img.verify()
    except Exception: 
        raise HTTPException(400, "Invalid image file")

    # Hugging Face Serverless API Call (RAM overhead = zero)
    url = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, content=raw, timeout=30.0)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Inference API failed ({response.status_code}): {response.text}"
                )
            results = response.json()
        except Exception as e:
            raise HTTPException(500, f"Inference request failed: {str(e)}")

    if not isinstance(results, list) or len(results) == 0:
        raise HTTPException(500, "Unexpected response format from AI model API")

    results = sorted(results, key=lambda x: float(x.get("score", 0)), reverse=True)
    top = results[0]

    return {
      "status": "ok",
      "model_version": MODEL_ID,
      "prediction": str(top.get("label", "UNKNOWN")),
      "confidence": float(top.get("score", 0)),
      "findings": [f"Model classification: {top.get('label', 'UNKNOWN')}"],
      "disclaimer": "Research prototype only; not clinically validated and not a diagnosis."
    }
