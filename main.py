import io, os
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import pipeline

MODEL_ID=os.getenv("TB_MODEL_ID","Owos/tb-classifier")
app=FastAPI(title="IndraMed X-Ray AI API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
classifier=None

@app.on_event("startup")
def load_model():
    global classifier
    classifier=pipeline("image-classification", model=MODEL_ID)

@app.get("/health")
def health():
    return {"status":"ok","model":MODEL_ID}

@app.post("/api/analyze-xray")
async def analyze_xray(image: UploadFile=File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400,"Image file required")
    raw=await image.read()
    if len(raw)>10*1024*1024: raise HTTPException(413,"Maximum 10 MB")
    try: img=Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception: raise HTTPException(400,"Invalid image")
    try: results=classifier(img)
    except Exception as e: raise HTTPException(500,f"Inference failed: {e}")
    results=sorted(results,key=lambda x:float(x.get("score",0)),reverse=True)
    top=results[0]
    return {
      "status":"ok",
      "model_version":MODEL_ID,
      "prediction":str(top.get("label","UNKNOWN")),
      "confidence":float(top.get("score",0)),
      "findings":[f"Model classification: {top.get('label','UNKNOWN')}"],
      "disclaimer":"Research prototype only; not clinically validated and not a diagnosis."
    }
