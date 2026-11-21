import os
import io
import time

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from google import genai


app = FastAPI(title="IndraMed AI Backend")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


API_KEY = os.getenv("GEMINI_API_KEY")
PRIMARY_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite")

client = genai.Client(api_key=API_KEY) if API_KEY else None


def generate_with_retry(contents, max_retries_per_model=2):
    models_to_try = [PRIMARY_MODEL]
    for m in [FALLBACK_MODEL, "gemini-2.5-pro", "gemini-2.0-flash"]:
        if m and m not in models_to_try:
            models_to_try.append(m)

    last_error = None

    for model_name in models_to_try:
        for attempt in range(max_retries_per_model + 1):
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                )
                text = getattr(resp, "text", None)
                if text and text.strip():
                    return text, model_name
                last_error = RuntimeError("Gemini returned an empty response.")
            except Exception as e:
                last_error = e
                err_str = str(e)
                is_transient = (
                    "503" in err_str
                    or "UNAVAILABLE" in err_str
                    or "429" in err_str
                    or "RESOURCE_EXHAUSTED" in err_str
                    or "high demand" in err_str.lower()
                    or "overloaded" in err_str.lower()
                )
                if is_transient and attempt < max_retries_per_model:
                    time.sleep(1.5 * (2 ** attempt))
                    continue
                break

    raise last_error


@app.get("/")
def home():
    return {
        "status": "IndraMed AI Backend is Running",
        "model": PRIMARY_MODEL,
        "fallback_model": FALLBACK_MODEL,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": PRIMARY_MODEL,
        "fallback_model": FALLBACK_MODEL,
    }


@app.get("/models")
def list_available_models():
    if not API_KEY or not client:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY missing in Render environment variables."
        )
    try:
        available = []
        for m in client.models.list():
            name = getattr(m, "name", "")
            if "gemini" in name.lower():
                available.append(name)
        return {"models": available}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze-xray")
async def analyze_xray(image: UploadFile = File(...)):
    try:
        if not API_KEY or not client:
            raise HTTPException(
                status_code=500,
                detail="GEMINI_API_KEY missing in Render environment variables."
            )

        if not image.content_type:
            raise HTTPException(
                status_code=400,
                detail="Image file required."
            )

        if not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Please upload a valid image file."
            )

        contents = await image.read()

        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Maximum image size is 10 MB."
            )

        try:
            pil_image = Image.open(io.BytesIO(contents))
            pil_image.verify()

            pil_image = Image.open(io.BytesIO(contents))
            pil_image.load()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file."
            )

        prompt = """
You are an AI-assisted medical imaging research system for IndraMed.ai.

Analyze the provided chest X-ray image.

IMPORTANT:
This is a research/prototype assessment and NOT a definitive medical diagnosis.
Do not claim certainty.
Do not prescribe treatment autonomously.

Return a concise structured assessment containing:

1. Image quality
2. Major visible findings
3. Lung fields
4. Pleura
5. Heart / mediastinum
6. Any suspicious abnormality
7. TB-related features, if visibly present
8. Overall impression
9. Recommended clinical correlation or further investigation

If image quality is insufficient, clearly state that.

If tuberculosis cannot be determined from the image alone, explicitly say so.

Use cautious clinical language and mention uncertainty where appropriate.
"""

        analysis_text, used_model = generate_with_retry([prompt, pil_image])

        return {
            "success": True,
            "model": used_model,
            "analysis": analysis_text,
            "impression": analysis_text,
            "disclaimer": (
                "Research/prototype AI output only. "
                "Not a confirmed diagnosis and not a substitute "
                "for qualified clinical assessment."
            )
        }

    except HTTPException:
        raise

    except Exception as e:
        print("X-RAY ANALYSIS ERROR:", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )
        
