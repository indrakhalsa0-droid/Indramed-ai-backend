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
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


API_KEY = os.getenv("GEMINI_API_KEY")
PRIMARY_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")

client = genai.Client(api_key=API_KEY) if API_KEY else None


def build_model_chain():
    chain = []
    for m in [
        PRIMARY_MODEL,
        FALLBACK_MODEL,
        "gemini-2.5-flash-lite",
        "gemini-3.8-flash",
        "gemini-2.5-flash",
    ]:
        if m and "2.0" not in m and m not in chain:
            chain.append(m)
    return chain


def generate_with_retry(contents, max_retries_per_model=1):
    models_to_try = build_model_chain()
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

                if "404" in err_str or "NOT_FOUND" in err_str:
                    break

                is_transient = (
                    "503" in err_str
                    or "UNAVAILABLE" in err_str
                    or "429" in err_str
                    or "RESOURCE_EXHAUSTED" in err_str
                    or "high demand" in err_str.lower()
                    or "overloaded" in err_str.lower()
                )
                if is_transient and attempt < max_retries_per_model:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                break

    raise last_error


@app.get("/")
def home():
    return {
        "status": "IndraMed AI Backend is Running",
        "model": PRIMARY_MODEL,
        "fallback_model": FALLBACK_MODEL,
        "models_chain": build_model_chain(),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": PRIMARY_MODEL,
        "fallback_model": FALLBACK_MODEL,
        "models_chain": build_model_chain(),
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
You are IndraMed AI, a structured clinical radiograph analysis system.

Analyze the provided X-ray image thoroughly and clearly.
Do NOT write "not for medical use", "research prototype only", or repetitive disclaimers.
Present the response in clean Markdown using exact headings and VERTICAL numbered points using hyphen format:
1- point one
2- point two
3- point three

Never combine multiple points horizontally on one line. Put each point on its own new line.
Bold all key/main findings using **double asterisks**.

Use this exact structure:

### 1. Image Quality & Technical View
1- **Projection & Positioning**: ...
2- **Exposure, Penetration & Motion/Artifacts**: ...

### 2. Main Radiographic Findings
1- **Primary Finding**: ...
2- **Lung Fields & Parenchyma**: ...
3- **Pleura, Costophrenic Angles & Diaphragm**: ...
4- **Heart, Mediastinum & Great Vessels**: ...
5- **Bones & Soft Tissues**: ...

### 3. Differential Diagnosis
1- **Most Likely Diagnosis**: ...
2- **Alternative Possibility 1**: ...
3- **Alternative Possibility 2**: ...

### 4. Recommended Investigations
1- **Laboratory Tests**: ...
2- **Further Imaging**: ...
3- **Microbiological / Diagnostic Workup**: ...

### 5. Treatment & Clinical Management
1- **Immediate / Supportive Measures**: ...
2- **Targeted / Empirical Therapy Considerations**: ...
3- **Follow-Up & Specialist Referral**: ...

### 6. Summary Impression
1- **Key Takeaway**: ...
"""

        analysis_text, used_model = generate_with_retry([prompt, pil_image])

        return {
            "success": True,
            "model": used_model,
            "analysis": analysis_text,
            "impression": analysis_text,
        }

    except HTTPException:
        raise

    except Exception as e:
        print("X-RAY ANALYSIS ERROR:", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )
        
