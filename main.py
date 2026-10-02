import os
import io

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import google.generativeai as genai


app = FastAPI(title="IndraMed AI Backend")


# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# GEMINI CONFIGURATION
# =========================

API_KEY = os.getenv("GEMINI_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)


# =========================
# HOME / HEALTH
# =========================

@app.get("/")
def home():
    return {
        "status": "IndraMed AI Backend is Running",
        "model": "gemini-2.5-flash"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "gemini-2.5-flash"
    }


# =========================
# X-RAY ANALYSIS
# =========================

@app.post("/api/analyze-xray")
async def analyze_xray(image: UploadFile = File(...)):

    try:

        # Check API key
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="GEMINI_API_KEY missing in Render environment variables."
            )


        # Check file type
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


        # Read image
        contents = await image.read()


        # Maximum 10 MB
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Maximum image size is 10 MB."
            )


        # Validate image
        try:
            pil_image = Image.open(io.BytesIO(contents))
            pil_image.verify()

            # Re-open after verify
            pil_image = Image.open(io.BytesIO(contents))

        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file."
            )


        # =========================
        # GEMINI MODEL
        # =========================

        model = genai.GenerativeModel("gemini-2.5-flash")


        # =========================
        # MEDICAL X-RAY PROMPT
        # =========================

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

If the image quality is insufficient, clearly state that.

If tuberculosis cannot be determined from the image alone, explicitly say that.

Use cautious clinical language and mention uncertainty where appropriate.
"""


        # =========================
        # SEND IMAGE TO GEMINI
        # =========================

        response = model.generate_content(
            [
                prompt,
                pil_image
            ]
        )


        # =========================
        # RESPONSE
        # =========================

        if not response:
            raise HTTPException(
                status_code=500,
                detail="No response received from Gemini."
            )


        analysis_text = getattr(response, "text", None)


        if not analysis_text:
            raise HTTPException(
                status_code=500,
                detail="Gemini returned an empty response."
            )


        return {
            "success": True,
            "model": "gemini-2.5-flash",
            "analysis": analysis_text,
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
