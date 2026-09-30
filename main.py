import os
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import google.generativeai as genai

app = FastAPI()

# Enable CORS for Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
API_KEY = os.getenv("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

@app.get("/")
def home():
    return {"status": "IndraMed AI Backend is Running"}

@app.post("/api/analyze-xray")
async def analyze_xray(image: UploadFile = File(...)):
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY missing in Render env vars.")

        # Read uploaded image file
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))

        # Initialize official Gemini 1.5 Flash model
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = "Analyze this medical X-ray image and provide concise clinical findings and recommendations."

        response = model.generate_content([prompt, pil_image])

        return {
            "success": True,
            "analysis": response.text
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        
