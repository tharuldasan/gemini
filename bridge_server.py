import os
import io
import base64
import soundfile as sf
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import numpy as np
import google.genai as genai
from google.genai.types import Audio, ResponseModalities

# ========== Load environment variables ==========
load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not found. Please set it in .env or server environment variables.")

# ========== Flask app setup ==========
app = Flask(__name__)

# ========== Gemini model setup ==========
MODEL = "gemini-2.0-flash-exp"  # Speech-to-speech capable model
client = genai.Client(api_key=API_KEY)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Welcome to Gemini Speech-to-Speech Bridge Server 🌐",
        "usage": "POST /upload with audio/wav file to get AI-generated response audio."
    })


# ========== Upload route ==========
@app.route("/upload", methods=["POST"])
def upload():
    """
    Receives WAV data -> Sends to Gemini STS model -> Returns generated speech audio as base64.
    """

    # Get audio data from POST body
    audio_bytes = request.data
    if not audio_bytes:
        return jsonify({"error": "No audio data received"}), 400

    try:
        # Decode the incoming WAV
        audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)  # Convert to mono if stereo

        # Convert audio data to float32 numpy
        audio_array = np.array(audio_array, dtype=np.float32)

        # Send to Gemini for speech-to-speech
        response = client.models.generate_content(
            model=MODEL,
            contents=[Audio(data=audio_array, mime_type="audio/wav")],
            config={"response_modalities": [ResponseModalities.AUDIO]}
        )

        # Extract response audio
        audio_data = response.audio_data
        if not audio_data:
            return jsonify({"error": "Model returned no audio"}), 500

        # Save to a temporary WAV buffer
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_data, 22050, format='WAV')
        wav_buffer.seek(0)

        # Encode WAV to base64 for transport
        wav_b64 = base64.b64encode(wav_buffer.read()).decode("utf-8")

        return jsonify({
            "status": "ok",
            "wav_b64": wav_b64,
            "mime": "audio/wav"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ========== Run local server ==========
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Bridge server running on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port)
