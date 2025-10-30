import io
import base64
import numpy as np
import soundfile as sf
from flask import Flask, request, jsonify
import google.genai as genai
from google.genai.types import Audio, ResponseModalities

# === Your Gemini API key ===
API_KEY = "AIzaSyDNgpM6Q4qojqjrnXnQF1vimN6zSlYtn3w"   # <-- Replace this with your actual key
MODEL = "gemini-2.0-flash-exp"

# === Initialize client ===
client = genai.Client(api_key=API_KEY)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Gemini STS Bridge Server is running 🎙️",
        "usage": "POST raw WAV data to /upload"
    })

@app.route("/upload", methods=["POST"])
def upload():
    audio_bytes = request.data
    if not audio_bytes:
        return jsonify({"error": "No audio data received"}), 400

    try:
        # Read incoming WAV
        audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)  # convert to mono

        # Convert to float32 numpy array
        audio_array = np.array(audio_array, dtype=np.float32)

        # Send audio to Gemini (Speech-to-Speech)
        response = client.models.generate_content(
            model=MODEL,
            contents=[Audio(data=audio_array, mime_type="audio/wav")],
            config={"response_modalities": [ResponseModalities.AUDIO]}
        )

        if not hasattr(response, "audio_data") or response.audio_data is None:
            return jsonify({"error": "No audio returned from Gemini"}), 500

        # Convert returned audio to WAV + Base64
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, response.audio_data, 22050, format='WAV')
        wav_buffer.seek(0)
        wav_b64 = base64.b64encode(wav_buffer.read()).decode("utf-8")

        return jsonify({"status": "ok", "mime": "audio/wav", "wav_b64": wav_b64})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Running on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
