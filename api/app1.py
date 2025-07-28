import io
import traceback
import subprocess
import numpy as np
import base64
import soundfile as sf

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Your actual modules
from whisper import load_model as load_whisper
from config import Config
from rag import retrieve_context
from tts import tts_model
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

app = FastAPI()

# Enable CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models globally
print("📦 Loading Whisper model...")
whisper_model = load_whisper("base")

print("🧠 Loading LLM model...")
llm = ChatOllama(model=Config.OLLAMA_MODEL)

memory = ConversationBufferMemory(memory_key="history", return_messages=True)

PROMPT = PromptTemplate(
    input_variables=["history", "input", "context"],
    template="""
{history}
User: {input}
Context: {context}
Assistant:"""
)

def convert_to_wav(audio_bytes: bytes, input_mime: str) -> bytes:
    """Convert audio to WAV format using ffmpeg in memory."""
    print(f"🔄 Converting audio to WAV from {input_mime}...")
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", "pipe:0", "-f", "wav", "pipe:1"],
            input=audio_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        print("✅ Audio converted to WAV successfully.")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("❌ ffmpeg error:", e.stderr.decode())
        raise HTTPException(status_code=400, detail="Failed to convert audio to WAV format.")

@app.post("/speak")
async def speak(audio: UploadFile = File(...)):
    try:
        if not audio.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Expected audio.")

        audio_bytes = await audio.read()
        print(f"🔹 Received file: {audio.filename or 'blob'}")
        print(f"🔹 Content-Type: {audio.content_type}")
        print(f"🔸 Received audio size: {len(audio_bytes)} bytes")

        # Convert to WAV if not already
        if audio.content_type != "audio/wav":
            audio_wav = convert_to_wav(audio_bytes, audio.content_type)
        else:
            audio_wav = audio_bytes

        # Read and decode WAV
        try:
            data, samplerate = sf.read(io.BytesIO(audio_wav))
            print(f"🔍 WAV decoded — sample rate: {samplerate}, shape: {np.array(data).shape}")
        except RuntimeError as e:
            print("❌ soundfile.read() failed:", e)
            raise HTTPException(status_code=400, detail="Failed to read audio file. Possibly unsupported format.")

        audio_np = np.array(data, dtype=np.float32)

        # Transcribe audio
        print("🗣️ Transcribing audio...")
        result = whisper_model.transcribe(audio_np, fp16=False)
        user_text = result.get("text", "").strip()
        print(f"📝 Transcription result: {user_text}")

        if not user_text:
            raise HTTPException(status_code=400, detail="No transcribable speech detected.")

        # Retrieve context and call LLM
        print("📚 Retrieving context...")
        context = retrieve_context(user_text)

        history = memory.load_memory_variables({}).get("history", "")
        full_prompt = PROMPT.format(history=history, input=user_text, context=context)

        print("🤖 Invoking LLM...")
        response = llm.invoke(full_prompt)
        assistant_text = getattr(response, "content", str(response)).strip()
        print(f"💬 LLM response: {assistant_text}")

        memory.save_context({"input": user_text}, {"output": assistant_text})

        # Generate TTS
        print("🎤 Synthesizing speech...")
        tts_audio_bytes = tts_model.tts(assistant_text)
        if not tts_audio_bytes:
            raise HTTPException(status_code=500, detail="TTS failed to generate audio.")

        audio_base64 = base64.b64encode(tts_audio_bytes).decode("utf-8")
        print("📦 TTS audio generated and encoded.")

        return JSONResponse(status_code=200, content={
            "text": user_text,
            "response_text": assistant_text,
            "audio_base64": audio_base64
        })

    except HTTPException as http_err:
        print(f"⚠️ HTTP error: {http_err.detail}")
        return JSONResponse(status_code=http_err.status_code, content={"error": http_err.detail})

    except Exception as e:
        print("🔥 Unhandled server error:")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Internal server error: {str(e)}"})
