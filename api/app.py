import io
import json
import traceback
import subprocess
import numpy as np
import base64
import soundfile as sf
import re

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# AI modules
from whisper import load_model as load_whisper
from config import Config
from rag import retrieve_context
from tts import synthesize_with_phonemes
from typing import Optional, Dict, Any

from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# === FastAPI Setup ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load Models ===
print("📦 [INIT] Loading Whisper model...")
whisper_model = load_whisper("base")

print("🧠 [INIT] Loading LLM model...")
llm = ChatOllama(model=Config.OLLAMA_MODEL)
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

prompt_template = PromptTemplate(
    input_variables=["history", "input", "context"],
    template="""{history}
User: {input}
Context: {context}
Assistant:"""
)

# === Audio Helpers ===
def convert_to_wav(audio_bytes: bytes, mime_type: str) -> bytes:
    print("🔄 [AUDIO] Converting audio to WAV format...")
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", "pipe:0", "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1"],
            input=audio_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        print("✅ [AUDIO] Audio conversion successful.")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("❌ [FFMPEG] Error converting audio:", e.stderr.decode())
        raise HTTPException(status_code=400, detail="Audio conversion failed")

def clean_text(text: str) -> str:
    return re.sub(r"[^\x00-\x7F]+", "", text)

# === Main Endpoint ===
@app.post("/speak")
async def speak(request: Request):
    try:
        print("📥 [API] Incoming /speak request")
        body = await request.json()
        base64_audio = body.get("audio")
        if not base64_audio:
            raise HTTPException(status_code=400, detail="Missing base64 audio")

        print("🎧 [API] Decoding base64 audio...")
        audio_bytes = base64.b64decode(base64_audio.split(",")[-1])

        # Convert to WAV
        audio_wav = convert_to_wav(audio_bytes, "audio/wav")

        # Decode to NumPy
        print("📊 [AUDIO] Decoding WAV to numpy array...")
        try:
            data, _ = sf.read(io.BytesIO(audio_wav))
            audio_np = np.array(data, dtype=np.float32)
        except Exception as e:
            print("❌ [WAV] Decode error:", str(e))
            raise HTTPException(status_code=400, detail="Invalid audio")

        # Transcribe
        print("🔍 [WHISPER] Running transcription...")
        whisper_result = whisper_model.transcribe(audio_np, fp16=False, language='en')
        user_text = whisper_result.get("text", "").strip() or "Hello"
        print(f"📝 [WHISPER] Transcription: {user_text}")

        # Retrieve context
        print("📚 [RAG] Retrieving context...")
        context = retrieve_context(user_text)

        # Prompt LLM
        history = memory.load_memory_variables({}).get("history", "")
        full_prompt = prompt_template.format(history=history, input=user_text, context=context)

        print("💡 [LLM] Generating assistant response...")
        response = llm.invoke(full_prompt)
        assistant_text = getattr(response, "content", str(response)).strip()
        print(f"💬 [LLM] Assistant Response: {assistant_text}")

        memory.save_context({"input": user_text}, {"output": assistant_text})
        assistant_text = clean_text(assistant_text)

        # Text-to-Speech
        print("🔊 [TTS] Synthesizing voice response...")
        tts_result = synthesize_with_phonemes(assistant_text)
        if not tts_result:
            raise HTTPException(status_code=500, detail="TTS failed")

        tts_audio_bytes = tts_result["audio"]
        phonemes = tts_result["phonemes"]
        morph_targets = tts_result.get("morph_targets", [])

        print("📤 [API] Encoding audio to base64 for response...")
        base64_audio_response = base64.b64encode(tts_audio_bytes).decode("utf-8")

        print("✅ [API] Sending response")
        return {
            "transcription": user_text,
            "response_text": assistant_text,
            "audio_base64": f"data:audio/wav;base64,{base64_audio_response}",
            "phonemes": phonemes,
            "morph_targets": morph_targets
        }

    except HTTPException as http_err:
        print(f"❗ [HTTPException] {http_err.detail}")
        return JSONResponse(status_code=http_err.status_code, content={"error": http_err.detail})

    except Exception as e:
        print("❌ [EXCEPTION] Internal Server Error")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Internal error: {str(e)}"})
