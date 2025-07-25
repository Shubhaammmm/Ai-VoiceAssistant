from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid
import numpy as np
import soundfile as sf

from whisper import load_model as load_whisper
from config import Config
from rag import retrieve_context
from tts import tts_model  # We'll add phoneme support soon
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Setup folders
AUDIO_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Load models
whisper_model = load_whisper(Config.WHISPER_MODEL)
llm = ChatOllama(model=Config.OLLAMA_MODEL)
memory = ConversationBufferMemory(
    ai_prefix="Heckx:",
    human_prefix="You:",
    return_messages=False,
    k=Config.CONVERSATION_HISTORY_LIMIT
)

# Define FastAPI app
app = FastAPI()
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

# Prompt
template = """
You are Heckx, a witty and helpful AI assistant created by bobo. You provide concise, accurate answers with a touch of humor.
Use the relevant context below to answer questions accurately and clearly.
Keep responses under 30 words unless asked for more detail.

Conversation history:
{history}

Relevant context:
{context}

User's input: {input}

Your response:
"""
PROMPT = PromptTemplate(input_variables=["history", "input", "context"], template=template)

@app.post("/speak")
async def speak(audio: UploadFile = File(...)):
    # Save uploaded file to disk
    raw_path = f"audio/user_{uuid.uuid4().hex}.wav"
    with open(raw_path, "wb") as f:
        f.write(await audio.read())

    # Load audio into numpy
    data, samplerate = sf.read(raw_path)
    audio_np = np.array(data, dtype=np.float32)

    # Transcribe
    result = whisper_model.transcribe(audio_np, fp16=False)
    user_text = result["text"].strip()

    # RAG
    context = retrieve_context(user_text)

    # Prompt the LLM
    history = memory.load_memory_variables({})["history"]
    full_prompt = PROMPT.format(history=history, input=user_text, context=context)
    response = llm.invoke(full_prompt)
    assistant_text = response.content if hasattr(response, "content") else str(response)
    memory.save_context({"input": user_text}, {"output": assistant_text})

    # Synthesize speech
    output_path = f"audio/response_{uuid.uuid4().hex}.wav"
    tts_model.tts_to_file(text=assistant_text, file_path=output_path)

    return JSONResponse({
        "text": user_text,
        "response_text": assistant_text,
        "audio_url": f"http://localhost:8000/{output_path}"
        # optional: "phonemes": [...]
    })
