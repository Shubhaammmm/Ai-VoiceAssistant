from TTS.api import TTS
import soundfile as sf
import sounddevice as sd
import tempfile
import os

# Initialize Coqui TTS model (CPU only)
tts_model = TTS("tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

def speak_text(text: str):
    try:
        # Use a temporary file to store output
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            output_path = tmp_file.name
            tts_model.tts_to_file(text=text, file_path=output_path)

        # Load and play audio
        data, samplerate = sf.read(output_path)
        sd.play(data, samplerate)
        sd.wait()

        # Clean up the temp file
        os.remove(output_path)

    except Exception as e:
        print(f"[TTS Error] Failed to generate or play speech: {e}")

