# from TTS.api import TTS
# import numpy as np
# import io
# import soundfile as sf

# # Initialize TTS (CPU or GPU based on your preference)
# tts_model = TTS("tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

# def tts_to_bytes(text: str) -> bytes:
#     # Generate waveform directly (numpy array)
#     waveform = tts_model.tts(text)
#     # Save waveform to a buffer (as .wav)
#     buffer = io.BytesIO()
#     sf.write(buffer, waveform, 22050, format='WAV')
#     return buffer.getvalue()






import io
import soundfile as sf
from TTS.api import TTS
from typing import Optional, Dict, Any
from g2p_en import G2p

# Load TTS model
model_name = "tts_models/en/ljspeech/tacotron2-DDC_ph"
tts_model = TTS(model_name, progress_bar=False, gpu=False)
has_alignment = hasattr(tts_model, "tts_with_alignment")
g2p = G2p()

# Define phoneme-to-morph mapping
phoneme_to_morph = {
    "AA": "mouthOpen", "AE": "mouthSmile", "AH": "mouthOpen", "AO": "mouthOpen",
    "AW": "mouthOpen", "AY": "mouthSmile", "B": "mouthSmile", "CH": "mouthSmile",
    "D": "mouthSmile", "DH": "mouthOpen", "EH": "mouthOpen", "ER": "mouthOpen",
    "EY": "mouthSmile", "F": "mouthSmile", "G": "mouthOpen", "HH": "mouthOpen",
    "IH": "mouthOpen", "IY": "mouthSmile", "JH": "mouthSmile", "K": "mouthSmile",
    "L": "mouthSmile", "M": "mouthSmile", "N": "mouthSmile", "NG": "mouthSmile",
    "OW": "mouthOpen", "OY": "mouthOpen", "P": "mouthSmile", "R": "mouthSmile",
    "S": "mouthSmile", "SH": "mouthSmile", "T": "mouthSmile", "TH": "mouthOpen",
    "UH": "mouthOpen", "UW": "mouthOpen", "V": "mouthSmile", "W": "mouthSmile",
    "Y": "mouthSmile", "Z": "mouthSmile", "ZH": "mouthSmile"
}

def synthesize_with_phonemes(text: str) -> Optional[Dict[str, Any]]:
    try:
        print("🎙️ [TTS] Synthesizing text...")

        sample_rate = 22050
        if has_alignment:
            output = tts_model.tts_with_alignment(text, return_dict=True)
            waveform = output["wav"]
            phonemes = output["phoneme"]
            durations = output["phoneme_duration"]

            # Normalize durations to seconds
            total_duration = sum(durations)
            audio_duration = len(waveform) / sample_rate
            durations = [(d / total_duration) * audio_duration for d in durations]
        else:
            waveform = tts_model.tts(text)
            phonemes = g2p(text)

            # Remove non-alphabetic tokens
            phonemes = [p for p in phonemes if p.isalpha()]
            audio_duration = len(waveform) / sample_rate
            duration_per_phoneme = audio_duration / len(phonemes) if phonemes else 0.05
            durations = [duration_per_phoneme] * len(phonemes)

        # Encode audio as WAV
        buffer = io.BytesIO()
        sf.write(buffer, waveform, samplerate=sample_rate, format='WAV')
        buffer.seek(0)
        audio_bytes = buffer.read()

        # Quantize to 0.1 intervals
        timeline = []
        morph_targets = []
        step = 0.1
        num_steps = int(1.0 / step)  # 10 steps (0.0 to 0.9)
        valid_phonemes = [(p, phoneme_to_morph.get(p.upper())) for p in phonemes if phoneme_to_morph.get(p.upper())]

        for i, (p, morph) in enumerate(valid_phonemes):
            start = round(i * step, 1)
            end = round(min((i + 1) * step, 1.0), 1)

            timeline.append({"char": p, "start": start, "end": end})
            morph_targets.append({
                "morph": morph,
                "start": start,
                "end": end,
                "weight": 1.0
            })

        print(f"✅ Audio Duration: {round(audio_duration, 3)}s")
        print(f"🧠 Phonemes: {phonemes}")
        print(f"🎯 Morph targets: {len(morph_targets)} generated")

        return {
            "audio": audio_bytes,
            "phonemes": timeline,
            "morph_targets": morph_targets
        }

    except Exception as e:
        print(f"[TTS ERROR] {e}")
        return None
