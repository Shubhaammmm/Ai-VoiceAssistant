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

# Phoneme → morph mapping with weights (clamp applied later)
phoneme_to_morph_weight = {
    "AA": ("mouthOpen", 1.0), "AE": ("mouthOpen", 0.8), "AH": ("mouthOpen", 0.7),
    "AO": ("mouthOpen", 0.9), "AW": ("mouthOpen", 1.0), "AY": ("mouthOpen", 0.8),
    "B": ("mouthSmile", 1.0), "CH": ("mouthSmile", 0.8), "D": ("mouthSmile", 0.8),
    "DH": ("mouthSmile", 0.7), "EH": ("mouthOpen", 0.6), "ER": ("mouthSmile", 0.7),
    "EY": ("mouthOpen", 0.7), "F": ("mouthSmile", 0.7), "G": ("mouthSmile", 0.8),
    "HH": ("mouthOpen", 0.5), "IH": ("mouthOpen", 0.5), "IY": ("mouthOpen", 0.6),
    "JH": ("mouthSmile", 0.8), "K": ("mouthSmile", 0.7), "L": ("mouthSmile", 0.5),
    "M": ("mouthSmile", 0.9), "N": ("mouthSmile", 0.7), "NG": ("mouthSmile", 0.6),
    "OW": ("mouthOpen", 1.0), "OY": ("mouthOpen", 0.9), "P": ("mouthSmile", 0.9),
    "R": ("mouthSmile", 0.6), "S": ("mouthSmile", 0.6), "SH": ("mouthSmile", 0.7),
    "T": ("mouthSmile", 0.7), "TH": ("mouthSmile", 0.6), "UH": ("mouthOpen", 0.6),
    "UW": ("mouthOpen", 0.8), "V": ("mouthSmile", 0.8), "W": ("mouthSmile", 0.5),
    "Y": ("mouthSmile", 0.5), "Z": ("mouthSmile", 0.7), "ZH": ("mouthSmile", 0.7)
}

# Maximum morph weight allowed by your avatar
MAX_MORPH_WEIGHT = 0.80

def synthesize_with_phonemes(text: str) -> Optional[Dict[str, Any]]:
    try:
        print("🎙️ [TTS] Synthesizing text...")

        sample_rate = 22050
        if has_alignment:
            output = tts_model.tts_with_alignment(text, return_dict=True)
            waveform = output["wav"]
            phonemes = output["phoneme"]
            durations = output["phoneme_duration"]

            total_duration = sum(durations)
            audio_duration = len(waveform) / sample_rate
            durations = [(d / total_duration) * audio_duration for d in durations]
        else:
            waveform = tts_model.tts(text)
            phonemes = g2p(text)
            phonemes = [p for p in phonemes if p.isalpha()]
            audio_duration = len(waveform) / sample_rate
            duration_per_phoneme = audio_duration / len(phonemes) if phonemes else 0.05
            durations = [duration_per_phoneme] * len(phonemes)

        # Encode audio
        buffer = io.BytesIO()
        sf.write(buffer, waveform, samplerate=sample_rate, format='WAV')
        buffer.seek(0)
        audio_bytes = buffer.read()

        # Build output
        timeline = []
        morph_targets = []
        current_time = 0.0

        for i, p in enumerate(phonemes):
            duration = durations[i]
            start = round(current_time, 3)
            end = round(current_time + duration, 3)
            current_time = end

            morph, weight = phoneme_to_morph_weight.get(p.upper(), ("mouthSmile", 0.4))

            # Clamp weight to [0.0, 0.80]
            weight = max(0.0, min(weight, MAX_MORPH_WEIGHT))

            if not morph or weight <= 0 or start >= end:
                continue

            timeline.append({
                "char": p,
                "start": start,
                "end": end
            })
            morph_targets.append({
                "morph": morph,
                "start": start,
                "end": end,
                "weight": round(weight, 2)
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
