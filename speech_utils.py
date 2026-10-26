"""AI part 2 - Speech-to-text: transcribe the technician's voice note
with a pretrained Whisper model (faster-whisper, int8 on CPU)."""
import subprocess

import numpy as np

MODEL_SIZE = "base.en"   # ~75 MB; use "tiny.en" if the host runs out of memory


def load_model(size: str = MODEL_SIZE):
    from faster_whisper import WhisperModel
    return WhisperModel(size, device="cpu", compute_type="int8")


def decode_audio(audio_bytes: bytes) -> np.ndarray:
    """Any audio format -> 16 kHz mono float32, via ffmpeg (packages.txt).
    Done here rather than through faster-whisper's PyAV loader, which breaks
    with newer PyAV releases."""
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", "pipe:0",
         "-f", "s16le", "-ac", "1", "-ar", "16000", "pipe:1"],
        input=audio_bytes, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0


def transcribe(model, audio_bytes: bytes) -> str:
    segments, _ = model.transcribe(
        decode_audio(audio_bytes), language="en", beam_size=1, vad_filter=True,
        # Vocabulary hint: helps Whisper spell component references correctly
        initial_prompt="PCB repair note. Regulator U3, capacitor C12, connector J2, "
                       "5V rail, short to ground, corrosion, solder joint.")
    return " ".join(s.text.strip() for s in segments).strip()
