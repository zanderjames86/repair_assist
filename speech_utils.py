"""AI part 2 - Speech-to-text: transcribe the technician's voice note
with a pretrained Whisper model (faster-whisper, int8 on CPU)."""
import re
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


def _run(model, audio, prompt):
    segments, _ = model.transcribe(audio, language="en", beam_size=1, vad_filter=True, initial_prompt=prompt)
    return " ".join(s.text.strip() for s in segments).strip()


def transcribe(model, audio_bytes: bytes) -> str:
    audio = decode_audio(audio_bytes)
    # A short, plain prompt nudges the vocabulary. Long prompts full of part
    # references (U3, C12, 5V) made Whisper emit "!!!!" in testing, so if the
    # result has no words at all, retry without any prompt.
    text = _run(model, audio, "Electronics repair note about a circuit board.")
    if not re.search(r"[A-Za-z]{2,}", text):
        text = _run(model, audio, None)
    return text
