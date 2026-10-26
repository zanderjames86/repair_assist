"""AI part 2 - Speech-to-text: transcribe the technician's voice note
with a pretrained Whisper model (faster-whisper, int8 on CPU)."""
import io

MODEL_SIZE = "base.en"   # ~75 MB; use "tiny.en" if the host runs out of memory


def load_model(size: str = MODEL_SIZE):
    from faster_whisper import WhisperModel
    return WhisperModel(size, device="cpu", compute_type="int8")


def transcribe(model, audio_bytes: bytes) -> str:
    segments, _ = model.transcribe(
        io.BytesIO(audio_bytes), language="en", beam_size=1, vad_filter=True,
        # Vocabulary hint: helps Whisper spell component references correctly
        initial_prompt="PCB repair note. Regulator U3, capacitor C12, connector J2, "
                       "5V rail, short to ground, corrosion, solder joint.")
    return " ".join(s.text.strip() for s in segments).strip()
