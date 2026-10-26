"""Audio decoding for Whisper works without the model (no network needed)."""
import io
import shutil
import wave

import numpy as np
import pytest

from speech_utils import decode_audio


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_decode_resamples_to_16k_mono():
    sr, secs = 44100, 1.0
    t = np.arange(int(sr * secs)) / sr
    tone = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    stereo = np.stack([tone, tone], 1)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr); w.writeframes(stereo.tobytes())
    audio = decode_audio(buf.getvalue())
    assert audio.dtype == np.float32
    assert abs(len(audio) - 16000) < 50
    assert 0.4 < np.abs(audio).max() <= 0.51
