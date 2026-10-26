"""Diagnostic: Whisper transcripts of synthesised notes under different settings."""
import json, subprocess, sys, time, platform
sys.path.insert(0, ".")
from faster_whisper import WhisperModel
import ctranslate2, numpy as np
from speech_utils import decode_audio

print("python", platform.python_version(), "ctranslate2", ctranslate2.__version__,
      "cpu", ctranslate2.get_supported_compute_types("cpu"))
cases = json.load(open("tests/speech_cases.json"))
wavs = []
for i, c in enumerate(cases[:2]):
    subprocess.run(["espeak-ng", "-v", "en-us", "-s", "140", "-w", f"/tmp/n{i}.wav", c["text"]], check=True)
    a = decode_audio(open(f"/tmp/n{i}.wav", "rb").read())
    print(f"note{i}: {len(a)/16000:.1f}s peak {np.abs(a).max():.2f}")
    wavs.append(a)
FULL = ("PCB repair note. Regulator U3, capacitor C12, connector J2, "
        "5V rail, short to ground, corrosion, solder joint.")
for ct in ["int8"]:
    m = WhisperModel("base.en", device="cpu", compute_type=ct)
    for vad in (True, False):
        for prompt in (None, "PCB repair note.", FULL,
                       "Electronics repair note about a circuit board, mentioning a regulator, capacitor, connector, short to ground or corrosion."):
            t = time.time()
            out = [" ".join(s.text for s in m.transcribe(a, language="en", beam_size=1, vad_filter=vad,
                                                          initial_prompt=prompt)[0]).strip() for a in wavs]
            print(f"{ct:13s} vad={vad!s:5s} prompt={(prompt or "-")[:20]!s:20s} {time.time()-t:4.1f}s  {out}")

from speech_utils import load_model, transcribe
m = load_model()
for i in range(2):
    print("speech_utils.transcribe:", transcribe(m, open(f"/tmp/n{i}.wav", "rb").read()))
