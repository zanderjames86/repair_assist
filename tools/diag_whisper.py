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
for ct in ["int8", "int8_float32", "float32"]:
    m = WhisperModel("base.en", device="cpu", compute_type=ct)
    for vad in (True, False):
        for prompt in (None, "PCB repair note."):
            t = time.time()
            out = [" ".join(s.text for s in m.transcribe(a, language="en", beam_size=1, vad_filter=vad,
                                                          initial_prompt=prompt)[0]).strip() for a in wavs]
            print(f"{ct:13s} vad={vad!s:5s} prompt={bool(prompt)!s:5s} {time.time()-t:4.1f}s  {out}")
