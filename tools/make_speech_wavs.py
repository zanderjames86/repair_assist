"""Synthesise the test voice notes (tests/speech_cases.json) as 16 kHz mono WAVs
for the Android on-device speech test. Needs espeak-ng and ffmpeg."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

out = Path(sys.argv[1] if len(sys.argv) > 1 else "android/app/src/androidTest/assets/speech")
out.mkdir(parents=True, exist_ok=True)
shutil.copy("tests/speech_cases.json", out / "cases.json")
for i, c in enumerate(json.loads(Path("tests/speech_cases.json").read_text())):
    raw = out / f"raw{i}.wav"
    subprocess.run(["espeak-ng", "-v", "en-us", "-s", "140", "-w", str(raw), c["text"]],
                   check=True, stdin=subprocess.DEVNULL, timeout=60)
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(raw), "-ar", "16000", "-ac", "1",
                    "-c:a", "pcm_s16le", "-map_metadata", "-1", "-fflags", "+bitexact", str(out / f"note{i}.wav")],
                   check=True, stdin=subprocess.DEVNULL, timeout=60)
    raw.unlink()
    print("wrote", out / f"note{i}.wav")
