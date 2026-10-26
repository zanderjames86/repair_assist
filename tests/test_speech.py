"""Speech-to-text end to end: synthesised voice notes -> Whisper -> fault classifier.
Needs espeak-ng and network access to download the Whisper model (both available in CI)."""
import json
import re
import shutil
import subprocess

import joblib
import pytest

from conftest import ROOT

CASES = json.loads((ROOT / "tests" / "speech_cases.json").read_text())
pytestmark = pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="espeak-ng not installed")


@pytest.fixture(scope="module")
def whisper():
    try:
        from speech_utils import load_model
        return load_model()
    except Exception as e:      # no network to fetch the model
        pytest.skip(f"Whisper model unavailable: {e}")


@pytest.fixture(scope="module")
def transcripts(whisper, tmp_path_factory):
    from speech_utils import transcribe
    out = []
    d = tmp_path_factory.mktemp("speech")
    for i, c in enumerate(CASES):
        wav = d / f"note{i}.wav"
        subprocess.run(["espeak-ng", "-v", "en-us", "-s", "140", "-w", str(wav), c["text"]], check=True)
        out.append(transcribe(whisper, wav.read_bytes()))
    print("\n".join(out))
    return out


def test_keywords_recognised(transcripts):
    hits = total = 0
    for c, t in zip(CASES, transcripts):
        words = set(re.findall(r"\w+", t.lower()))
        hits += sum(k in words for k in c["keywords"])
        total += len(c["keywords"])
    assert hits / total >= 0.8, f"keyword recall {hits}/{total}"


def test_spoken_notes_classified_correctly(transcripts):
    clf = joblib.load(ROOT / "models" / "fault_classifier.joblib")
    pred = clf.predict(transcripts)
    correct = sum(p == c["fault"] for p, c in zip(pred, CASES))
    assert correct >= 3, list(zip(pred, [c["fault"] for c in CASES]))
