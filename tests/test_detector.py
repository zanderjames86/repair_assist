"""Part detector (YOLOv8n ONNX): loads, decodes correctly, finds parts on unseen boards."""
import json

import numpy as np
import pytest
from PIL import Image

from conftest import ROOT
from detector_utils import PartDetector, decode

META = ROOT / "models" / "parts.json"
needs_model = pytest.mark.skipif(not PartDetector.available(), reason="detector not trained yet")


def trained():
    return PartDetector.available() and not json.loads(META.read_text())["training"].get("smoke_test")


def test_decode_maps_letterbox_back_to_original():
    out = np.zeros((4 + 2, 1), np.float32)
    out[:, 0] = [320, 320, 100, 50, 0.9, 0.1]     # 1280x640 image at 640: scale .5, pad y 160
    d = decode(out, 0.5, 0, 160, 1280, 640)
    assert len(d) == 1 and d[0]["class_id"] == 0
    assert np.allclose(d[0]["box"], (540, 270, 740, 370))


def test_decode_nms_per_class():
    out = np.zeros((6, 3), np.float32)
    out[:, 0] = [100, 100, 40, 40, 0.9, 0]
    out[:, 1] = [102, 101, 40, 40, 0.8, 0]      # same class, overlapping -> suppressed
    out[:, 2] = [102, 101, 40, 40, 0, 0.7]      # other class -> kept
    d = decode(out, 1.0, 0, 0, 640, 640)
    assert [x["class_id"] for x in d] == [0, 1]


def test_android_fixture_is_current():
    """The Java parity fixture was produced by this decoder."""
    cases = json.loads((ROOT / "android/app/src/test/resources/detector_parity.json").read_text())
    for c in cases:
        out = np.array(c["out"], np.float32).reshape(4 + c["nc"], c["n"])
        s = min(c["size"] / c["w"], c["size"] / c["h"])
        px, py = (c["size"] - round(c["w"] * s)) // 2, (c["size"] - round(c["h"] * s)) // 2
        assert len(decode(out, s, px, py, c["w"], c["h"])) == len(c["dets"])


@needs_model
def test_model_files_match_between_web_and_android():
    a = (ROOT / "models" / "parts.onnx").read_bytes()
    b = (ROOT / "android/app/src/main/assets/models/parts.onnx").read_bytes()
    assert a == b
    meta = json.loads(META.read_text())
    assert meta["names"] == ["Capacitor", "Connector", "IC", "Inductor", "Resistor", "Switch", "Transistor/Diode"]


@pytest.mark.skipif(not trained(), reason="needs the fully trained detector")
def test_detector_quality_on_unseen_test_boards():
    meta = json.loads(META.read_text())
    m, per = meta["metrics"], meta["per_class_ap50"]
    # Sanity floor, not a target: tiny SMD passives on dense boards are hard for a
    # nano model, so check overall quality plus the large, distinctive parts.
    assert m["map50"] >= 0.25, m
    assert per["IC"] >= 0.45 and per["Connector"] >= 0.40, per


@pytest.mark.skipif(not trained(), reason="needs the fully trained detector")
def test_finds_parts_on_demo_boards():
    det = PartDetector()
    found = []
    for p in sorted((ROOT / "sample_images").glob("*.jpg"))[:10]:
        parts = det.detect(Image.open(p))
        assert all(0.3 < x["score"] <= 1 for x in parts)
        found.append(len(parts))
    assert sum(n > 0 for n in found) >= 9, found
