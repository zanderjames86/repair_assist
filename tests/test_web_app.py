"""The Streamlit app runs end to end without errors (headless AppTest)."""
from streamlit.testing.v1 import AppTest

from conftest import ROOT

CORROSION = "Green deposits on the pins, salt residue across the lower edge, moisture ingress."


def run_app():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def test_app_starts_and_has_three_tabs():
    at = run_app()
    assert len(at.tabs) == 3
    assert any("Model Evaluation" in t.label for t in at.tabs)


def test_ocr_autofills_serial():
    at = run_app()
    serial_box = next(s for s in at.selectbox if s.label.startswith("Serial number"))
    assert serial_box.value is not None, "OCR should auto-fill the serial for the first sample photo"


def test_full_bench_flow_and_save():
    at = run_app()
    at.radio[1].set_value("Type / example").run()
    note_pick = next(s for s in at.selectbox if s.label == "Example note")
    note_pick.set_value(CORROSION).run()
    assert not at.exception
    assert any(m.value == "Corrosion" for m in at.metric), [m.value for m in at.metric]
    at.button[0].click().run()
    assert not at.exception
    assert len(at.session_state.log) == 1
    entry = at.session_state.log[0]
    assert entry["fault"] == "Corrosion"
    assert entry["recommendation"] in {"Repairable", "BER (beyond economic repair)"}
