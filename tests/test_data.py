"""The synthetic training data is well-formed and reproducible."""
import hashlib

import pandas as pd

from conftest import ROOT
import generate_data

RECORDS = ROOT / "data" / "repair_records.csv"
REGISTRY = ROOT / "data" / "board_registry.csv"


def test_records_shape_and_columns():
    df = pd.read_csv(RECORDS)
    assert len(df) == 600
    assert list(df.columns) == ["job_id", "board_type", "age_years", "previous_repairs", "note",
                                "fault_category", "outcome", "repair_hours"]
    assert df.notna().all().all()
    assert df["job_id"].is_unique


def test_value_ranges():
    df = pd.read_csv(RECORDS)
    assert set(df["fault_category"]) == set(generate_data.FAULTS)
    assert set(df["outcome"]) == {"Repairable", "BER"}
    assert df["age_years"].between(1, 25).all()
    assert df["previous_repairs"].between(0, 8).all()
    assert df["repair_hours"].between(0.5, 12).all()
    # every fault class has a reasonable share (no class < 10%)
    assert df["fault_category"].value_counts(normalize=True).min() > 0.10
    assert 0.15 < (df["outcome"] == "BER").mean() < 0.45


def test_generation_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_data, "DATA_DIR", tmp_path)
    import random, numpy as np
    random.seed(generate_data.SEED); np.random.seed(generate_data.SEED)
    df = generate_data.make_records()
    df.to_csv(tmp_path / "r.csv", index=False)
    a = hashlib.sha256((tmp_path / "r.csv").read_bytes()).hexdigest()
    b = hashlib.sha256(RECORDS.read_bytes()).hexdigest()
    assert a == b, "repair_records.csv does not match a fresh generation with the fixed seed"


def test_notes_look_like_technician_notes():
    df = pd.read_csv(RECORDS)
    assert df["note"].str.len().between(15, 400).all()
    assert df["note"].str.endswith(".").all()


def test_registry():
    reg = pd.read_csv(REGISTRY, dtype={"serial": str})
    assert reg["serial"].is_unique
    assert reg["serial"].str.fullmatch(r"\d{4}").all()
    assert set(reg["board_type"]) <= set(generate_data.BOARD_TYPES)
    assert set(reg["last_fault"]) <= set(generate_data.FAULTS)


def test_demo_photos_exist_and_match_android_copies():
    reg = pd.read_csv(REGISTRY, dtype={"serial": str})
    assert len(reg) == 50
    assert reg["image"].is_unique and reg["source"].is_unique
    for name in reg["image"]:
        web = ROOT / "sample_images" / name
        app = ROOT / "android/app/src/main/assets/demo" / name
        assert web.exists() and app.exists(), name
        assert web.read_bytes() == app.read_bytes(), name
