"""Trained ML models reach their quality targets and behave sensibly."""
import json

import joblib
import pandas as pd
import pytest

from conftest import ROOT

MODELS = ROOT / "models"


@pytest.fixture(scope="module")
def metrics():
    return json.loads((MODELS / "metrics.json").read_text())


@pytest.fixture(scope="module")
def models():
    return (joblib.load(MODELS / "fault_classifier.joblib"),
            joblib.load(MODELS / "outcome_model.joblib"),
            joblib.load(MODELS / "hours_model.joblib"))


def test_fault_classifier_quality(metrics):
    fc = metrics["fault_classifier"]
    chosen = next(c for c in fc["comparison"] if c["model"] == fc["chosen"])
    assert chosen["cv_f1_mean"] >= 0.90
    assert chosen["test_f1_macro"] >= 0.90
    # the chosen model is the best (or tied) on cross-validation
    assert chosen["cv_f1_mean"] >= max(c["cv_f1_mean"] for c in fc["comparison"]) - 1e-9
    cm = fc["confusion_matrix"]
    assert sum(map(sum, cm)) == fc["test_size"]


def test_outcome_model_beats_baselines(metrics):
    o, h = metrics["outcome"], metrics["hours"]
    assert o["test_roc_auc"] >= 0.75
    assert o["test_accuracy"] >= o["baseline_accuracy"]
    assert h["test_mae"] < 0.6 * h["baseline_mae"]
    assert h["test_r2"] >= 0.8
    assert abs(sum(o["feature_importance"].values()) - 1) < 0.01


@pytest.mark.parametrize("note,expected", [
    ("Dead short on the 3.3V rail, capacitor C12 shorted to ground.", "short_circuit"),
    ("Cracked solder joint on R7, no continuity on the trace.", "open_circuit"),
    ("Op amp U5 output stuck high, relay coil open.", "component_failure"),
    ("Green deposits on the pins, salt residue across the lower edge.", "corrosion"),
    ("Connector J2 pins bent, loose contact at the plug.", "connector_fault"),
    ("Fuse blown on the input, 3.3V regulator output low.", "power_supply"),
])
def test_fault_examples(models, note, expected):
    assert models[0].predict([note])[0] == expected


def test_ber_risk_rises_with_age_and_corrosion(models):
    _, outcome, hours = models
    def p(age, fault, prev=0):
        X = pd.DataFrame([{"board_type": "RF-COMM", "fault_category": fault,
                           "age_years": age, "previous_repairs": prev}])
        return outcome.predict_proba(X)[0, 1], hours.predict(X)[0]
    assert p(20, "corrosion")[0] > p(2, "corrosion")[0]
    assert p(10, "corrosion")[0] > p(10, "connector_fault")[0]
    assert p(10, "corrosion")[1] > p(10, "connector_fault")[1]


def test_retraining_reproduces_metrics(tmp_path, monkeypatch, metrics):
    import train_models
    monkeypatch.setattr(train_models, "MODELS", tmp_path)
    train_models.main()
    fresh = json.loads((tmp_path / "metrics.json").read_text())
    assert fresh["fault_classifier"]["comparison"] == metrics["fault_classifier"]["comparison"]
    assert fresh["outcome"]["test_roc_auc"] == metrics["outcome"]["test_roc_auc"]
