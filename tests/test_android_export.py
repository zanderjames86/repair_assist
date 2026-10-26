"""The JSON exported for the phone reproduces scikit-learn exactly.
(The Android JVM unit tests check the Kotlin/Java side against parity.json.)"""
import json
import math
import re

import joblib
import numpy as np
import pandas as pd

from conftest import ROOT

ASSETS = ROOT / "android" / "app" / "src" / "main" / "assets" / "models"


def tfidf_vec(m, note):
    sw = set(m["stop_words"])
    toks = [t for t in re.findall(r"(?u)\b\w\w+\b", note.lower()) if t not in sw]
    terms = toks + [toks[i] + " " + toks[i + 1] for i in range(len(toks) - 1)]
    x = np.zeros(len(m["idf"]))
    counts = {}
    for t in terms:
        if t in m["vocabulary"]:
            counts[m["vocabulary"][t]] = counts.get(m["vocabulary"][t], 0) + 1
    for k, c in counts.items():
        x[k] = (1 + math.log(c)) * m["idf"][k]
    n = np.linalg.norm(x)
    return x / n if n else x


def test_fault_export_matches_sklearn():
    m = json.loads((ASSETS / "fault_lr.json").read_text())
    pipe = joblib.load(ROOT / "models" / "fault_classifier.joblib")
    notes = list(pd.read_csv(ROOT / "data" / "repair_records.csv")["note"]) + ["", "U3 HOT!!"]
    W, b = np.array(m["coef"]), np.array(m["intercept"])
    ref = pipe.predict_proba(notes)
    for note, r in zip(notes, ref):
        z = W @ tfidf_vec(m, note) + b
        p = np.exp(z - z.max()); p /= p.sum()
        assert np.allclose(p, r, atol=1e-9)


def forest_predict(f, row):
    enc = f["encoder"]["categorical"]
    x = [1.0 if row["board_type"] == c else 0.0 for c in enc["board_type"]]
    x += [1.0 if row["fault_category"] == c else 0.0 for c in enc["fault_category"]]
    x += [row["age_years"], row["previous_repairs"]]
    total = 0.0
    for t in f["trees"]:
        n = 0
        while t["left"][n] != -1:
            n = t["left"][n] if np.float32(x[t["feature"][n]]) <= t["threshold"][n] else t["right"][n]
        total += t["value"][n]
    return total / len(f["trees"])


def test_forest_exports_match_sklearn():
    outcome = joblib.load(ROOT / "models" / "outcome_model.joblib")
    hours = joblib.load(ROOT / "models" / "hours_model.joblib")
    fo = json.loads((ASSETS / "outcome_rf.json").read_text())
    fh = json.loads((ASSETS / "hours_rf.json").read_text())
    df = pd.read_csv(ROOT / "data" / "repair_records.csv").head(120)
    X = df[["board_type", "fault_category", "age_years", "previous_repairs"]]
    po, ph = outcome.predict_proba(X)[:, 1], hours.predict(X)
    for (_, row), a, b in zip(X.iterrows(), po, ph):
        assert abs(forest_predict(fo, row) - a) < 1e-9
        assert abs(forest_predict(fh, row) - b) < 1e-9


def test_assets_are_in_sync_with_models():
    exported = json.loads((ASSETS / "metrics.json").read_text())
    current = json.loads((ROOT / "models" / "metrics.json").read_text())
    assert exported == current, "run export_android.py after retraining"
    reg = json.loads((ASSETS / "registry.json").read_text())
    csv = pd.read_csv(ROOT / "data" / "board_registry.csv", dtype={"serial": str})
    assert [r["serial"] for r in reg] == list(csv["serial"])
