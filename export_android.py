"""
Export the trained scikit-learn models to plain JSON so the Android app can
run them fully offline, with no Python and no ML runtime on the device.

  fault_lr.json    TF-IDF vocabulary + IDF weights + Logistic Regression coefficients
  outcome_rf.json  Random Forest classifier trees (P(BER) at each leaf)
  hours_rf.json    Random Forest regressor trees (hours at each leaf)
  metrics.json     evaluation results shown on the phone's "Models" screen
  registry.json    board history lookup

It also writes parity.json: hundreds of inputs with scikit-learn's exact
outputs. The Android unit tests replay them and fail if the Kotlin
implementation differs by more than 1e-6.

Run after train_models.py:   python export_android.py
"""
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
ASSETS = ROOT / "android" / "app" / "src" / "main" / "assets" / "models"
TEST_RES = ROOT / "android" / "app" / "src" / "test" / "resources"
DEVICE_TEST_ASSETS = ROOT / "android" / "app" / "src" / "androidTest" / "assets"


def export_fault(pipe):
    tfidf, lr = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    vocab = {term: int(i) for term, i in tfidf.vocabulary_.items()}
    assert tfidf.token_pattern == r"(?u)\b\w\w+\b" and tfidf.lowercase
    assert tfidf.sublinear_tf and tfidf.norm == "l2" and tfidf.ngram_range == (1, 2)
    return {
        "classes": [str(c) for c in lr.classes_],
        "stop_words": sorted(tfidf.get_stop_words()),
        "ngram_max": 2,
        "vocabulary": vocab,
        "idf": [float(v) for v in tfidf.idf_],
        "coef": [[float(v) for v in row] for row in lr.coef_],
        "intercept": [float(v) for v in lr.intercept_],
    }


def export_encoder(pipe):
    prep = pipe.named_steps["prep"]
    ohe = prep.named_transformers_["cat"]
    cats = {name: [str(c) for c in cs] for name, cs in zip(["board_type", "fault_category"], ohe.categories_)}
    return {"categorical": cats, "numeric": ["age_years", "previous_repairs"]}


def export_forest(pipe, classifier):
    forest = pipe.named_steps["model"]
    trees = []
    for est in forest.estimators_:
        t = est.tree_
        if classifier:
            v = t.value[:, 0, :]
            leaf = (v[:, 1] / v.sum(axis=1)).tolist()
        else:
            leaf = t.value[:, 0, 0].tolist()
        trees.append({
            "feature": t.feature.tolist(),
            "threshold": [float(x) for x in t.threshold],
            "left": t.children_left.tolist(),
            "right": t.children_right.tolist(),
            "value": [float(x) for x in leaf],
        })
    return {"encoder": export_encoder(pipe), "trees": trees}


def parity_cases(fault, outcome, hours, records):
    rnd = random.Random(7)
    notes = list(records["note"].sample(300, random_state=3))
    notes += [
        "", "   ", "U3", "NO OUTPUT ON THE 5V RAIL!!!", "corrosion, corrosion; corrosion.",
        "Board is fine, nothing wrong", "Résistance R7 ouverte — café", "pins bent\nnew line\ttab",
        "No output on the 5V rail, visible corrosion near the connector, regulator U3 is hot.",
        "Green deposits on the pins, salt residue across the lower edge, moisture ingress.",
    ]
    proba = fault.predict_proba(notes)
    note_cases = [{"note": n, "proba": [float(p) for p in row]} for n, row in zip(notes, proba)]

    boards = sorted(records["board_type"].unique()) + ["UNKNOWN-BOARD"]
    faults = sorted(records["fault_category"].unique())
    rows = [{"board_type": rnd.choice(boards), "fault_category": rnd.choice(faults),
             "age_years": rnd.randint(0, 30), "previous_repairs": rnd.randint(0, 10)}
            for _ in range(300)]
    X = pd.DataFrame(rows)
    p_ber = outcome.predict_proba(X)[:, 1]
    h = hours.predict(X)
    tab_cases = [{**r, "p_ber": float(a), "hours": float(b)} for r, a, b in zip(rows, p_ber, h)]
    return {"notes": note_cases, "tabular": tab_cases}


ELECTRONICS_WORDS = """
zero one two three four five six seven eight nine ten twelve fifteen twenty thirty fifty hundred
volt volts amp amps milliamp ohm ohms watt watts hertz kilohm megohm microfarad
board circuit pcb track trace pad pads via vias pin pins lead leads leg legs joint joints solder flux
resistor resistors capacitor capacitors diode diodes transistor transistors mosfet fet regulator regulators
inductor coil choke transformer relay relays fuse fuses crystal oscillator chip ic ics microcontroller
processor memory connector connectors plug socket header cable harness ribbon wire wires switch button
led leds display screen sensor motor driver supply power ground rail rails input output signal
short shorted open broken cracked burnt burned hot warm cold dead bad faulty failed failing damaged
corrosion corroded rust rusty oxidation salt water moisture wet green white deposits residue
bent loose intermittent worn missing lifted replaced replace retest tested test testing working
voltage current resistance continuity measure measures measured reading low high zero noise ripple
no not on off and or the a an is are was near at to of in with from under over across top bottom edge
side left right front back near this that there it its it's after before then still now also very
""".split()


def speech_vocabulary(fault):
    """Word list that constrains the on-device recogniser (Vosk) to this domain:
    every word the fault classifier knows plus a general electronics lexicon."""
    words = {t for t in fault.named_steps["tfidf"].vocabulary_ if " " not in t and t.isalpha()}
    words |= set(ELECTRONICS_WORDS)
    return sorted(words)


def main():
    fault = joblib.load(MODELS / "fault_classifier.joblib")
    outcome = joblib.load(MODELS / "outcome_model.joblib")
    hours = joblib.load(MODELS / "hours_model.joblib")
    records = pd.read_csv(ROOT / "data" / "repair_records.csv")
    registry = pd.read_csv(ROOT / "data" / "board_registry.csv", dtype={"serial": str})

    ASSETS.mkdir(parents=True, exist_ok=True)
    TEST_RES.mkdir(parents=True, exist_ok=True)

    def dump(obj, path):
        path.write_text(json.dumps(obj, separators=(",", ":")))
        print(f"wrote {path.relative_to(ROOT)}  ({path.stat().st_size / 1024:.0f} KB)")

    dump(export_fault(fault), ASSETS / "fault_lr.json")
    dump(export_forest(outcome, classifier=True), ASSETS / "outcome_rf.json")
    dump(export_forest(hours, classifier=False), ASSETS / "hours_rf.json")
    dump(json.loads((MODELS / "metrics.json").read_text()), ASSETS / "metrics.json")
    dump(registry.to_dict(orient="records"), ASSETS / "registry.json")
    dump(speech_vocabulary(fault), ASSETS / "speech_vocab.json")
    parity = parity_cases(fault, outcome, hours, records)
    dump(parity, TEST_RES / "parity.json")
    DEVICE_TEST_ASSETS.mkdir(parents=True, exist_ok=True)
    dump(parity, DEVICE_TEST_ASSETS / "parity.json")    # same check, run on the emulator


if __name__ == "__main__":
    main()
