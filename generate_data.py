"""
Generate a SYNTHETIC dataset of PCB repair records.

Everything here is invented and modelled on general electronics-repair
patterns. No real facility, equipment or repair data is used.

Outputs:
  data/repair_records.csv  - 600 labelled repair jobs (training data)
  data/board_registry.csv  - board history lookup used by the app
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = Path(__file__).parent / "data"

# Generic commercial-style board families (deliberately non-specific)
BOARD_TYPES = {
    "PSU-CTRL":   {"complexity": 1.0, "ber_bias": 0.0},
    "MCU-MAIN":   {"complexity": 1.4, "ber_bias": 0.3},
    "SENSOR-IF":  {"complexity": 1.1, "ber_bias": 0.1},
    "RF-COMM":    {"complexity": 1.8, "ber_bias": 0.7},
    "MOTOR-DRV":  {"complexity": 1.2, "ber_bias": 0.2},
    "DISPLAY-DRV": {"complexity": 1.3, "ber_bias": 0.2},
}

FAULTS = ["short_circuit", "open_circuit", "component_failure",
          "corrosion", "connector_fault", "power_supply"]

REFS = {
    "ic": ["U1", "U2", "U3", "U5", "U7", "U12"],
    "cap": ["C4", "C9", "C12", "C21", "C33"],
    "res": ["R7", "R15", "R22", "R40"],
    "fet": ["Q1", "Q2", "Q5"],
    "conn": ["J1", "J2", "J4", "P3"],
    "diode": ["D2", "D6", "D9"],
}

def ref(kind):
    return random.choice(REFS[kind])

# Phrase banks per fault category. Each lambda returns a fresh phrase.
PHRASES = {
    "short_circuit": [
        lambda: f"{ref('cap')} shorted to ground",
        lambda: "dead short on the 5V rail",
        lambda: "supply goes into current limit at power up",
        lambda: f"{ref('fet')} drain shorted to source",
        lambda: "solder bridge between adjacent pins",
        lambda: "near zero ohms between VCC and ground",
        lambda: f"{ref('ic')} running hot, internal short suspected",
        lambda: "burn mark under the shorted capacitor",
    ],
    "open_circuit": [
        lambda: "open track near the mounting hole",
        lambda: f"cracked solder joint on {ref('res')}",
        lambda: "no continuity on the trace to the connector",
        lambda: f"{ref('res')} measures open",
        lambda: "lifted pad, track broken",
        lambda: "via has no continuity between layers",
        lambda: "intermittent signal, hairline crack in track",
        lambda: "dry joint found under magnification",
    ],
    "component_failure": [
        lambda: f"{ref('ic')} failed, no output on the pin",
        lambda: f"electrolytic {ref('cap')} bulging, high ESR",
        lambda: f"{ref('diode')} failed open on diode test",
        lambda: "crystal oscillator not starting",
        lambda: f"op amp {ref('ic')} output stuck high",
        lambda: f"{ref('fet')} gate damaged, not switching",
        lambda: "relay coil open, relay not clicking",
        lambda: "optocoupler not responding",
    ],
    "corrosion": [
        lambda: "visible corrosion near the connector",
        lambda: "green deposits on the pins",
        lambda: "salt residue across the lower edge",
        lambda: "moisture ingress, tracks eaten away",
        lambda: "white crusty oxidation around the vias",
        lambda: "conformal coating peeled, corrosion underneath",
        lambda: f"corroded legs on {ref('ic')}",
        lambda: "board shows water damage and rust on the shield",
    ],
    "connector_fault": [
        lambda: f"connector {ref('conn')} pins bent",
        lambda: f"loose contact in {ref('conn')}",
        lambda: "cable harness intermittent at the plug",
        lambda: "connector housing cracked",
        lambda: f"pin 3 of {ref('conn')} pushed back",
        lambda: "header pins worn, poor mating",
        lambda: "ribbon cable socket latch broken",
        lambda: "contact resistance high on the plug",
    ],
    "power_supply": [
        lambda: "no output on the 5V rail",
        lambda: "3.3V regulator output low",
        lambda: f"regulator {ref('ic')} is hot, output drooping",
        lambda: "fuse blown on the input",
        lambda: "ripple on the supply too high",
        lambda: "switching regulator not starting up",
        lambda: "input voltage present but no rail out",
        lambda: "power LED off, supply dead",
    ],
}

# Neutral filler that appears in any note (makes the text task realistic)
FILLER_OPEN = ["board received from unit,", "checked on the bench,",
               "initial inspection done,", "unit reported fault,",
               "visual check complete,", ""]
FILLER_CLOSE = ["will replace and retest", "replaced and tested ok",
                "needs further testing", "parts ordered",
                "awaiting spare", "passes power up test now", ""]

def make_note(fault):
    """Build a technician-style note: 1-2 phrases from the true fault,
    sometimes 1 phrase from another fault (realistic confusion)."""
    opener = random.choice(FILLER_OPEN)
    parts = [p() for p in random.sample(PHRASES[fault], k=random.choice([1, 2, 2]))]
    if random.random() < 0.25:
        other = random.choice(FAULTS)
        if other != fault:
            parts.insert(random.randint(0, len(parts)), random.choice(PHRASES[other])())
    closer = random.choice(FILLER_CLOSE)
    if closer:
        parts.append(closer)
    note = (opener + " " + ", ".join(parts)).strip() + "."
    return note[0].upper() + note[1:]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

FAULT_HOURS = {"short_circuit": 2.5, "open_circuit": 2.0, "component_failure": 1.8,
               "corrosion": 4.0, "connector_fault": 1.2, "power_supply": 2.2}
FAULT_BER = {"short_circuit": 0.2, "open_circuit": 0.0, "component_failure": -0.2,
             "corrosion": 1.4, "connector_fault": -0.8, "power_supply": 0.0}

def make_records(n=600):
    rows = []
    for i in range(n):
        board = random.choice(list(BOARD_TYPES))
        fault = random.choice(FAULTS)
        age = int(np.clip(np.random.gamma(3, 2.5), 1, 25))
        prev = int(np.clip(np.random.poisson(0.15 * age), 0, 8))
        info = BOARD_TYPES[board]

        logit = (-3.2 + 0.16 * age + 0.45 * prev + FAULT_BER[fault]
                 + info["ber_bias"] + np.random.normal(0, 0.6))
        ber = np.random.rand() < sigmoid(logit)

        hours = (FAULT_HOURS[fault] * info["complexity"] + 0.05 * age
                 + 0.3 * prev + np.random.normal(0, 0.5))
        hours = round(float(np.clip(hours, 0.5, 12)), 1)

        rows.append({
            "job_id": f"J{1000 + i}",
            "board_type": board,
            "age_years": age,
            "previous_repairs": prev,
            "note": make_note(fault),
            "fault_category": fault,
            "outcome": "BER" if ber else "Repairable",
            "repair_hours": hours,
        })
    return pd.DataFrame(rows)

def make_registry():
    """Board history the app looks up after OCR reads the serial."""
    rows = []
    serials = [2291, 3107, 4452, 5018, 6630, 7714, 8125, 9043]
    boards = list(BOARD_TYPES)
    for i, s in enumerate(serials):
        board = boards[i % len(boards)]
        age = random.randint(2, 18)
        rows.append({
            "serial": str(s),
            "board_type": board,
            "revision": random.choice(["A", "B", "C"]),
            "age_years": age,
            "previous_repairs": random.randint(0, max(1, age // 4)),
            "last_fault": random.choice(FAULTS),
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    records = make_records()
    registry = make_registry()
    records.to_csv(DATA_DIR / "repair_records.csv", index=False)
    registry.to_csv(DATA_DIR / "board_registry.csv", index=False)
    print(f"Wrote {len(records)} repair records and {len(registry)} registry boards")
    print(records["fault_category"].value_counts().to_string())
    print(records["outcome"].value_counts().to_string())
    print(records.sample(5, random_state=1)[["note", "fault_category"]].to_string())
