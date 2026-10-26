"""Shared fixture: synthetic YOLO raw outputs + the Python decoder's result.
android/.../DetectorParityTest.java must decode them identically."""
import json
import random

import numpy as np

from detector_utils import decode

OUT = "android/app/src/test/resources/detector_parity.json"


def case(seed, nc=7, n=600, size=1024, w=1400, h=1050):
    rnd = np.random.default_rng(seed)
    out = np.zeros((4 + nc, n), np.float32)
    centres = rnd.uniform(50, size - 50, (25, 2))
    for a in range(n):                     # clusters of overlapping boxes, like real YOLO output
        c = centres[a % 25] + rnd.normal(0, 6, 2)
        out[0, a], out[1, a] = c
        out[2, a], out[3, a] = rnd.uniform(8, 120, 2)
        out[4 + (a % 25) % nc, a] = rnd.uniform(0, 1)
        out[4:, a] += rnd.uniform(0, 0.2, nc)
    s = min(size / w, size / h)
    px, py = (size - round(w * s)) // 2, (size - round(h * s)) // 2
    dets = decode(out, s, px, py, w, h, 0.30, 0.45)
    return {"nc": nc, "n": n, "size": size, "w": w, "h": h, "out": out.ravel().tolist(),
            "dets": [{"c": d["class_id"], "s": d["score"], "b": d["box"]} for d in dets]}


if __name__ == "__main__":
    cases = [case(1), case(2, w=900, h=1600), case(3, n=50)]
    json.dump(cases, open(OUT, "w"))
    print([len(c["dets"]) for c in cases], "detections per case")
