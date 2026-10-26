"""
Build the part-detector training set from the Roboflow 100 "printed circuit
board" dataset (CC BY 4.0, https://universe.roboflow.com/roboflow-100/printed-circuit-board/dataset/4).

  * keeps one copy of each photographed board (Roboflow's train split holds
    several augmented copies of the same photo; YOLO augments on the fly)
  * keeps Roboflow's split, which is already separated by board
    (no board appears in more than one split)
  * merges the 23 original labels into 7 part types and drops non-parts
    (pads, test points, "unknown", "EM")
  * resizes every photo so the long side is IMG_SIDE pixels

Usage:  python detector/prepare_dataset.py path/to/unzipped/rf100  [out_dir]
"""
import collections
import json
import re
import shutil
import sys
from pathlib import Path

import yaml
from PIL import Image

IMG_SIDE = 1280
MERGE = {
    "IC": "IC", "iC": "IC", "Clock": "IC",
    "Capacitor": "Capacitor", "Electrolytic Capacitor": "Capacitor", "Capacitor Jumper": "Capacitor",
    "Resistor": "Resistor", "Resistor Network": "Resistor", "Resistor Jumper": "Resistor",
    "Connector": "Connector", "Pins": "Connector", "Jumper": "Connector",
    "Transistor": "Transistor/Diode", "Diode": "Transistor/Diode", "Led": "Transistor/Diode",
    "Inductor": "Inductor", "Ferrite Bead": "Inductor",
    "Button": "Switch", "Switch": "Switch",
    # dropped: Pads, Test Point, Unknown Unlabeled, EM
}
CLASSES = ["Capacitor", "Connector", "IC", "Inductor", "Resistor", "Switch", "Transistor/Diode"]


def to_box(values):
    """YOLO box (cx, cy, w, h); polygons (rare in this export) become their bounding box."""
    v = [float(x) for x in values]
    if len(v) == 4:
        return v
    xs, ys = v[0::2], v[1::2]
    return [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)]


def source_name(path: Path) -> str:
    return re.sub(r"\.rf\.[0-9a-f]+$", "", path.stem)


def main(src: Path, out: Path):
    names = yaml.safe_load((src / "data.yaml").read_text())["names"]
    if out.exists():
        shutil.rmtree(out)
    stats = {"images": {}, "boxes": collections.Counter(), "dropped": collections.Counter()}
    for split in ["train", "valid", "test"]:
        (out / split / "images").mkdir(parents=True)
        (out / split / "labels").mkdir(parents=True)
        seen = set()
        for img_path in sorted((src / split / "images").glob("*.jpg")):
            board = source_name(img_path)
            if board in seen:
                continue
            seen.add(board)
            lines = []
            for line in (src / split / "labels" / (img_path.stem + ".txt")).read_text().splitlines():
                if not line.strip():
                    continue
                c, *box = line.split()
                original = names[int(c)]
                merged = MERGE.get(original)
                if merged is None:
                    stats["dropped"][original] += 1
                    continue
                stats["boxes"][merged] += 1
                lines.append(" ".join([str(CLASSES.index(merged))] + [f"{x:.6f}" for x in to_box(box)]))
            img = Image.open(img_path).convert("RGB")
            s = IMG_SIDE / max(img.size)
            img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
            img.save(out / split / "images" / f"{board}.jpg", quality=90)
            (out / split / "labels" / f"{board}.txt").write_text("\n".join(lines) + "\n")
        stats["images"][split] = len(seen)

    (out / "data.yaml").write_text(yaml.safe_dump({
        "path": str(out.resolve()), "train": "train/images", "val": "valid/images",
        "test": "test/images", "names": dict(enumerate(CLASSES)),
    }, sort_keys=False))
    summary = {"images": stats["images"], "boxes": dict(stats["boxes"]), "dropped": dict(stats["dropped"]),
               "classes": CLASSES, "source": "Roboflow 100 printed-circuit-board v4 (CC BY 4.0)"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / "dataset"
    main(src, out)
