"""
Build the 50 demo photos and the matching board registry.

Photos: real PCB photographs from the Roboflow 100 "printed circuit board"
dataset (CC BY 4.0). Only boards from its validation and test splits are used,
so the part detector never saw them during training.

Each photo gets a synthetic yellow asset tag (board family, revision, serial
number) pasted in the corner with the fewest components, the way a repair
shop tags incoming boards. The registry (age, previous repairs, last fault)
is synthetic, generated with a fixed seed.

Writes: sample_images/*.jpg, android/app/src/main/assets/demo/*.jpg,
        data/board_registry.csv
Usage:  python make_demo_images.py path/to/unzipped/rf100
"""
import random
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
WEB_OUT = ROOT / "sample_images"
APP_OUT = ROOT / "android" / "app" / "src" / "main" / "assets" / "demo"
REGISTRY = ROOT / "data" / "board_registry.csv"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
N_DEMO = 50
LONG_SIDE = 1400
SEED = 11

BOARD_FAMILIES = ["PSU-CTRL", "MCU-MAIN", "SENSOR-IF", "RF-COMM", "MOTOR-DRV", "DISPLAY-DRV"]
FAULTS = ["short_circuit", "open_circuit", "component_failure", "corrosion", "connector_fault", "power_supply"]
TAG_YELLOW = (255, 212, 0)


def source_name(path):
    return re.sub(r"\.rf\.[0-9a-f]+$", "", path.stem)


def boxes_for(label_file, w, h):
    out = []
    for line in label_file.read_text().splitlines():
        if line.strip():
            v = list(map(float, line.split()[1:]))
            if len(v) > 4:   # polygon -> bounding box
                xs, ys = v[0::2], v[1::2]
                v = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)]
            x, y, bw, bh = v
            out.append(((x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h))
    return out


def overlap(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def make_tag(lines, width):
    h = int(width * 0.42)
    tag = Image.new("RGB", (width, h), TAG_YELLOW)
    d = ImageDraw.Draw(tag)
    border = max(3, width // 90)
    d.rectangle([0, 0, width - 1, h - 1], outline=(20, 20, 20), width=border)
    sizes = [int(h * 0.21), int(h * 0.19), int(h * 0.21)]
    y = int(h * 0.09)
    for text, size in zip(lines, sizes):
        f = ImageFont.truetype(FONT, size)
        d.text((int(width * 0.06), y), text, fill=(10, 10, 10), font=f)
        y += int(size * 1.35)
    return tag


def place_tag(img, tag, boxes):
    """Corner (with a small margin) whose tag area covers the fewest components."""
    m = int(img.width * 0.015)
    corners = [(m, m), (img.width - tag.width - m, m),
               (m, img.height - tag.height - m), (img.width - tag.width - m, img.height - tag.height - m)]
    def cost(c):
        r = (c[0], c[1], c[0] + tag.width, c[1] + tag.height)
        return sum(overlap(r, b) for b in boxes)
    best = min(corners, key=cost)
    img.paste(tag, best)
    return best


def main(src: Path):
    rnd = random.Random(SEED)
    held_out = {}
    for split in ["test", "valid"]:
        for p in sorted((src / split / "images").glob("*.jpg")):
            w, h = Image.open(p).size
            if max(w, h) / min(w, h) <= 2.2:      # skip long strip boards: poor demo photos
                held_out.setdefault(source_name(p), (split, p))
    names = sorted(held_out)
    named = [n for n in names if not n.startswith("pcb")]          # well-known dev boards first
    others = [n for n in names if n.startswith("pcb")]
    chosen = (named + others)[:N_DEMO]

    for d in (WEB_OUT, APP_OUT):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    serials = rnd.sample(range(1000, 9999), N_DEMO)
    rows = []
    for i, board in enumerate(chosen):
        split, path = held_out[board]
        img = Image.open(path).convert("RGB")
        s = LONG_SIDE / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
        label = src / split / "labels" / (path.stem + ".txt")
        family = BOARD_FAMILIES[i % len(BOARD_FAMILIES)]
        rev = rnd.choice("ABC")
        serial = str(serials[i])
        tag_w = max(320, int(0.30 * (img.width * img.height) ** 0.5))
        tag = make_tag([family, f"REV {rev}", f"S/N {serial}"], tag_w)
        place_tag(img, tag, boxes_for(label, img.width, img.height))
        fname = f"board_{serial}.jpg"
        img.save(WEB_OUT / fname, quality=88)
        shutil.copy(WEB_OUT / fname, APP_OUT / fname)
        age = rnd.randint(1, 20)
        rows.append({
            "serial": serial, "board_type": family, "revision": rev, "age_years": age,
            "previous_repairs": rnd.randint(0, max(1, age // 4)), "last_fault": rnd.choice(FAULTS),
            "image": fname, "source": board.replace("_jpg", "").replace("_png", ""),
        })
    pd.DataFrame(rows).to_csv(REGISTRY, index=False)
    print(f"wrote {len(rows)} demo photos and {REGISTRY.relative_to(ROOT)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
