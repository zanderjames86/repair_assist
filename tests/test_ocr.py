"""OCR reads the asset tag on the 50 real board photos (Tesseract pipeline)."""
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest
from PIL import Image

from conftest import ROOT
from ocr_utils import find_label, parse_label, read_text

REG = pd.read_csv(ROOT / "data" / "board_registry.csv", dtype={"serial": str})
BOARDS = sorted(REG["board_type"].unique())


def _read(row):
    img = Image.open(ROOT / "sample_images" / row["image"])
    return (parse_label(read_text(img, clean=True), BOARDS),
            parse_label(read_text(img, clean=False), BOARDS))


@pytest.fixture(scope="module")
def results():
    rows = REG.to_dict("records")
    with ThreadPoolExecutor(4) as ex:   # tesseract runs as a subprocess, threads are enough
        return list(zip(rows, ex.map(_read, rows)))


def test_serial_accuracy_with_cleanup(results):
    ok = sum(clean["serial"] == row["serial"] for row, (clean, _) in results)
    print(f"serial read correctly on {ok}/{len(results)} photos")
    assert ok / len(results) >= 0.95


def test_no_wrong_serials(results):
    """A missed read is acceptable (the technician picks it), a wrong one is not."""
    wrong = [row["image"] for row, (clean, _) in results
             if clean["serial"] not in (None, row["serial"])]
    assert not wrong, wrong


def test_board_type_and_revision(results):
    board = sum(c["board_type"] == r["board_type"] for r, (c, _) in results)
    rev = sum(c["revision"] == r["revision"] for r, (c, _) in results)
    assert board >= 47 and rev >= 47


def test_cleanup_beats_raw_ocr(results):
    clean = sum(c["serial"] == r["serial"] for r, (c, _) in results)
    raw = sum(w["serial"] == r["serial"] for r, (_, w) in results)
    assert clean > raw + 20


def test_tag_is_located():
    img = Image.open(ROOT / "sample_images" / REG.iloc[0]["image"])
    crop = find_label(img)
    assert crop.size[0] < img.size[0] / 2 and 1.6 < crop.size[0] / crop.size[1] < 3.6


@pytest.mark.parametrize("text,expected", [
    ("PSU-CTRL\nREV A\nS/N 2291", ("PSU-CTRL", "A", "2291")),
    ("RF - C0MM REVC S|N: 5O18", ("RF-COMM", "C", "5018")),
    ("mcu_main rev. b sn-3107", ("MCU-MAIN", "B", "3107")),
    ("nothing useful here", (None, None, None)),
])
def test_parse_label(text, expected):
    p = parse_label(text, BOARDS)
    assert (p["board_type"], p["revision"], p["serial"]) == expected
