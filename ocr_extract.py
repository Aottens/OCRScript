#!/usr/bin/env python3
"""Extract symbol labels and parameter text from CX Designer screenshots."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pytesseract

SYMBOL_REGEX = re.compile(r"^[A-Z][A-Z0-9_]*(\[[0-9]+\])?(\.[A-Z0-9_]+(\[[0-9]+\])?)*$")


@dataclass
class OcrLine:
    text: str
    conf: float
    bbox: Tuple[int, int, int, int]


@dataclass
class SymbolMatch:
    symbol: str
    conf: float
    bbox: Tuple[int, int, int, int]
    parameter: Optional[str] = None
    parameter_conf: Optional[float] = None
    unit: Optional[str] = None


@dataclass
class TemplateRois:
    symbol_roi: Optional[Tuple[int, int, int, int]]
    param_roi: Optional[Tuple[int, int, int, int]]
    unit_roi: Optional[Tuple[int, int, int, int]]


def load_symbols(path: Optional[str]) -> Optional[set[str]]:
    if not path:
        return None
    symbols: set[str] = set()
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            symbol = row[0].strip()
            if symbol:
                symbols.add(symbol)
    return symbols


def preprocess_image(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    thresh = cv2.adaptiveThreshold(
        scaled,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )
    blurred = cv2.medianBlur(thresh, 3)
    return blurred


def preprocess_symbols_image(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return cv2.medianBlur(scaled, 3)


def normalize_symbol_text(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_\.\[\]]", "", text)
    return cleaned.upper()


def ensure_tesseract() -> None:
    env_candidates = [
        os.environ.get("TESSERACT_CMD"),
        os.environ.get("TESSERACT_PATH"),
    ]
    candidates: List[str] = []
    for candidate in env_candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_dir():
            for name in ("tesseract.exe", "tesseract"):
                exe_path = path / name
                if exe_path.exists():
                    candidates.append(str(exe_path))
        else:
            candidates.append(str(path))

    current_cmd = pytesseract.pytesseract.tesseract_cmd
    if current_cmd:
        candidates.append(current_cmd)
    candidates.append("tesseract")

    resolved: Optional[str] = None
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_file():
            resolved = str(path)
            break
        found = shutil.which(candidate)
        if found:
            resolved = found
            break

    if not resolved:
        raise FileNotFoundError(
            "Tesseract is niet gevonden. Zet TESSERACT_CMD naar het volledige pad "
            "van tesseract.exe of voeg de installatiemap toe aan PATH."
        )

    pytesseract.pytesseract.tesseract_cmd = resolved
    try:
        _ = pytesseract.get_tesseract_version()
    except Exception as exc:
        raise FileNotFoundError(
            f"Tesseract is niet bruikbaar via '{resolved}'. Controleer je installatie "
            "of zet TESSERACT_CMD naar het juiste pad."
        ) from exc


def parse_page_tab(filename: str) -> Tuple[str, str]:
    base = Path(filename).stem
    if "__Tab_" in base and base.startswith("Page_"):
        page_part, tab_part = base.split("__Tab_", maxsplit=1)
        page = page_part.replace("Page_", "", 1)
        return page, tab_part
    return base, ""


def extract_symbol_lines(
    image: np.ndarray,
    symbol_lookup: Optional[Dict[str, str]],
    conf_threshold: float,
    psm: int,
) -> List[OcrLine]:
    config = f"--oem 1 --psm {psm}"
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=config)
    lines: Dict[Tuple[int, int, int], List[int]] = {}
    for idx, line_num in enumerate(data["line_num"]):
        if int(data["conf"][idx]) < 0:
            continue
        key = (data["block_num"][idx], data["par_num"][idx], line_num)
        lines.setdefault(key, []).append(idx)

    results: List[OcrLine] = []
    for indices in lines.values():
        words: List[str] = []
        confs: List[int] = []
        xs: List[int] = []
        ys: List[int] = []
        ws: List[int] = []
        hs: List[int] = []
        for idx in indices:
            text = data["text"][idx].strip()
            if not text:
                continue
            words.append(text)
            confs.append(int(data["conf"][idx]))
            xs.append(int(data["left"][idx]))
            ys.append(int(data["top"][idx]))
            ws.append(int(data["width"][idx]))
            hs.append(int(data["height"][idx]))
        if not words:
            continue
        line_text = " ".join(words)
        avg_conf = float(sum(confs) / max(len(confs), 1))
        x0 = min(xs)
        y0 = min(ys)
        x1 = max(x + w for x, w in zip(xs, ws))
        y1 = max(y + h for y, h in zip(ys, hs))
        bbox = (x0, y0, x1 - x0, y1 - y0)
        normalized = normalize_symbol_text(line_text)
        if symbol_lookup is not None:
            matched = symbol_lookup.get(normalized)
            if matched:
                results.append(OcrLine(text=matched, conf=avg_conf, bbox=bbox))
            continue
        if avg_conf < conf_threshold:
            continue
        if SYMBOL_REGEX.match(normalized):
            results.append(OcrLine(text=normalized, conf=avg_conf, bbox=bbox))
    return results


def ocr_roi(
    image: np.ndarray,
    roi: Tuple[int, int, int, int],
    psm: int,
) -> Tuple[str, float]:
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        return "", 0.0
    cropped = image[y : y + h, x : x + w]
    config = f"--oem 1 --psm {psm}"
    data = pytesseract.image_to_data(cropped, output_type=pytesseract.Output.DICT, config=config)
    words: List[str] = []
    confs: List[int] = []
    for idx, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue
        conf = int(data["conf"][idx])
        if conf < 0:
            continue
        words.append(text)
        confs.append(conf)
    if not words:
        return "", 0.0
    text_out = " ".join(words)
    avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
    return text_out, avg_conf


def is_valid_parameter(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 3:
        return False
    if stripped.replace(".", "").replace(",", "").isdigit():
        return False
    return True


def match_parameter_text(
    image: np.ndarray,
    symbol_match: SymbolMatch,
    band: int,
    max_expand: int,
    param_roi: Optional[Tuple[int, int, int, int]],
) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    x, y, w, h = symbol_match.bbox
    image_h, image_w = image.shape[:2]
    candidates: List[Tuple[str, float, float, Tuple[int, int, int, int]]] = []

    def clip_roi(x0: int, y0: int, x1: int, y1: int) -> Optional[Tuple[int, int, int, int]]:
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(image_w, x1)
        y1 = min(image_h, y1)
        if param_roi is not None:
            rx, ry, rw, rh = param_roi
            x0 = max(x0, rx)
            y0 = max(y0, ry)
            x1 = min(x1, rx + rw)
            y1 = min(y1, ry + rh)
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1 - x0, y1 - y0)

    for expand in (0, max_expand):
        y0 = y - band - expand
        y1 = y + h + band + expand
        roi = clip_roi(0, y0, x - 5, y1)
        if roi:
            for psm in (7, 6):
                text, conf = ocr_roi(image, roi, psm)
                if text and text.strip() != symbol_match.symbol:
                    distance = abs((y + h / 2) - (roi[1] + roi[3] / 2))
                    candidates.append((text, conf, distance, roi))
        if candidates:
            break

    if not candidates:
        roi = clip_roi(0, y - max_expand, x - 5, y)
        if roi:
            text, conf = ocr_roi(image, roi, 6)
            if text and text.strip() != symbol_match.symbol:
                distance = abs((y - (roi[1] + roi[3] / 2)))
                candidates.append((text, conf, distance, roi))

    if not candidates:
        return None, None, None

    best_score = float("-inf")
    best: Optional[Tuple[str, float]] = None
    for text, conf, distance, _roi in candidates:
        cleaned = text.strip()
        if not is_valid_parameter(cleaned):
            continue
        score = conf - (distance * 0.1)
        if score > best_score:
            best_score = score
            best = (cleaned, conf)

    if not best:
        return None, None, None

    return best[0], best[1], None


def apply_templates_fallback(
    image: np.ndarray,
    template: TemplateRois,
    symbol_lookup: Optional[Dict[str, str]],
    conf_threshold: float,
) -> List[OcrLine]:
    if not template.symbol_roi:
        return []
    x, y, w, h = template.symbol_roi
    cropped = image[y : y + h, x : x + w]
    symbol_lines = extract_symbol_lines(cropped, symbol_lookup, conf_threshold, psm=6)
    adjusted: List[OcrLine] = []
    for line in symbol_lines:
        bx, by, bw, bh = line.bbox
        adjusted.append(OcrLine(text=line.text, conf=line.conf, bbox=(bx + x, by + y, bw, bh)))
    return adjusted


def load_templates(path: Optional[str]) -> Dict[str, TemplateRois]:
    if not path:
        return {}
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    templates: Dict[str, TemplateRois] = {}
    for key, value in raw.items():
        templates[key] = TemplateRois(
            symbol_roi=tuple(value.get("symbol_roi")) if value.get("symbol_roi") else None,
            param_roi=tuple(value.get("param_roi")) if value.get("param_roi") else None,
            unit_roi=tuple(value.get("unit_roi")) if value.get("unit_roi") else None,
        )
    return templates


def select_template(name: str, templates: Dict[str, TemplateRois]) -> Optional[TemplateRois]:
    for key, tpl in templates.items():
        if fnmatch(name, key):
            return tpl
    return None


def write_outputs(path: str, rows: Iterable[Dict[str, str]]) -> None:
    fieldnames = [
        "file",
        "page",
        "tab",
        "symbol",
        "parameter_text",
        "unit",
        "symbol_confidence",
        "parameter_confidence",
        "x_symbol",
        "y_symbol",
        "w_symbol",
        "h_symbol",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def draw_debug(image: np.ndarray, matches: Sequence[SymbolMatch]) -> np.ndarray:
    debug = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    for match in matches:
        x, y, w, h = match.bbox
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            debug,
            match.symbol,
            (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    return debug


def run_extraction(
    in_dir: Path,
    out_csv: Path,
    symbols_csv: Optional[str],
    templates_json: Optional[str],
    debug_dir: Optional[Path],
    min_symbols: int,
    symbol_conf: float,
) -> int:
    ensure_tesseract()
    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")

    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    symbols_set = load_symbols(symbols_csv)
    symbol_lookup = None
    if symbols_set is not None:
        symbol_lookup = {normalize_symbol_text(symbol): symbol for symbol in symbols_set}
    templates = load_templates(templates_json)

    files = sorted(in_dir.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No PNG screenshots found in: {in_dir}")

    rows: List[Dict[str, str]] = []
    for file_path in files:
        image_bgr = cv2.imread(str(file_path))
        if image_bgr is None:
            print(f"Warning: unable to read {file_path}", file=sys.stderr)
            continue
        processed = preprocess_image(image_bgr)
        symbols_image = preprocess_symbols_image(image_bgr)
        page, tab = parse_page_tab(file_path.name)

        symbol_lines = extract_symbol_lines(processed, symbol_lookup, symbol_conf, psm=11)
        if not symbol_lines:
            symbol_lines = extract_symbol_lines(processed, symbol_lookup, symbol_conf, psm=6)
        if not symbol_lines:
            symbol_lines = extract_symbol_lines(symbols_image, symbol_lookup, symbol_conf, psm=6)
        avg_conf = sum(line.conf for line in symbol_lines) / max(len(symbol_lines), 1)

        template = select_template(file_path.name, templates) or select_template(page, templates)
        if (len(symbol_lines) < min_symbols or avg_conf < symbol_conf) and template:
            symbol_lines = apply_templates_fallback(processed, template, symbol_lookup, symbol_conf)

        if not symbol_lines:
            print(f"Warning: no symbols detected in {file_path.name}", file=sys.stderr)

        matches: List[SymbolMatch] = []
        for line in symbol_lines:
            match = SymbolMatch(symbol=line.text, conf=line.conf, bbox=line.bbox)
            param_text, param_conf, unit = match_parameter_text(
                processed,
                match,
                band=12,
                max_expand=30,
                param_roi=template.param_roi if template else None,
            )
            match.parameter = param_text
            match.parameter_conf = param_conf
            match.unit = unit
            matches.append(match)

            x, y, w, h = match.bbox
            rows.append(
                {
                    "file": file_path.name,
                    "page": page,
                    "tab": tab,
                    "symbol": match.symbol,
                    "parameter_text": match.parameter or "",
                    "unit": match.unit or "",
                    "symbol_confidence": f"{match.conf:.2f}",
                    "parameter_confidence": f"{match.parameter_conf:.2f}" if match.parameter_conf else "",
                    "x_symbol": str(x),
                    "y_symbol": str(y),
                    "w_symbol": str(w),
                    "h_symbol": str(h),
                }
            )

        if debug_dir:
            debug_image = draw_debug(processed, matches)
            debug_path = debug_dir / f"{file_path.stem}_debug.png"
            cv2.imwrite(str(debug_path), debug_image)
            log = {
                "file": file_path.name,
                "symbol_lines": [
                    {
                        "text": line.text,
                        "confidence": line.conf,
                        "bbox": line.bbox,
                    }
                    for line in symbol_lines
                ],
                "symbols": [
                    {
                        "symbol": match.symbol,
                        "confidence": match.conf,
                        "bbox": match.bbox,
                        "parameter": match.parameter,
                        "parameter_confidence": match.parameter_conf,
                    }
                    for match in matches
                ],
            }
            log_path = debug_dir / f"{file_path.stem}_log.json"
            with open(log_path, "w", encoding="utf-8") as handle:
                json.dump(log, handle, indent=2)

    write_outputs(str(out_csv), rows)
    if not rows:
        print(
            "Warning: no rows were extracted. Try enabling --debug_dir to inspect OCR output.",
            file=sys.stderr,
        )
    print(f"Wrote {len(rows)} rows to {out_csv}")
    return 0


def main() -> int:
    readme_usage = (
        "OCRScript usage:\n"
        "  python ocr_extract.py --in_dir screenshots --out_csv output.csv \n"
        "  Optional: --symbols_csv symbols.csv --templates_json templates.json --debug_dir debug\n"
    )

    parser = argparse.ArgumentParser(
        description="Extract symbol/parameter tables from CX Designer screenshots.",
        epilog=readme_usage,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--in_dir", required=True, help="Input directory with PNG screenshots")
    parser.add_argument("--out_csv", required=True, help="Output CSV path")
    parser.add_argument("--symbols_csv", help="Optional CSV with valid symbols (1 column)")
    parser.add_argument("--templates_json", help="Optional JSON with fallback ROIs")
    parser.add_argument("--debug_dir", help="Optional directory for debug outputs")
    parser.add_argument("--min_symbols", type=int, default=2, help="Minimum symbols before fallback")
    parser.add_argument("--symbol_conf", type=float, default=55.0, help="Confidence threshold for symbols")
    args = parser.parse_args()

    try:
        return run_extraction(
            in_dir=Path(args.in_dir),
            out_csv=Path(args.out_csv),
            symbols_csv=args.symbols_csv,
            templates_json=args.templates_json,
            debug_dir=Path(args.debug_dir) if args.debug_dir else None,
            min_symbols=args.min_symbols,
            symbol_conf=args.symbol_conf,
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
