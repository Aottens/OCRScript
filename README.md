# OCRScript

## Installatie

Installeer de Python afhankelijkheden (inclusief OpenCV) voordat je het script uitvoert:

```bash
python -m pip install opencv-python numpy pytesseract pillow pyqt5
```

> **Let op:** Als `cv2` ontbreekt krijg je een fout zoals `ModuleNotFoundError: No module named 'cv2'`.
> Als Tesseract niet gevonden wordt kun je `TESSERACT_CMD` instellen naar het volledige pad
> van `tesseract.exe` (bijv. `C:\Program Files\Tesseract-OCR\tesseract.exe`).

## Gebruik

```bash
python ocr_extract.py --in_dir screenshots --out_csv output.csv \
  --symbols_csv symbols.csv --templates_json templates.json --debug_dir debug
```

Zie `python ocr_extract.py --help` voor alle opties.

## GUI

Start de PyQt GUI met:

```bash
python ocr_gui.py
```

Gebruik de knop "Check Tesseract" om snel te controleren welke executable gevonden is.
