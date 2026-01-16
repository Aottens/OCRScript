# OCRScript

## Installatie

Installeer de Python afhankelijkheden (inclusief OpenCV) voordat je het script uitvoert:

```bash
python -m pip install opencv-python numpy pytesseract pillow
```

> **Let op:** Als `cv2` ontbreekt krijg je een fout zoals `ModuleNotFoundError: No module named 'cv2'`.

## Gebruik

```bash
python ocr_extract.py --in_dir screenshots --out_csv output.csv \
  --symbols_csv symbols.csv --templates_json templates.json --debug_dir debug
```

Zie `python ocr_extract.py --help` voor alle opties.
