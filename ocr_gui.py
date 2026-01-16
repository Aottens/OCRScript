#!/usr/bin/env python3
"""PyQt GUI for OCRScript extraction."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

import ocr_extract


class OcrGui(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OCRScript - CX Designer OCR")
        self.setMinimumWidth(720)
        layout = QtWidgets.QVBoxLayout(self)

        self.in_dir = self._path_row("Input map", layout, self._browse_dir)
        self.out_csv = self._path_row("Output CSV", layout, self._browse_file_save)
        self.symbols_csv = self._path_row("Symbols CSV (optioneel)", layout, self._browse_file)
        self.templates_json = self._path_row("Templates JSON (optioneel)", layout, self._browse_file)
        self.debug_dir = self._path_row("Debug map (optioneel)", layout, self._browse_dir)

        options_layout = QtWidgets.QFormLayout()
        self.min_symbols = QtWidgets.QSpinBox()
        self.min_symbols.setRange(0, 999)
        self.min_symbols.setValue(2)
        options_layout.addRow("Min symbols", self.min_symbols)

        self.symbol_conf = QtWidgets.QDoubleSpinBox()
        self.symbol_conf.setRange(0.0, 100.0)
        self.symbol_conf.setValue(55.0)
        self.symbol_conf.setDecimals(1)
        options_layout.addRow("Symbol confidence", self.symbol_conf)

        layout.addLayout(options_layout)

        action_row = QtWidgets.QHBoxLayout()
        self.check_button = QtWidgets.QPushButton("Check Tesseract")
        self.check_button.clicked.connect(self._check_tesseract)
        action_row.addWidget(self.check_button)

        self.run_button = QtWidgets.QPushButton("Start OCR")
        self.run_button.clicked.connect(self._run)
        action_row.addWidget(self.run_button)
        layout.addLayout(action_row)

        self.status = QtWidgets.QTextEdit()
        self.status.setReadOnly(True)
        layout.addWidget(self.status)

    def _path_row(
        self,
        label: str,
        parent_layout: QtWidgets.QVBoxLayout,
        browse_handler,
    ) -> QtWidgets.QLineEdit:
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(label))
        edit = QtWidgets.QLineEdit()
        row.addWidget(edit)
        button = QtWidgets.QPushButton("Browse")
        button.clicked.connect(lambda: browse_handler(edit))
        row.addWidget(button)
        parent_layout.addLayout(row)
        return edit

    def _browse_dir(self, target: QtWidgets.QLineEdit) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Kies map")
        if directory:
            target.setText(directory)

    def _browse_file(self, target: QtWidgets.QLineEdit) -> None:
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Kies bestand")
        if filename:
            target.setText(filename)

    def _browse_file_save(self, target: QtWidgets.QLineEdit) -> None:
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Kies output CSV", filter="CSV (*.csv)")
        if filename:
            target.setText(filename)

    def _run(self) -> None:
        self.status.clear()
        in_dir = self.in_dir.text().strip()
        out_csv = self.out_csv.text().strip()
        if not in_dir or not out_csv:
            self._log("Input map en output CSV zijn verplicht.")
            return

        symbols_csv = self._optional_path(self.symbols_csv.text())
        templates_json = self._optional_path(self.templates_json.text())
        debug_dir = self._optional_path(self.debug_dir.text())

        self._log("Start verwerking...")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            config = ocr_extract.ExtractionConfig(
                min_symbols=self.min_symbols.value(),
                symbol_conf=self.symbol_conf.value(),
            )
            ocr_extract.run_extraction(
                in_dir=Path(in_dir),
                out_csv=Path(out_csv),
                symbols_csv=symbols_csv,
                templates_json=templates_json,
                debug_dir=Path(debug_dir) if debug_dir else None,
                config=config,
            )
        except Exception as exc:
            self._log(f"Fout: {exc}")
        else:
            self._log(self._summarize_output(out_csv))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _check_tesseract(self) -> None:
        self.status.clear()
        self._log("Controleer Tesseract...")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            ocr_extract.ensure_tesseract()
            cmd = ocr_extract.pytesseract.pytesseract.tesseract_cmd
        except Exception as exc:
            self._log(f"Fout: {exc}")
        else:
            self._log(f"Tesseract gevonden: {cmd}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _optional_path(self, value: str) -> str | None:
        cleaned = value.strip()
        return cleaned if cleaned else None

    def _log(self, message: str) -> None:
        self.status.append(message)

    def _summarize_output(self, out_csv: str) -> str:
        path = Path(out_csv)
        if not path.exists():
            return "Klaar, maar output CSV is niet gevonden."
        try:
            with open(path, newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                rows = list(reader)
        except Exception:
            return "Klaar, maar output CSV kon niet worden gelezen."
        if len(rows) <= 1:
            return (
                "Klaar, maar er zijn geen rijen geëxtraheerd. "
                "Probeer een debug map in te stellen om OCR-boxes te inspecteren."
            )
        return f"Klaar. {len(rows) - 1} rijen geëxtraheerd."


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    window = OcrGui()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
