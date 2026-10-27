from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pymupdf
from src.parser import DocumentParseError, _ocr_blocks, extract_document_from_pdf, order_page_blocks


class PdfStructureTests(unittest.TestCase):
    def test_two_columns_are_read_column_by_column(self):
        def block(name, x, y, width=220):
            return {"text": name + " word " * 8, "bbox": [x, y, x + width, y + 20]}
        blocks = [block("right-top", 330, 100), block("left-bottom", 30, 180), block("right-bottom", 330, 180), block("left-top", 30, 100)]
        ordered, layout = order_page_blocks(blocks, 600)
        self.assertEqual(layout, "two_column_heuristic")
        self.assertEqual([block["text"].split()[0] for block in ordered], ["left-top", "left-bottom", "right-top", "right-bottom"])

    def test_digital_page_does_not_get_ocr_when_another_page_is_scanned(self):
        import io
        from PIL import Image
        document = pymupdf.open()
        document.new_page().insert_text((50, 80), "Digital semiconductor device research with measured current and gate voltage. " * 2)
        image = Image.new("RGB", (400, 300), "white")
        output = io.BytesIO()
        image.save(output, format="PNG")
        document.new_page().insert_image(pymupdf.Rect(0, 0, 400, 300), stream=output.getvalue())
        content = document.tobytes()
        document.close()
        text = "The scanned page reports a measured threshold voltage and leakage current."
        raw = {"text": text, "bbox": [10, 20, 300, 40], "lines": [{"text": text, "bbox": [10, 20, 300, 40], "font_size": 10}]}
        with patch("src.parser._ocr_blocks", return_value=([raw], "test OCR")) as ocr:
            parsed = extract_document_from_pdf(content, "mixed.pdf")
        self.assertEqual(ocr.call_count, 1)
        self.assertEqual(parsed["pages"][0]["parse_method"], "기본 파싱")
        self.assertEqual(parsed["pages"][1]["parse_method"], "test OCR")
        self.assertTrue(any(block["page_num"] == 2 and block["bbox"] == [10, 20, 300, 40] for block in parsed["blocks"]))

    def test_ocr_coordinates_are_converted_to_pdf_units(self):
        document = pymupdf.open()
        page = document.new_page(width=144, height=72)
        engine = Mock(return_value=([([[0, 0], [100, 0], [100, 50], [0, 50]], "Measured drain current", 0.99)], None))
        with patch("src.parser.get_rapid_ocr_engine", return_value=engine):
            blocks, method = _ocr_blocks(page)
        self.assertEqual(method, "RapidOCR")
        engine.assert_called_once()
        self.assertAlmostEqual(blocks[0]["bbox"][2], 40.0)
        self.assertAlmostEqual(blocks[0]["bbox"][3], 20.0)
        document.close()

    def test_blank_scan_reports_no_text(self):
        with pymupdf.open() as document:
            page = document.new_page()
            engine = Mock(return_value=(None, None))
            with patch("src.parser.get_rapid_ocr_engine", return_value=engine):
                with self.assertRaisesRegex(DocumentParseError, "텍스트"):
                    _ocr_blocks(page)

    def test_ocr_failure_is_reported_explicitly(self):
        with pymupdf.open() as document:
            page = document.new_page()
            engine = Mock(side_effect=RuntimeError("inference failed"))
            with patch("src.parser.get_rapid_ocr_engine", return_value=engine):
                with self.assertRaisesRegex(DocumentParseError, "RapidOCR"):
                    _ocr_blocks(page)


if __name__ == "__main__":
    unittest.main(verbosity=2)
