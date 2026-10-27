"""Regression checks for local startup and accidental expensive work."""

from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class StartupTests(unittest.TestCase):
    def test_01_landing_page_does_not_load_ai_engines(self):
        from streamlit.testing.v1 import AppTest

        previously_loaded = set(sys.modules)
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.tabs), 4)
        newly_loaded = set(sys.modules) - previously_loaded
        self.assertFalse({"torch", "transformers", "onnxruntime", "src.parser", "src.vector_store"} & newly_loaded)

    def test_02_typing_and_settings_do_not_submit_queries(self):
        from streamlit.testing.v1 import AppTest

        search = Mock()
        generate = Mock()
        search_module = types.ModuleType("src.search_hybrid")
        search_module.search_hybrid_all = search
        llm_module = types.ModuleType("src.llm_generator")
        llm_module.generate_gemini_rag_stream = generate
        with patch.dict(sys.modules, {"src.search_hybrid": search_module, "src.llm_generator": llm_module}):
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
            app.text_input(key="rag_query_input").set_value("GAA device structure").run()
            app.text_input(key="debug_query_input").set_value("IGZO").run()
            app.slider(key="sidebar_vec_slider").set_value(0.5).run()
            self.assertEqual(len(app.exception), 0)
            search.assert_not_called()
            generate.assert_not_called()

    def test_03_digital_pdf_does_not_start_ocr(self):
        import pymupdf
        from src.parser import extract_text_from_pdf

        document = pymupdf.open()
        document.new_page().insert_text((60, 70), "Semiconductor device research. " * 4)
        pdf_bytes = document.tobytes()
        document.close()
        with patch("src.parser.get_rapid_ocr_engine") as ocr:
            text, method = extract_text_from_pdf(pdf_bytes)
            self.assertIn("Semiconductor", text)
            self.assertNotIn("OCR", method)
            ocr.assert_not_called()

    @unittest.skipUnless(sys.platform == "win32", "Windows single-instance lock")
    def test_04_only_one_launcher_can_own_the_server(self):
        import launcher

        with tempfile.TemporaryDirectory(dir=ROOT / ".local") as folder:
            with patch.object(launcher, "STATE_DIR", Path(folder)):
                first = launcher.InstanceLock()
                second = launcher.InstanceLock()
                try:
                    self.assertTrue(first.acquire())
                    self.assertFalse(second.acquire())
                    first.close()
                    self.assertTrue(second.acquire())
                finally:
                    first.close()
                    second.close()


if __name__ == "__main__":
    (ROOT / ".local").mkdir(exist_ok=True)
    unittest.main(verbosity=2)
