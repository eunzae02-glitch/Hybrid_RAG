"""Regression cases for evidence selection, refusals and explicit submission."""

from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.retrieval_quality import is_research_content, normalize_search_query
from src.search_bm25 import build_vector_query, expand_query, load_ontology, tokenize

EVIDENCE = {
    "file_name": "device.pdf", "chunk_id": 7, "page_num": 2, "section_title": "Results",
    "text": "[Results]\nWe measured the threshold voltage and leakage current of the fabricated transistor. The results show improved gate control under the stated measurement conditions.",
}


class RetrievalQualityTests(unittest.TestCase):
    def test_korean_particle_does_not_hide_english_term(self):
        self.assertEqual(tokenize("MOSFET에 대해"), ["mosfet", "에", "대해"])
        self.assertEqual(normalize_search_query("mosfet에 대해 알려줘"), "mosfet")
        tokens, matches = expand_query("mosfet에 대해 알려줘", load_ontology())
        self.assertIn("mosfet", tokens)
        self.assertIn("transistor", tokens)
        self.assertTrue(any(match.startswith("MOSFET ->") for match in matches))
        self.assertFalse(any(match.startswith("FET ->") for match in matches))
        self.assertIn("Field Effect Transistor", build_vector_query("mosfet에 대해 알려줘"))

    def test_scaffolding_is_not_paper_evidence(self):
        self.assertFalse(is_research_content("[RESULTS AND DISCUSSION]\nIV."))
        self.assertFalse(is_research_content("[Introduction]\n[참고문헌(References) 섹션 자동 절사 완료]"))
        self.assertFalse(is_research_content("[Introduction]\nCorresponding author. Department of Semiconductor Engineering, University. Phone: 123456. E-mail: author@example.com. " * 3))
        self.assertTrue(is_research_content(EVIDENCE["text"]))
        self.assertTrue(is_research_content(EVIDENCE["text"] + " Corresponding author: author@example.com"))

    def test_context_selector_skips_title_and_contact_chunks(self):
        from src.rag_ui import select_context

        papers = [{"matched_chunks": [{"chunk": {"file_name": "device.pdf", "chunk_id": 1, "text": "[Results]\nIV."}}, {"chunk": EVIDENCE}]}]
        self.assertEqual(select_context(papers), [{"chunk": EVIDENCE}])

    def test_refusal_detection_does_not_reject_partial_answers(self):
        from src.llm_generator import INSUFFICIENT_EVIDENCE, is_insufficient_response

        self.assertTrue(is_insufficient_response(INSUFFICIENT_EVIDENCE))
        self.assertFalse(is_insufficient_response("관련 정보를 찾을 수 없습니다만, 일부 실험 결과는 다음과 같습니다."))

    def test_ui_refusal_is_neutral_and_result_survives_rerun(self):
        from streamlit.testing.v1 import AppTest
        from src.llm_generator import INSUFFICIENT_EVIDENCE

        search = Mock(return_value=([], [], [{"matched_chunks": [{"chunk": EVIDENCE}]}], [], 1))
        def respond(**kwargs):
            kwargs["on_model_used"]("verified-model")
            yield INSUFFICIENT_EVIDENCE
        with patch("src.search_hybrid.search_hybrid_all", search), patch("src.llm_generator.generate_gemini_rag_stream", side_effect=respond) as generate:
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
            app.text_input(key="rag_query_input").set_value("mosfet에 대해 알려줘").run()
            app.button(key="rag_btn").click().run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.session_state["rag_result"]["status"], "insufficient")
            self.assertEqual(app.session_state["rag_result"]["model"], "verified-model")
            notices = " ".join(item.value for item in app.info)
            self.assertIn("이번 검색", notices)
            self.assertNotIn("양자", notices)
            self.assertNotIn("안전하게 차단", notices)
            app.slider(key="sidebar_vec_slider").set_value(0.5).run()
            self.assertEqual(app.session_state["rag_result"]["text"], INSUFFICIENT_EVIDENCE)
            generate.assert_called_once()
            search.assert_called_once()

    def test_api_failure_is_not_a_missing_evidence_answer(self):
        from streamlit.testing.v1 import AppTest
        from src.llm_generator import RagGenerationError

        with patch("src.search_hybrid.search_hybrid_all", return_value=([], [], [{"matched_chunks": [{"chunk": EVIDENCE}]}], [], 1)), patch("src.llm_generator.generate_gemini_rag_stream", side_effect=RagGenerationError("API 접근 오류")):
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
            app.text_input(key="rag_query_input").set_value("MOSFET").run()
            app.button(key="rag_btn").click().run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.session_state["rag_result"]["status"], "error")
            self.assertTrue(app.error)

    def test_environment_api_key_reaches_answer_generation(self):
        from streamlit.testing.v1 import AppTest
        from src import config_manager

        def respond(**kwargs):
            kwargs["on_model_used"]("verified-model")
            yield "The paper reports improved gate control."

        papers = [{"matched_chunks": [{"chunk": EVIDENCE}]}]
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(config_manager, "_CONFIG_PATH", Path(folder) / "config.json"), \
                 patch.dict("os.environ", {"GEMINI_API_KEY": "environment-test-key"}), \
                 patch("src.search_hybrid.search_hybrid_all", return_value=([], [], papers, [], 1)), \
                 patch("src.llm_generator.generate_gemini_rag_stream", side_effect=respond) as generate:
                app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
                app.text_input(key="rag_query_input").set_value("MOSFET").run()
                app.button(key="rag_btn").click().run(timeout=30)
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(app.session_state["rag_result"]["status"], "answer")
                self.assertEqual(generate.call_args.kwargs["api_key"], "environment-test-key")

    def test_answer_is_cleared_when_library_revision_changes(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        app.session_state["rag_result"] = {"index_revision": "outdated", "query": "old question", "status": "answer", "text": "old answer"}
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertNotIn("rag_result", app.session_state)

    def test_no_api_call_when_only_scaffolding_is_retrieved(self):
        from streamlit.testing.v1 import AppTest

        papers = [{"matched_chunks": [{"chunk": {"file_name": "device.pdf", "chunk_id": 1, "text": "[Results]\nIV."}}]}]
        with patch("src.search_hybrid.search_hybrid_all", return_value=([], [], papers, [], 1)), patch("src.llm_generator.generate_gemini_rag_stream") as generate:
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
            app.text_input(key="rag_query_input").set_value("MOSFET").run()
            app.button(key="rag_btn").click().run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.session_state["rag_result"]["status"], "no_context")
            generate.assert_not_called()

    def test_partial_stream_failure_does_not_append_another_model_answer(self):
        from src.llm_generator import RagGenerationError, generate_gemini_rag_stream

        def broken_stream(**kwargs):
            yield types.SimpleNamespace(text="Partial answer")
            raise RuntimeError("connection interrupted")
        client = Mock()
        client.models.generate_content_stream.side_effect = broken_stream
        with patch("src.llm_generator.genai.Client", return_value=client):
            stream = generate_gemini_rag_stream("MOSFET", [{"chunk": EVIDENCE}], "test-key")
            self.assertEqual(next(stream), "Partial answer")
            with self.assertRaises(RagGenerationError):
                next(stream)
            client.models.generate_content_stream.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
