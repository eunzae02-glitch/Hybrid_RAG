"""Paper-grounded Q&A with persistent results and explicit evidence scope."""

import logging

import streamlit as st

from src.retrieval_quality import is_searchable_chunk
from src.index_repository import IndexIntegrityError, IndexRepository


def select_context(papers: list[dict], max_papers: int = 4, chunks_per_paper: int = 3, max_context_tokens=3000, snapshot=None) -> list[dict]:
    selected = []
    seen = set()
    revision = next((paper.get("index_revision") for paper in papers if paper.get("index_revision")), None)
    if snapshot is None and revision and revision != "legacy":
        snapshot = IndexRepository().load_revision(revision)
    parents = snapshot.parent_map if snapshot is not None else {}
    remaining = max_context_tokens
    # Round-robin across papers so one long paper cannot consume all context.
    for level in range(chunks_per_paper):
        for paper in papers[:max_papers]:
            items = paper.get("matched_chunks", [])
            if level >= len(items):
                continue
            item = items[level]
            chunk = item["chunk"]
            key = (chunk.get("file_name"), chunk.get("parent_id", chunk.get("chunk_id")))
            if key in seen or not is_searchable_chunk(chunk):
                continue
            if "parent_id" in chunk and "searchable" in chunk:
                parent = parents.get(chunk["parent_id"])
                if parent is None:
                    raise IndexIntegrityError("검색 청크의 원문 문맥을 찾지 못했습니다. 색인을 재생성해 주세요.")
                if remaining < max(64, chunk["content_token_count"]):
                    continue
                from src.chunker_smart import source_spans_for_child
                start, end = 0, len(parent["text"])
                token_count = parent["token_count"]
                if token_count > min(remaining, 1000):
                    from src.token_budget import get_token_budget
                    budget = get_token_budget()
                    offsets = budget.offsets(parent["text"])
                    allowance = min(remaining, 1000) - 2
                    first = max(0, chunk["parent_token_start"] - allowance // 3)
                    last = min(len(offsets), first + allowance)
                    if last < chunk["parent_token_end"]:
                        last = chunk["parent_token_end"]
                        first = max(0, last - allowance)
                    start, end = offsets[first][0], offsets[last - 1][1]
                    token_count = budget.count(parent["text"][start:end], special=False)
                context = {**parent, "chunk_id": parent["parent_id"], "text": parent["text"][start:end], "context_type": "parent", "context_partial": start != 0 or end != len(parent["text"]), "source_spans": source_spans_for_child(parent, start, end), "matched_child_id": chunk["child_id"], "token_count": token_count}
                selected.append({**item, "chunk": context, "retrieved_child": chunk})
                remaining -= token_count
            else:
                selected.append(item)  # compatibility for unversioned test/legacy evidence
            seen.add(key)
    return selected


def render_result(result: dict):
    st.markdown("### 답변 및 논문 근거")
    st.caption(f"질문: {result['query']}")
    if result.get("model"):
        st.caption(f"응답 모델: {result['model']}")
    status = result["status"]
    if status == "error":
        st.error(result["text"])
    elif status == "empty_library":
        st.warning("현재 등록된 논문이 없습니다. '논문 관리' 탭에서 PDF를 먼저 등록해 주세요.")
    elif status == "no_context":
        st.info("이번 검색에서 답변 근거로 사용할 본문을 찾지 못했습니다. 질문을 구체화하거나 관련 논문을 등록해 주세요.")
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(result["text"])
        if status == "insufficient":
            st.info("이번 검색에서 전달한 본문만으로 충분한 답변을 만들지 못했습니다. 아래 근거를 확인하거나 질문을 더 구체적으로 입력해 주세요.")

    chunks = result.get("chunks", [])
    if chunks:
        with st.expander(f"📄 답변에 사용한 원문 문맥 {len(chunks)}개 확인"):
            for index, item in enumerate(chunks, start=1):
                chunk = item["chunk"]
                st.markdown(f"**[{index}] {chunk['file_name']}**")
                label = "문맥" if chunk.get("context_type") == "parent" else "청크"
                st.caption(f"{label} #{chunk['chunk_id']} · p.{chunk.get('page_num', '?')} · {chunk.get('section_title', '본문')}")
                if chunk.get("matched_child_id"):
                    st.caption(f"검색 청크: {chunk['matched_child_id']} · 원문 블록 {len(chunk.get('source_spans', []))}개")
                if chunk.get("context_partial"):
                    st.caption("긴 원문 중 검색 위치 주변의 일부를 전달했습니다.")
                st.write(chunk["text"])
                st.divider()


def render_rag_tab(api_key: str, model_name: str, w_bm25: float, w_vec: float):
    current_revision = IndexRepository().load().revision
    previous_result = st.session_state.get("rag_result", {})
    if previous_result.get("index_revision", current_revision) != current_revision:
        st.session_state.pop("rag_result", None)
        st.info("논문 색인이 변경되어 이전 답변을 초기화했습니다. 질문을 다시 실행해 주세요.")
    st.subheader("🤖 논문 기반 질의응답")
    st.caption("등록된 논문의 본문을 근거로 답변합니다. 논문에 없는 일반적인 정의나 원리는 설명 범위가 제한될 수 있습니다.")
    query = st.text_input(
        "질문할 반도체 키워드 또는 기술 문장을 입력하세요",
        placeholder="예: GAA 소자의 신뢰성에 대해 등록 논문에서 확인한 내용을 알려줘",
        key="rag_query_input",
    )
    submitted = st.button("🤖 Gemini AI 답변 생성", type="primary", key="rag_btn")
    if submitted:
        if not query.strip():
            st.warning("검색어를 입력해 주세요.")
        else:
            result = {"query": query.strip(), "chunks": [], "model": None, "index_revision": current_revision}
            try:
                with st.spinner("질문과 관련된 논문 본문을 찾고 있습니다..."):
                    from src.search_hybrid import search_hybrid_all

                    _, _, papers, _, total_chunks = search_hybrid_all(query, top_k=4, w_bm25=w_bm25, w_vec=w_vec)
                    result["index_revision"] = next((p["index_revision"] for p in papers if p.get("index_revision")), current_revision)
                    result["chunks"] = select_context(papers)
                if total_chunks == 0:
                    result["status"] = "empty_library"
                elif not result["chunks"]:
                    result["status"] = "no_context"
                else:
                    from src.llm_generator import generate_gemini_rag_stream, is_insufficient_response

                    def record_model(name):
                        result["model"] = name

                    live_answer = st.empty()
                    try:
                        with live_answer.container():
                            with st.chat_message("assistant", avatar="🤖"):
                                result["text"] = st.write_stream(generate_gemini_rag_stream(
                                    query=query, retrieved_chunks=result["chunks"], api_key=api_key,
                                    model_name=model_name, on_model_used=record_model,
                                ))
                    finally:
                        live_answer.empty()
                    result["status"] = "insufficient" if is_insufficient_response(result["text"]) else "answer"
            except Exception as error:
                from src.llm_generator import RagGenerationError

                result["status"] = "error"
                if isinstance(error, (RagGenerationError, IndexIntegrityError)):
                    result["text"] = str(error)
                else:
                    logging.exception("Paper Q&A failed")
                    result["text"] = "논문 검색 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
            st.session_state["rag_result"] = result

    if "rag_result" in st.session_state:
        render_result(st.session_state["rag_result"])
