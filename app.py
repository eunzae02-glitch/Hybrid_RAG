import streamlit as st
import sys
import os

# src 경로 추가
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.storage import (
    ingest_pdf,
    get_index_status,
    load_corpus_db,
    is_file_already_uploaded,
    get_uploaded_papers_info,
    remove_paper_from_db
)
from src.search_bm25 import load_ontology
from src.config_manager import get_api_key, save_api_key, clear_saved_api_key, is_using_custom_key
from src.server_control import get_server_state, request_shutdown
from src.chunk_config import ChunkSettings

# Streamlit 페이지 설정
st.set_page_config(
    page_title="논문RAG | 반도체 연구 라이브러리",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 스타일 적용 (프리미엄 하이브리드 UI)
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .rrf-card-header {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
        padding: 12px 18px;
        border-radius: 12px 12px 0 0;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .rrf-card-body {
        border: 2px solid #7C3AED;
        border-top: none;
        padding: 15px;
        border-radius: 0 0 12px 12px;
        background-color: #F8FAFC;
        margin-bottom: 1rem;
    }
    .badge-basic {
        background-color: #22C55E;
        color: white;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-ocr {
        background-color: #F59E0B;
        color: white;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
""", unsafe_allow_html=True)

# 메인 헤더
st.markdown('<div class="main-header">🤖 반도체 논문 하이브리드 RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">논문을 모으고, 근거를 찾고, 연구 내용을 정리하세요.</div>', unsafe_allow_html=True)

library_chunks = load_corpus_db()
library_papers = len({chunk.get("file_name") for chunk in library_chunks})
overview_papers, overview_chunks, overview_search = st.columns(3)
overview_papers.metric("등록 논문", f"{library_papers:,} 편")
overview_chunks.metric("검색 가능한 청크", f"{len(library_chunks):,} 개")
overview_search.metric("검색 방식", "하이브리드")
st.divider()

# 사이드바
with st.sidebar:
    st.header("연구 라이브러리")
    st.caption("반도체 논문 검색 · 근거 기반 요약")

    with st.expander("🔑 Gemini API 설정"):
    
        # Show where the key comes from without displaying the secret itself.
        _using_custom = is_using_custom_key()
        if _using_custom:
            st.success("✅ 로컬에 저장한 개인 키 사용 중")
        elif os.environ.get("GEMINI_API_KEY", "").strip():
            st.info("✅ GEMINI_API_KEY 환경 변수 사용 중")
        else:
            st.info("🔑 API 키를 설정하면 AI 답변을 생성할 수 있습니다.")
    
        gemini_api_key = st.text_input(
            "Google API Key",
            type="password",
            value="",
            help="개인 키를 입력해 저장하면 이 PC의 data/user_config.json에 보관됩니다. 이 파일은 Git에서 제외됩니다.",
            key="api_key_input"
        )
    
        _col_save, _col_reset = st.columns(2)
        with _col_save:
            if st.button("💾 저장", use_container_width=True, key="btn_save_key"):
                if gemini_api_key.strip():
                    save_api_key(gemini_api_key.strip())
                    st.success("개인 키가 저장되었습니다!")
                    st.rerun()
                else:
                    st.warning("키를 입력해 주세요.")
        with _col_reset:
            if st.button("🗑️ 저장 키 삭제", use_container_width=True, key="btn_reset_key"):
                clear_saved_api_key()
                st.success("로컬에 저장한 키를 삭제했습니다.")
                st.rerun()


    selected_gemini_model = st.selectbox(
        "🤖 Gemini 모델 선택",
        options=["gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite"],
        index=0,
        help="RAG 답변 생성에 사용할 Gemini 모델을 선택합니다."
    )
    
    with st.expander("⚙️ 문서 처리 설정"):
        active_index = get_index_status()
        active_settings = ChunkSettings.from_dict(active_index.get("settings"))
        min_len = st.number_input("페이지별 OCR 전환 기준 (글자)", min_value=10, max_value=500, value=50)
        chunk_mode_sel = st.radio(
            "검색 청크 분할 방식",
            options=["문장 경계 우선", "고정 토큰 길이"],
            index=0 if active_settings.mode == "smart" else 1,
            help="두 방식 모두 페이지 출처와 임베딩 입력 한도를 보존합니다. 긴 문장은 토큰 경계에서 나눌 수 있습니다."
        )
        child_tokens = st.number_input("검색 청크 목표 길이 (토큰)", min_value=8, max_value=126, value=active_settings.child_target_tokens, step=1)
        overlap_tokens = st.number_input("검색 청크 최대 겹침 (토큰)", min_value=0, max_value=child_tokens - 1, value=min(active_settings.overlap_tokens, child_tokens - 1), step=1)
        parent_tokens = st.number_input("답변용 원문 문맥 목표 길이 (토큰)", min_value=child_tokens, value=max(active_settings.parent_target_tokens, child_tokens), step=50)
        chunk_settings = ChunkSettings(child_target_tokens=child_tokens, overlap_tokens=overlap_tokens, parent_target_tokens=parent_tokens, mode="smart" if chunk_mode_sel == "문장 경계 우선" else "fixed", section_prefix_tokens=active_settings.section_prefix_tokens)
        st.caption("섹션 접두사와 특수 토큰을 포함한 최종 임베딩 입력은 128토큰 이하로 검사합니다.")
        if chunk_settings.to_dict() != active_settings.to_dict():
            st.caption("설정은 다음 업로드에 적용됩니다. 전체 논문에 적용하려면 '데이터 현황'에서 색인을 재생성하세요.")

    with st.expander("⚖️ 검색 가중치 설정"):

        st.caption("BM25와 Vector DB의 결합 비율을 실시간으로 조절합니다.")
    
        # 세션 상태 초기화 및 콜백 함수 정의
        if "vec_weight_val" not in st.session_state:
            st.session_state["vec_weight_val"] = 0.7

        def set_preset_vec_weight(val: float):
            st.session_state["vec_weight_val"] = val
            if "sidebar_vec_slider" in st.session_state:
                st.session_state["sidebar_vec_slider"] = val

        def on_slider_change():
            if "sidebar_vec_slider" in st.session_state:
                st.session_state["vec_weight_val"] = st.session_state["sidebar_vec_slider"]
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.button("⚡ 기본 0.3:0.7", use_container_width=True, key="sb_p1", on_click=set_preset_vec_weight, args=(0.7,))
        with col_p2:
            st.button("⚖️ 균등 0.5:0.5", use_container_width=True, key="sb_p2", on_click=set_preset_vec_weight, args=(0.5,))
            
        col_p3, col_p4 = st.columns(2)
        with col_p3:
            st.button("🔍 키워드 0.7:0.3", use_container_width=True, key="sb_p3", on_click=set_preset_vec_weight, args=(0.3,))
        with col_p4:
            st.button("🧬 시맨틱 0.0:1.0", use_container_width=True, key="sb_p4", on_click=set_preset_vec_weight, args=(1.0,))

        vec_weight = st.slider(
            "Vector DB 가중치 (0.0 ~ 1.0)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state["vec_weight_val"],
            step=0.05,
            key="sidebar_vec_slider",
            on_change=on_slider_change
        )
        st.session_state["vec_weight_val"] = vec_weight
        bm25_weight = round(1.0 - vec_weight, 2)
    
        st.markdown(f"""
        <div style="background:#1E293B; padding:10px; border-radius:8px; margin-top:5px; border-left:4px solid #7C3AED;">
            <span style="color:#94A3B8; font-size:0.85rem;">현재 RRF 결합 공식:</span><br>
            <code style="color:#A78BFA; font-size:0.85rem;">RRF = {bm25_weight:.2f}/(60+r_BM) + {vec_weight:.2f}/(60+r_Vec)</code>
        </div>
        """, unsafe_allow_html=True)


    st.markdown("---")
    st.caption("반도체 용어·동의어 사전")
    ontology_data = load_ontology()
    st.caption(f"등록된 키워드/동의어: 총 {len(ontology_data)}개 파라미터")
    with st.expander("📚 용어 사전 보기"):
        st.json(ontology_data)

    if get_server_state().get("status") == "ready":
        st.divider()
        st.caption("내 컴퓨터에서 실행 중인 연구 라이브러리")
        if st.button("로컬 서버 종료", key="stop_local_server", use_container_width=True):
            if request_shutdown():
                st.info("서버를 종료합니다. 다시 사용하려면 실행 파일을 열어 주세요.")
                st.stop()

# 메인 탭 4개 구성
main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
    "🤖 AI 질의응답", 
    "⚡ 검색 비교", 
    "📄 논문 관리", 
    "💾 데이터 현황"
])

# ==========================================
# TAB 1: Gemini AI RAG 검색 (순수 AI 답변 전용)
# ==========================================
with main_tab1:
    from src.rag_ui import render_rag_tab

    render_rag_tab(get_api_key(), selected_gemini_model, bm25_weight, vec_weight)

# ==========================================
# TAB 2: 3분할 랭킹 비교 (LLM 미사용 디버그 전용)
# ==========================================
with main_tab2:
    st.subheader("⚡ 3분할 검색 엔진 랭킹 비교 디버그 (LLM 미사용)")
    st.caption("LLM 생성 없이 [BM25 키워드], [Vector DB 의미], [RRF 하이브리드 결합] 랭킹을 3분할 화면으로 순수 검색 품질만 초고속 검증합니다.")
    
    # 상단 인터랙티브 RRF 가중치 프리셋 바 (콜백 연동)
    col_w_info, col_w_btn1, col_w_btn2, col_w_btn3, col_w_btn4 = st.columns([3, 1, 1, 1, 1])
    with col_w_info:
        st.markdown(f"**⚖️ 현재 RRF 가중치**: <span style='color:#7C3AED;'><b>BM25 {bm25_weight:.2f}</b></span> ✕ <span style='color:#2563EB;'><b>Vector {vec_weight:.2f}</b></span>", unsafe_allow_html=True)
    with col_w_btn1:
        st.button("⚡ 0.3 : 0.7", key="t2_btn_rec", use_container_width=True, on_click=set_preset_vec_weight, args=(0.7,))
    with col_w_btn2:
        st.button("⚖️ 0.5 : 0.5", key="t2_btn_eq", use_container_width=True, on_click=set_preset_vec_weight, args=(0.5,))
    with col_w_btn3:
        st.button("🔍 0.7 : 0.3", key="t2_btn_kw", use_container_width=True, on_click=set_preset_vec_weight, args=(0.3,))
    with col_w_btn4:
        st.button("🧬 0.0 : 1.0", key="t2_btn_sem", use_container_width=True, on_click=set_preset_vec_weight, args=(1.0,))

            
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    
    debug_query = st.text_input(
        "디버깅할 검색어 또는 키워드를 입력하세요", 
        placeholder="예: EUV 극자외선 노광 기술이나 GAA 핀펫 대안 구조, DRAM 비트라인 스케일링",
        key="debug_query_input"
    )
    
    col_debug_btn, _ = st.columns([1, 4])
    with col_debug_btn:
        debug_clicked = st.button("⚡ 랭킹 디버깅 실행", type="primary", use_container_width=True, key="debug_btn")
        
    if debug_clicked:
        if not debug_query.strip():
            st.warning("검색어를 입력해 주세요.")
        else:
            with st.spinner("3가지 검색 엔진 랭킹 계산 중..."):
                from src.search_hybrid import search_hybrid_all

                bm25_papers, vector_papers, rrf_papers, synonyms_found, total_chunks = search_hybrid_all(
                    debug_query, 
                    top_k=3,
                    w_bm25=bm25_weight,
                    w_vec=vec_weight
                )
                
            if total_chunks == 0:
                st.warning("현재 등록된 논문이 없습니다. '📄 논문 관리' 탭에서 PDF를 먼저 등록해 주세요.")
            else:
                if synonyms_found:
                    st.info(f"💡 **반도체 온톨로지 사전 감지 및 쿼리 확장**: { ' | '.join(synonyms_found) }")
                    
                st.markdown("---")
                
                # 3분할 열 구성 (columns(3))
                col_bm, col_vec, col_rrf = st.columns(3)
                
                # ----------------------------------------------------
                # [왼쪽 열]: BM25 키워드 단독 검색 Top 3
                # ----------------------------------------------------
                with col_bm:
                    st.markdown("### 🔍 [1] BM25 키워드 단독")
                    st.caption("정확한 용어/약어 키워드 매칭 중심")
                    st.markdown("---")
                    
                    if not bm25_papers:
                        st.write("매칭된 키워드 결과가 없습니다.")
                    else:
                        for p_idx, paper in enumerate(bm25_papers, start=1):
                            file_name = paper["file_name"]
                            final_s = paper.get("final_score", paper["max_score"])
                            max_s = paper["max_score"]
                            matched_chunks = paper["matched_chunks"]
                            
                            st.markdown(f"**Rank {p_idx}. {file_name}**")
                            st.caption(f"🏆 BM25 점수: **{final_s:.4f}** (최고: {max_s:.4f})")
                            
                            with st.expander(f"BM25 청크 보기 ({len(matched_chunks)}개)", expanded=(p_idx == 1)):
                                for c_idx, item in enumerate(matched_chunks[:2], start=1):
                                    c = item['chunk']
                                    sec = c.get("section_title", "본문")
                                    p_n = c.get("page_num", 1)
                                    st.write(f"**Chunk #{c.get('chunk_id')} (📌 `{sec}` | 📄 p.{p_n})** (Score: `{item['score']:.4f}`)")
                                    st.caption(c.get('text', '')[:200] + "...")
                                    st.markdown("---")
                            st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

                # ----------------------------------------------------
                # [가운데 열]: Vector DB 의미 단독 검색 Top 3
                # ----------------------------------------------------
                with col_vec:
                    st.markdown("### 🧬 [2] Vector DB 의미 단독")
                    st.caption("ChromaDB 임베딩 코사인 유사도 중심")
                    st.markdown("---")
                    
                    if not vector_papers:
                        st.write("유사한 의미의 결과가 없습니다.")
                    else:
                        for p_idx, paper in enumerate(vector_papers, start=1):
                            file_name = paper["file_name"]
                            final_v = paper.get("final_vector_score", paper["max_similarity"])
                            max_sim = paper["max_similarity"]
                            matched_chunks = paper["matched_chunks"]
                            
                            st.markdown(f"**Rank {p_idx}. {file_name}**")
                            st.caption(f"🏆 Vector 점수: **{final_v:.4f}** (최고 유사도: {max_sim:.4f})")
                            
                            with st.expander(f"Vector 청크 보기 ({len(matched_chunks)}개)", expanded=(p_idx == 1)):
                                for c_idx, item in enumerate(matched_chunks[:2], start=1):
                                    c = item['chunk']
                                    sec = c.get("section_title", "본문")
                                    p_n = c.get("page_num", 1)
                                    st.write(f"**Chunk #{c.get('chunk_id')} (📌 `{sec}` | 📄 p.{p_n})** (Sim: `{item['similarity']:.4f}`)")
                                    st.caption(c.get('text', '')[:200] + "...")
                                    st.markdown("---")
                            st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

                # ----------------------------------------------------
                # [오른쪽 열]: RRF 하이브리드 결합 Top 3 (강조)
                # ----------------------------------------------------
                with col_rrf:
                    st.markdown("### ⚡ [3] RRF 하이브리드 결합")
                    st.caption(f"RRF(d) = {bm25_weight:.2f}/(60+r_BM) + {vec_weight:.2f}/(60+r_Vec) · 같은 원문 문맥 중복 제외")
                    st.markdown("---")

                    
                    if not rrf_papers:
                        st.write("하이브리드 결과가 없습니다.")
                    else:
                        for p_idx, paper in enumerate(rrf_papers, start=1):
                            file_name = paper["file_name"]
                            sum_rrf = paper.get("sum_rrf_score", paper.get("max_rrf_score", 0.0))
                            matched_chunks = paper["matched_chunks"]
                            top_chunk_info = matched_chunks[0] if matched_chunks else {}
                            
                            b_rank = top_chunk_info.get("bm25_rank", "-")
                            v_rank = top_chunk_info.get("vector_rank", "-")
                            
                            st.markdown(f"""
                                <div class="rrf-card-header">
                                    🏆 Rank {p_idx}. {file_name}
                                </div>
                                <div class="rrf-card-body">
                                    <strong>⚡ RRF 누적 결합 점수: <span style="color:#7C3AED; font-size:1.2rem;">{sum_rrf:.6f}</span></strong><br>
                                    <small style="color:#64748B;">(최고 청크 랭킹: BM25 {b_rank}위 | Vector {v_rank}위)</small>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            with st.expander(f"⚡ RRF 청크 상세 보기 (총 {len(matched_chunks)}개 매칭)", expanded=(p_idx == 1)):
                                for c_idx, item in enumerate(matched_chunks[:2], start=1):
                                    c = item['chunk']
                                    sec = c.get("section_title", "본문")
                                    p_n = c.get("page_num", 1)
                                    st.write(f"**Chunk #{c.get('chunk_id')} (📌 `{sec}` | 📄 p.{p_n})** (RRF: `{item['rrf_score']:.6f}`)")
                                    st.caption(f"BM25 #{item.get('bm25_rank', '-')}위 (점수: {item.get('bm25_score', 0):.2f}) | Vector #{item.get('vector_rank', '-')}위 (유사도: {item.get('vector_sim', 0):.4f})")
                                    st.text_area(
                                        label=f"RRF Chunk #{c.get('chunk_id')} ({sec})",
                                        value=c.get('text', ''),
                                        height=120,
                                        key=f"debug_rrf_txt_{p_idx}_{c.get('chunk_id')}_{c_idx}"
                                    )
                                    st.markdown("---")


# ==========================================
# TAB 3: 논문 업로드 & 파싱 (다중 파일 지원)
# ==========================================
with main_tab3:
    st.subheader("📄 반도체 논문 PDF 다중 업로드 & 파싱")
    st.caption("여러 편의 논문 PDF를 동시에 선택(Ctrl 또는 Shift 다중 선택)하여 일괄 파싱 및 DB 인덱싱을 수행할 수 있습니다. (최대 10개 이상 동시 지원)")
    
    uploaded_files = st.file_uploader(
        "검증할 PDF 논문 파일 다중 선택 (.pdf)", 
        type=["pdf"],
        accept_multiple_files=True,
        key="multi_pdf_uploader"
    )

    if uploaded_files:
        st.info(f"📂 **선택된 파일**: 총 {len(uploaded_files)}편의 PDF 문서가 등록 준비되었습니다.")
        
        # 일괄 덮어쓰기 옵션
        overwrite_all = st.checkbox("이미 존재하는 중복 논문 발견 시 자동으로 최신 내용으로 덮어쓰기 재처리", value=True)
        
        if st.button(f"🚀 선택된 {len(uploaded_files)}개 논문 일괄 파싱 & DB 저장 시작", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            success_count = 0
            total_new_chunks = 0
            
            results_summary = []
            
            for f_idx, uploaded_file in enumerate(uploaded_files, start=1):
                file_name = uploaded_file.name
                file_bytes = uploaded_file.getvalue()
                file_size_kb = len(file_bytes) / 1024
                
                status_text.markdown(f"**[{f_idx}/{len(uploaded_files)}]** `{file_name}` 파싱 및 인덱싱 처리 중...")
                
                already_exists = is_file_already_uploaded(file_name)
                if already_exists and not overwrite_all:
                    results_summary.append({"file": file_name, "status": "건너뜀 (중복 파일)", "chunks": 0})
                    progress_bar.progress(f_idx / len(uploaded_files))
                    continue
                try:
                    new_chunks = ingest_pdf(file_name, file_bytes, settings=chunk_settings, min_text_len_per_page=min_len, overwrite=overwrite_all)
                    success_count += 1
                    total_new_chunks += len(new_chunks)
                    results_summary.append({"file": file_name, "status": "등록 및 색인 검증 완료", "chunks": len(new_chunks)})
                except Exception as error:
                    results_summary.append({"file": file_name, "status": f"등록 실패: {error}", "chunks": 0})

                progress_bar.progress(f_idx / len(uploaded_files))
                
            status_text.empty()
            progress_bar.empty()
            
            st.session_state["upload_report"] = {"success_count": success_count, "chunks": total_new_chunks, "results": results_summary}
            st.rerun()

    upload_report = st.session_state.get("upload_report")
    if upload_report:
        if upload_report["success_count"]:
            st.success(f"논문 {upload_report['success_count']}편, 검색 청크 {upload_report['chunks']}개를 반영했습니다.")
        else:
            st.info("새로 반영된 논문이 없습니다. 아래 처리 결과를 확인해 주세요.")
        with st.expander("📤 최근 업로드 처리 결과", expanded=True):
            st.table(upload_report["results"])


    st.divider()

    st.subheader("📚 업로드된 논문 목록 기록")
    papers_info = get_uploaded_papers_info()

    
    if papers_info:
        st.markdown(f"현재 시스템에 총 **{len(papers_info)}편**의 논문이 등록되어 있습니다.")
        for paper in papers_info:
            with st.expander(f"📖 {paper['file_name']} ({paper['size_kb']:.1f} KB, 청크 {paper['chunk_count']}개)"):
                col_a, col_b, col_c = st.columns([2, 2, 1])
                with col_a:
                    st.write(f"**파일명**: `{paper['file_name']}`")
                with col_b:
                    st.write(f"**파싱 방식**: `{paper['parse_method']}`")
                with col_c:
                    if st.button("🗑️ 삭제", key=f"del_{paper['file_name']}"):
                        try:
                            with st.spinner("논문을 제외한 색인을 검증하고 있습니다..."):
                                remove_paper_from_db(paper['file_name'])
                            st.session_state.pop("upload_report", None)
                            st.rerun()
                        except Exception as error:
                            st.error(f"삭제하지 못했습니다. 기존 데이터는 유지됩니다. {error}")
    else:
        st.info("아직 등록된 논문이 없습니다. 상단에서 PDF 파일을 업로드해 보세요.")

# ==========================================
# TAB 4: DB 현황 확인
# ==========================================
with main_tab4:
    st.subheader("📊 논문 색인 현황")
    index_status = get_index_status()
    report = index_status.get("validation", {})
    has_papers = bool(get_uploaded_papers_info())
    columns = st.columns(4)
    columns[0].metric("등록 논문", f"{len(get_uploaded_papers_info()):,} 편")
    columns[1].metric("원문 문맥", f"{report.get('parents', 0):,} 개")
    columns[2].metric("검색 청크", f"{len(load_corpus_db()):,} 개")
    columns[3].metric("최대 임베딩 입력", f"{report.get('max_input_tokens', 0)} / {report.get('model_limit_tokens', 128)} 토큰")
    if not has_papers:
        st.info("아직 등록된 논문이 없습니다. '논문 관리'에서 PDF를 업로드하면 검색 색인이 생성됩니다.")
    elif index_status.get("format_version") == 2:
        st.success("토큰 한도, 페이지 출처, 원문 범위 및 벡터 일치 검사를 통과한 색인을 사용 중입니다.")
    else:
        st.warning("기존 색인입니다. 아래 버튼으로 토큰 기반 색인을 생성해 주세요.")
    if has_papers:
        st.caption("현재 문서 처리 설정으로 원본 PDF를 다시 읽습니다. 새 색인이 검증된 후 전환하며 이전 버전은 보존됩니다.")
        if st.button("현재 설정으로 전체 색인 재생성", key="rebuild_index"):
            from src.index_builder import rebuild_library
            status = st.empty()
            try:
                with st.spinner("논문을 다시 읽고 새 색인을 검증하고 있습니다..."):
                    rebuilt = rebuild_library(settings=chunk_settings, min_text_len_per_page=min_len, progress=status.write)
                st.session_state.pop("rag_result", None)
                st.rerun()
            except Exception as error:
                st.error(f"재색인에 실패했습니다. 기존 색인은 유지됩니다. {error}")
    with st.expander("색인 검증 결과 및 설정"):
        st.json({"revision": index_status.get("revision"), "settings": index_status.get("settings", {}), "validation": report})
    with st.expander("검색 청크 예시"):
        st.json(load_corpus_db()[:2])
