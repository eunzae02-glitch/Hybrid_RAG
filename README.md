# 논문RAG · 반도체 연구 라이브러리

반도체 논문 PDF를 등록하고, 키워드·의미 검색과 출처를 포함한 AI 질의응답을 사용하는 로컬 웹 애플리케이션입니다.

## 실행

GitHub에 공개하는 저장소에는 논문 PDF, 파싱 결과, 벡터 DB, 모델 가중치, API 키, 실행 파일을 포함하지 않습니다. 처음 실행할 때 아래 순서로 환경을 준비하세요.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_models.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

브라우저에서 `http://127.0.0.1:8501`을 열고, 사용 권한이 있는 PDF를 직접 등록하세요. 검색은 로컬에서 실행됩니다. AI 답변을 쓰려면 웹의 API 설정에 개인 Gemini 키를 입력하거나 `GEMINI_API_KEY` 환경 변수를 설정해야 합니다. 저장 키는 `data/user_config.json`에만 기록되며 Git에서 제외됩니다. 공유 컴퓨터에서는 저장하지 말고 환경 변수를 사용하세요.

- 기본 주소: **http://127.0.0.1:8501**
- 기본 포트가 사용 중이면 Streamlit이 안내하는 다른 로컬 주소로 접속하세요.
- 종료하려면 실행 중인 터미널에서 `Ctrl+C`를 누르세요.

Windows 실행 파일을 별도로 만들 때는 `pip install -r requirements-dev.txt` 후 `python scripts/build_launcher.py`를 사용합니다. 이 런처는 완전 무설치 배포용으로 별도 `runtime/` 폴더를 요구하므로 GitHub 소스 실행에는 위 Streamlit 명령을 사용하세요.

## 화면

| 탭 | 기능 |
| --- | --- |
| AI 질의응답 | 논문 근거를 검색해 Gemini로 답변 생성 |
| 검색 비교 | BM25, 벡터 검색, 하이브리드 검색 결과 비교 |
| 논문 관리 | PDF 일괄 등록, 텍스트 추출·OCR, 삭제 |
| 데이터 현황 | 원문·검색 청크 수, 검증 결과 확인 및 전체 색인 재생성 |

질문을 입력하는 것만으로 검색이나 API 호출을 실행하지 않습니다. 각 실행 버튼을 눌러 요청을 제출하세요.

AI 답변은 등록 논문의 본문을 참조합니다. 정의나 동작 원리가 논문에 없으면 그 한계를 밝히고, 확인 가능한 관련 연구 내용을 설명합니다. 근거 부족은 이번 검색 결과의 한계로 표시하며 DB 전체에 정보가 없다고 단정하지 않습니다. 답변에는 실제 응답 모델, 파일명, PDF 페이지와 원문 문맥 ID가 표시됩니다. 설정을 조절해도 답변은 유지되며, 업로드·삭제·재색인으로 논문 색인이 바뀌면 이전 답변을 초기화합니다.

검색은 다운로드한 로컬 임베딩 모델과 로컬 ChromaDB를 사용합니다. 스캔 페이지의 OCR 엔진은 RapidOCR 하나이며 별도 OCR 프로그램 설치가 필요하지 않습니다. 최초 의미 검색이나 스캔 PDF의 최초 OCR은 관련 모델을 읽는 시간이 추가될 수 있습니다. Gemini 답변 생성에는 인터넷 연결과 유효한 API 키가 필요합니다. 생성된 답변의 사실과 수치는 원문 근거로 검토해야 합니다.

현재 RapidOCR 인식 모델은 `ch_PP-OCRv3_rec_infer.onnx`이며 영문 스캔 테스트 PDF로 동작을 확인했습니다. 한글 스캔 문서는 별도 한국어 인식 모델 적용과 품질 검증이 필요합니다. 텍스트 레이어가 있는 한글 PDF는 OCR 없이 PyMuPDF로 추출합니다.

## 청킹과 검색 구조

`페이지별 PDF 블록 → 원문 문맥(parent) → 검색 청크(child) → BM25·벡터 검색 → RRF → 원문 문맥 확장 → 답변`

- PyMuPDF로 PDF 페이지별 텍스트와 블록 좌표를 추출합니다. 텍스트가 부족하고 이미지가 있는 페이지만 RapidOCR로 읽으며, OCR 엔진은 최초 필요 시 한 번 초기화해 재사용합니다. 두 단 문서의 읽기 순서와 제목·저자·반복 머리말·참고문헌을 규칙으로 구분합니다.
- 원문 문맥은 같은 페이지·섹션·종류의 인접 블록을 목표 500토큰까지 묶습니다. 하나의 긴 블록은 원문 보존을 위해 이 길이를 넘을 수 있습니다.
- 검색 청크는 본문 목표 100토큰, 최대 겹침 16토큰입니다. 문장 경계를 우선하고 긴 문장은 나눕니다. 고정 토큰 방식에서도 겹침 설정을 적용합니다.
- 실제 로컬 토크나이저로 **섹션 접두사와 특수 토큰까지 포함한 입력을 128토큰 이하**로 검사합니다. 임베딩 직전에도 검사하여 조용히 잘리는 입력을 차단합니다.
- BM25와 Chroma는 동일한 청크 ID를 사용합니다. 기본 RRF 가중치 **BM25:Vector = 3:7**을 유지하며, 논문 점수는 원문 문맥별 최고 청크만 남겨 중복 가산을 막습니다. 이 가중치가 최적이라는 평가 결과를 의미하지는 않습니다.
- 답변에는 검색 청크의 상위 원문 문맥을 전달합니다. 총 3,000 로컬 토크나이저 토큰 이내이며, 긴 문맥은 검색 위치 주변 일부를 사용하고 화면에 표시합니다. 이는 Gemini의 토큰 수와 동일한 척도가 아닙니다.
- 각 청크에는 파일·페이지·블록 ID·블록 좌표·원문 문자 범위를 저장합니다. 페이지 경계를 넘어 문맥을 합치지 않습니다. 페이지 번호는 PDF의 1부터 시작하는 페이지 순서입니다.

문서 처리 설정을 전체 라이브러리에 적용하려면 **데이터 현황 → 현재 설정으로 전체 색인 재생성**을 누르세요. 업로드는 현재 설정으로 전체 문서의 청크를 구성하고, 동일한 임베딩 입력은 기존 벡터를 재사용합니다. 새 색인이 토큰·겹침·출처·원문 범위·벡터 일치 검증을 통과한 뒤 활성 버전을 한 번에 바꿉니다. 실패하면 기존 색인과 원본 PDF를 유지합니다.

복잡한 표·수식·두 단 PDF의 의미와 읽기 순서는 규칙만으로 완벽하게 복원할 수 없습니다. 좌표와 문자 범위 검증은 추출된 텍스트와의 일치를 검사하며, OCR 정확도나 표 해석을 보증하지 않습니다.

## 로컬 데이터와 실행 기록

다음 파일은 앱 사용 시 로컬에 생기지만 GitHub에는 올라가지 않습니다. 공개 저장소의 `data/`에는 용어 사전과 빈 업로드 폴더만 남습니다.

- `data/uploaded_pdfs/`: 등록한 원본 PDF
- `data/active_index.json`: 현재 사용 중인 색인 버전과 manifest 해시
- `data/indexes/<버전>/documents.json`: 페이지·블록·좌표를 포함한 추출 원문
- `data/indexes/<버전>/parents.json`, `children.json`: 원문 문맥과 검색 청크
- `data/indexes/<버전>/chroma_db/`: 해당 버전의 로컬 벡터 색인
- `data/indexes/<버전>/manifest.json`: 파일 해시, 설정, 모델·파이프라인 지문, 검증 결과
- `data/corpus.json`, `data/chroma_db/`: 이전 형식의 보존본; 공개 저장소에서는 제외
- `data/ontology.json`: 반도체 용어·동의어 사전
- `models/`: 로컬 임베딩 모델
- `logs/server.log`: 서버 실행 및 오류 기록
- `logs/launcher.log`: 시작·접속·종료 기록
- `.local/`: 실행 상태 및 빌드 작업 파일

파일 변경 감시는 비활성화해 AI 라이브러리 탐색 충돌을 방지합니다. 서버 로그는 파일에 직접 기록하며, 첫 화면에서는 검색 모델과 OCR 엔진을 초기화하지 않습니다.

Chroma 연결은 DB 경로별로 재사용하며, 여러 브라우저 세션이 동시에 처음 검색해도 연결 초기화가 겹치지 않도록 생성 과정을 잠금으로 보호합니다.

원문·청크 파일은 내용 해시로 검사합니다. 모델이나 청킹·임베딩 구현이 바뀐 경우 재색인을 안내합니다. 청크 수가 같다는 이유로 이전 벡터를 정상으로 판단하지 않습니다. 이전 색인 버전은 자동으로 지우지 않으므로 반복 재생성 시 디스크 사용량이 늘어납니다.

## 개발 파일

`app.py`와 `src/`가 실제 실행 소스입니다. `launcher.py`는 별도 Windows 배포본의 서버 준비 확인, 중복 실행 방지, 브라우저 열기, 종료 처리를 담당합니다.

런처를 수정한 후 재빌드:

```powershell
python scripts\build_launcher.py
```

회귀 검사:

```powershell
.\.venv\Scripts\python.exe -X utf8 tests\test_config_manager.py
.\.venv\Scripts\python.exe -X utf8 tests\test_startup.py
.\.venv\Scripts\python.exe -X utf8 tests\test_rag_quality.py
.\.venv\Scripts\python.exe -X utf8 tests\test_chunk_pipeline.py
.\.venv\Scripts\python.exe -X utf8 tests\test_pdf_structure.py
.\.venv\Scripts\python.exe -X utf8 tests\test_vector_client.py
```

웹 화면 밖에서 전체 PDF 재색인:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\rebuild_index.py
```

합성 검색 비교 자료는 `scripts/generate_20x20_benchmark.py`로 별도 생성할 수 있습니다. 생성된 PDF는 실제 논문이 아니며 기본 검색 DB에 자동 등록되지 않습니다.
