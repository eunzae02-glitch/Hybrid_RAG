"""
download_models.py - 오프라인 임베딩 모델 로컬 저장 스크립트
빌드 전 1회만 실행하면 ./models/ 폴더에 모델 가중치가 저장됩니다.
인터넷 없는 환경에서도 임베딩이 정상 동작합니다.
"""
import os
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "paraphrase-multilingual-MiniLM-L12-v2")

def download_embedding_model():
    print(f"[모델 다운로드] '{MODEL_NAME}' 모델을 로컬에 저장합니다...")
    print(f"[저장 경로] {LOCAL_MODEL_DIR}")
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    model = SentenceTransformer(MODEL_NAME)
    model.save(LOCAL_MODEL_DIR)
    print(f"[완료] 모델이 '{LOCAL_MODEL_DIR}'에 저장되었습니다.")
    print("[안내] 이제 인터넷 없이도 임베딩 모델이 로컬에서 로드됩니다.")

if __name__ == "__main__":
    download_embedding_model()
