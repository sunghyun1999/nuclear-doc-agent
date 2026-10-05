# Nuclear Doc Agent

AI-Assisted Regulatory Review for SMR Safety Analysis.
IAEA SSR-2/1 등 원자력 규제 문서에 대한 RAG 기반 Q&A + Requirement 온톨로지 시각화.

## Architecture

```
[PDF Upload] → [Chunking + Clause Parsing] → [Embedding (local)] → [ChromaDB]
                                                                         ↓
[User Query] → [Embedding] → [Vector Search] → [Cross-Encoder Re-ranking] → [Gemini] → [Answer + Citations + Confidence]
                                                                                                    ↓
                                                                                       [Evaluation Pipeline]

[Requirement Ontology] → [PDF Parsing] → [Domain Knowledge + Semantic Similarity] → [Interactive Graph]
```

## Tech Stack

| Component | Technology | Cost |
|---|---|---|
| Backend | FastAPI (Python) | - |
| LLM | Google Gemini (free tier) | $0 |
| Embedding | sentence-transformers `all-MiniLM-L6-v2` (local) | $0 |
| Re-ranking | Cross-encoder `ms-marco-MiniLM-L-6-v2` (local) | $0 |
| Vector DB | ChromaDB (persistent, local) | $0 |
| Frontend | React + TypeScript + Vite | - |

## Quick Start

### 1. Backend

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

`.env` 파일 생성:
```bash
cp .env.example .env
```

`.env`에 Gemini API 키 입력 (https://aistudio.google.com/apikey 에서 무료 발급):
```
GEMINI_API_KEY=your_key_here
```

서버 실행:
```bash
uvicorn main:app --reload --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

### 3. 브라우저에서 접속

http://localhost:5173

### 4. PDF 업로드

- 사이드바에서 chunking 전략 선택 (Fixed Size / Paragraph)
- "Upload PDF" 클릭 → IAEA SSR-2/1 PDF 업로드
- 업로드 완료 후 바로 질문 가능

> `backend/uploads/` 에 IAEA-SSR-2-1-Rev1.pdf가 이미 있으면 자동으로 인덱싱된 상태.

## Features

### Chat — RAG Q&A
- 자연어 질문 → 규제 문서 기반 답변
- **출처 인용**: 문서명, 페이지 번호, 규제 조항(Requirement XX, §XX) 표시
- **Confidence 배지**: HIGH(초록) / MEDIUM(주황) / LOW(빨강)으로 답변 신뢰도 표시
- **Re-ranking 토글**: Cross-encoder 재순위 ON/OFF 비교 가능
- **Off-topic 거부**: 문서에 없는 내용은 "정보가 없다"고 명시 (hallucination 방지)

### Evaluation — 품질 측정
- 10개 골든 테스트셋 (IAEA SSR-2/1 기반)
- Retrieval Recall: 기대 페이지 검색 비율
- Answer Quality: LLM-as-judge 자동 채점 (0~1)
- Re-ranking Impact: 재순위가 검색 품질을 개선한 비율
- Chunking Comparison: Fixed-size vs Paragraph 전략 비교

### Ontology — Requirement 교차 참조 그래프
- 82개 IAEA SSR-2/1 Requirement를 노드로 시각화
- **Domain edges** (40개): 원자력 도메인 지식 기반 관계 (예: Defence in Depth → DEC)
- **Semantic edges** (21개): 임베딩 유사도 기반 자동 탐지
- 카테고리별 색상 구분 (9개 카테고리)
- 노드 호버 시 연결된 Requirement + 관계 설명 표시
- 카테고리/엣지 타입 필터링

## Design Decisions

### Why Re-ranking?
Bi-encoder 검색은 빠르지만 부정확. Cross-encoder가 (query, chunk) 쌍을 직접 비교해서 정밀도를 높임. 규제 문서는 속도보다 정확도가 중요.

### Why Local Embeddings?
- API 비용 $0
- 데이터가 외부로 나가지 않음 (원자력 문서 보안)
- `all-MiniLM-L6-v2`: 384차원, 좋은 품질

### Why Confidence?
원자력 도메인에서 AI 답변을 그대로 신뢰하면 안 됨. Confidence가 "근거가 얼마나 있는지"를 알려줘서 Human-in-the-loop 의사결정을 도움.

### Why Ontology Graph?
규제 요건은 독립적이지 않고 서로 얽혀 있음. 하나의 설계가 충족해야 할 요건이 8개 이상일 수 있고, 이를 시각화해서 놓치는 요건을 방지.

### Why Evaluation Pipeline?
"잘 되는 것 같다"는 프로덕션에서 통하지 않음. 정량적 메트릭으로 품질을 측정하고, 프롬프트/모델/청킹 변경 시 regression 감지.

## Gemini 모델 참고

- 기본: `gemini-3.1-flash-lite` (무료, 안정적)
- `gemini-3.8-flash`도 사용 가능하나 일일 20회 제한 있음
- `.env`에 `GEMINI_MODEL=모델명`으로 변경 가능

## Project Structure

```
backend/
  main.py          # FastAPI 서버 + API 엔드포인트
  config.py        # 설정 (env 변수)
  ingestion.py     # PDF → 청킹 → 임베딩 → ChromaDB
  chunker.py       # 청킹 전략 + 규제 조항 파싱
  embedder.py      # 로컬 임베딩 (sentence-transformers)
  retriever.py     # 벡터 검색
  reranker.py      # Cross-encoder 재순위
  generator.py     # Gemini 답변 생성 + Confidence
  ontology.py      # Requirement 온톨로지 그래프
  evaluation.py    # 평가 파이프라인
  eval_dataset.json # 골든 테스트셋 10개

frontend/
  src/App.tsx           # 메인 UI (Chat + Evaluation 탭)
  src/OntologyGraph.tsx # Requirement 그래프 시각화
```
