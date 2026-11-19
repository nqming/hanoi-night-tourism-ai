"""FastAPI application for Hanoi Smart Tourism & Night Economy AI Platform.

Run locally:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.schemas import (
    ChatRequest,
    OptimizeItineraryRequest,
    POICreate,
    POIUpdate,
    RecommendRequest,
)
from backend.storage import POIStorage
from ml.inference import ContrastiveRecommender
from ml.optimizer import optimize_itinerary
from ml.rag_assistant import RAGTourismAssistant

FRONTEND_DIR = ROOT_DIR / "frontend"
CRAWL_REPORT_PATH = ROOT_DIR / "data" / "processed" / "crawl_report.json"

app = FastAPI(
    title="Hanoi Night-Tourism AI Platform",
    description=(
        "Hệ thống du lịch thông minh và kinh tế đêm tại Hà Nội tích hợp "
        "Two-Tower Contrastive Learning (PyTorch), Tối ưu hóa lộ trình có khung thời gian (TOPTW) "
        "và Trợ lý ảo RAG đa ngôn ngữ."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize stateful services
storage = POIStorage()
recommender = ContrastiveRecommender()
assistant = RAGTourismAssistant(recommender)


@app.get("/api/health", tags=["System"])
def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "hanoi-night-tourism-ai",
        "model": "Two-Tower Contrastive Learning (PyTorch InfoNCE)",
        "total_pois": len(storage.list_all()),
        "temperature": round(recommender.model.temperature, 4),
    }


@app.get("/api/pois", tags=["POIs & Night Economy Management"])
def list_pois(
    category: Optional[str] = Query(default=None),
    district: Optional[str] = Query(default=None),
    night_only: bool = Query(default=False),
    q: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=500),
) -> Dict[str, Any]:
    pois = storage.list_all()
    if category and category != "all":
        pois = [p for p in pois if p["category"] == category]
    if district and district != "all":
        pois = [p for p in pois if p["district"] == district]
    if night_only:
        pois = [p for p in pois if p.get("is_night_economy")]
    if q:
        needle = q.strip().lower()
        pois = [
            p
            for p in pois
            if needle in p["name"].lower()
            or needle in p.get("name_en", "").lower()
            or needle in p.get("description_vi", "").lower()
            or needle in p.get("description_en", "").lower()
            or needle in p.get("district", "").lower()
            or any(needle in t.lower() for t in p.get("tags", []))
        ]
    return {"total": len(pois), "items": pois[:limit]}


@app.get("/api/pois/{poi_id}", tags=["POIs & Night Economy Management"])
def get_poi(poi_id: str) -> Dict[str, Any]:
    poi = storage.get(poi_id)
    if not poi:
        raise HTTPException(status_code=404, detail="Không tìm thấy địa điểm.")
    return poi


@app.post("/api/pois", tags=["POIs & Night Economy Management"], status_code=201)
def create_poi(payload: POICreate) -> Dict[str, Any]:
    if payload.close_hour <= payload.open_hour:
        raise HTTPException(status_code=400, detail="Giờ đóng cửa phải lớn hơn giờ mở cửa (với giờ qua đêm, nhập 24 + giờ, ví dụ 26.0 = 02:00 sáng).")
    created = storage.create(payload.model_dump())
    # Immediately project the new POI into the Contrastive Learning embedding space (Zero-Shot Cold Start!)
    recommender.encode_single_poi(created)
    return {"message": "Đã thêm điểm đến và cập nhật vector Contrastive Embedding thành công.", "poi": created}


@app.put("/api/pois/{poi_id}", tags=["POIs & Night Economy Management"])
def update_poi(poi_id: str, payload: POIUpdate) -> Dict[str, Any]:
    updated = storage.update(poi_id, payload.model_dump(exclude_unset=True))
    if not updated:
        raise HTTPException(status_code=404, detail="Không tìm thấy địa điểm để cập nhật.")
    recommender.encode_single_poi(updated)
    return {"message": "Đã cập nhật điểm đến và làm mới vector AI.", "poi": updated}


@app.delete("/api/pois/{poi_id}", tags=["POIs & Night Economy Management"])
def delete_poi(poi_id: str) -> Dict[str, Any]:
    ok = storage.delete(poi_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Không tìm thấy địa điểm để xóa.")
    recommender.remove_poi(poi_id)
    return {"message": "Đã xóa điểm đến khỏi hệ thống.", "deleted_id": poi_id}


@app.post("/api/recommend", tags=["AI Contrastive Recommendation"])
def recommend_pois(req: RecommendRequest) -> Dict[str, Any]:
    if req.end_hour <= req.start_hour:
        raise HTTPException(status_code=400, detail="Giờ kết thúc phải lớn hơn giờ bắt đầu.")
    pois = storage.list_all()
    results = recommender.recommend(
        ctx=req.model_dump(),
        pois=pois,
        top_k=req.top_k,
        enforce_open_window=req.enforce_open_window,
        only_night_economy=req.only_night_economy,
    )
    return {
        "model": "Two-Tower Contrastive Learning (InfoNCE)",
        "query": req.model_dump(),
        "total_returned": len(results),
        "recommendations": results,
    }


@app.post("/api/optimize-itinerary", tags=["TOPTW Itinerary Optimization"])
def optimize_trip(req: OptimizeItineraryRequest) -> Dict[str, Any]:
    if req.end_hour <= req.start_hour:
        raise HTTPException(status_code=400, detail="Giờ kết thúc phải lớn hơn giờ bắt đầu.")
    pois = storage.list_all()
    ranked = recommender.recommend(
        ctx=req.model_dump(),
        pois=pois,
        top_k=40,
        enforce_open_window=False,
        only_night_economy=req.only_night_economy,
    )
    result = optimize_itinerary(
        ranked_candidates=ranked,
        all_pois=pois,
        start_hour=req.start_hour,
        end_hour=req.end_hour,
        budget_vnd=req.budget_vnd,
        max_stops=req.max_stops,
        transport_mode=req.transport_mode,
        start_lat=req.start_lat,
        start_lon=req.start_lon,
        weather=req.weather,
    )
    return result


@app.post("/api/chat", tags=["Bilingual RAG AI Assistant"])
def chat_assistant(req: ChatRequest) -> Dict[str, Any]:
    pois = storage.list_all()
    return assistant.answer(message=req.message, pois=pois, lang=req.lang)


@app.get("/api/metrics", tags=["Analytics & Evaluation"])
def get_metrics() -> Dict[str, Any]:
    crawl_report = {}
    if CRAWL_REPORT_PATH.exists():
        with CRAWL_REPORT_PATH.open("r", encoding="utf-8") as f:
            crawl_report = json.load(f)
    return {
        "model_evaluation": recommender.load_metrics(),
        "dataset_stats": storage.stats(),
        "crawl_report": crawl_report,
    }


# Serve static Frontend assets cleanly
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", include_in_schema=False)
def serve_index() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(str(index_path))
