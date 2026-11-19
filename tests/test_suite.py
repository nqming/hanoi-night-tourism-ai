"""Comprehensive unit and integration tests for Hanoi Night-Tourism AI Platform.

Compatible with both `pytest` (CI/CD) and built-in `python -m unittest`.
"""

from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path

import httpx
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.main import app
from ml.model import TwoTowerContrastiveNet, contrastive_infonce_loss
from ml.optimizer import optimize_itinerary


def _asgi_request(method: str, url: str, **kwargs) -> httpx.Response:
    async def _run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.request(method, url, **kwargs)

    return asyncio.run(_run())


class TestDataIntegrity(unittest.TestCase):
    def test_crawled_pois_schema_and_hanoi_bounds(self) -> None:
        pois_path = ROOT_DIR / "data" / "processed" / "hanoi_pois.json"
        self.assertTrue(pois_path.exists(), "hanoi_pois.json must exist")
        with pois_path.open("r", encoding="utf-8") as f:
            pois = json.load(f)
        self.assertGreaterEqual(len(pois), 200)
        night_count = sum(1 for p in pois if p.get("is_night_economy"))
        self.assertGreater(night_count, 180)
        for p in pois:
            self.assertIn("id", p)
            self.assertIn("name", p)
            self.assertGreaterEqual(p["lat"], 20.90)
            self.assertLessEqual(p["lat"], 21.15)
            self.assertGreaterEqual(p["lon"], 105.70)
            self.assertLessEqual(p["lon"], 105.98)
            self.assertGreater(p["close_hour"], p["open_hour"])


class TestContrastiveModel(unittest.TestCase):
    def test_two_tower_forward_and_infonce_loss(self) -> None:
        torch.manual_seed(0)
        model = TwoTowerContrastiveNet(context_dim=32, poi_dim=28, hidden_dim=64, embed_dim=16)
        ctx_x = torch.randn(16, 32)
        pos_x = torch.randn(16, 28)
        neg_x = torch.randn(16, 28)

        u, v_pos, v_neg, scale = model(ctx_x, pos_x, neg_x)
        self.assertEqual(u.shape, (16, 16))
        self.assertEqual(v_pos.shape, (16, 16))
        self.assertIsNotNone(v_neg)
        assert v_neg is not None
        self.assertEqual(v_neg.shape, (16, 16))

        # Check L2 unit normalization of embeddings
        u_norms = torch.norm(u, dim=-1)
        self.assertTrue(torch.allclose(u_norms, torch.ones_like(u_norms), atol=1e-4))

        loss_initial = contrastive_infonce_loss(u, v_pos, v_neg, scale)
        self.assertGreater(float(loss_initial.item()), 0.0)

        # Verify 5 gradient steps decrease InfoNCE loss on fixed batch
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(5):
            opt.zero_grad()
            u2, vp2, vn2, s2 = model(ctx_x, pos_x, neg_x)
            loss = contrastive_infonce_loss(u2, vp2, vn2, s2)
            loss.backward()
            opt.step()
        self.assertLess(float(loss.item()), float(loss_initial.item()))

    def test_saved_checkpoint_metrics_beat_baselines(self) -> None:
        metrics_path = ROOT_DIR / "ml" / "checkpoints" / "metrics.json"
        self.assertTrue(metrics_path.exists(), "Trained checkpoint metrics.json must exist")
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        bench = metrics["benchmark_comparison"]
        cl_ndcg = bench["two_tower_contrastive_learning"]["ndcg_at_10"]
        tfidf_ndcg = bench["tfidf_content_baseline"]["ndcg_at_10"]
        pop_ndcg = bench["popularity_baseline"]["ndcg_at_10"]
        self.assertGreater(cl_ndcg, tfidf_ndcg)
        self.assertGreater(cl_ndcg, pop_ndcg)


class TestTOPTWOptimizer(unittest.TestCase):
    def test_overnight_time_windows_and_budget_enforced(self) -> None:
        pois_path = ROOT_DIR / "data" / "processed" / "hanoi_pois.json"
        with pois_path.open("r", encoding="utf-8") as f:
            pois = json.load(f)

        # Build mock ranked candidates
        ranked = [
            {
                "poi": p,
                "cosine_similarity": 0.65 if p["is_night_economy"] else 0.1,
                "match_percent": 85.0 if p["is_night_economy"] else 40.0,
                "reasons": ["Test"],
            }
            for p in pois[:35]
        ]

        budget = 400000
        res = optimize_itinerary(
            ranked_candidates=ranked,
            all_pois=pois,
            start_hour=21.0,
            end_hour=26.0,  # 02:00 AM next day!
            budget_vnd=budget,
            max_stops=4,
            transport_mode="motorbike",
        )
        opt = res["optimized"]
        self.assertGreaterEqual(opt["summary"]["stop_count"], 2)
        self.assertLessEqual(opt["summary"]["total_cost_vnd"], budget)
        # Ensure timestamps are strictly monotonic
        prev_dep = 21.0
        for stop in opt["stops"]:
            self.assertGreaterEqual(stop["arrival_hour"], prev_dep - 0.01)
            self.assertGreater(stop["departure_hour"], stop["arrival_hour"])
            prev_dep = stop["departure_hour"]

    def test_no_backtracking_to_start_location(self) -> None:
        """If starting right at Hoan Kiem Lake, any Hoan Kiem Lake stop must be visited
        BEFORE moving outward to Opera House / Ba Dinh (never leave Hoan Kiem and loop back).
        """
        from ml.optimizer import haversine_km

        pois_path = ROOT_DIR / "data" / "processed" / "hanoi_pois.json"
        with pois_path.open("r", encoding="utf-8") as f:
            pois = json.load(f)

        by_id = {p["id"]: p for p in pois}
        # Deliberately feed candidates in reverse distance order (Opera House first, Hoan Kiem second)
        candidates = [
            {"poi": by_id["curated_006"], "cosine_similarity": 0.72, "match_percent": 90.0, "reasons": []},  # Nhà hát Lớn (0.98 km)
            {"poi": by_id["curated_007"], "cosine_similarity": 0.72, "match_percent": 90.0, "reasons": []},  # Phố đi bộ Hồ Gươm (0.28 km)
            {"poi": by_id["curated_005"], "cosine_similarity": 0.71, "match_percent": 89.0, "reasons": []},  # Múa rối Thăng Long (0.15 km)
        ]
        start_lat, start_lon = 21.0313, 105.8520
        res = optimize_itinerary(
            ranked_candidates=candidates,
            all_pois=pois,
            start_hour=18.0,
            end_hour=23.0,
            budget_vnd=800000,
            max_stops=3,
            start_lat=start_lat,
            start_lon=start_lon,
        )
        stops = res["optimized"]["stops"]
        self.assertEqual(len(stops), 3)
        dists = [
            haversine_km(start_lat, start_lon, s["poi"]["lat"], s["poi"]["lon"]) for s in stops
        ]
        # The stop farthest from Hoan Kiem (Opera House ~0.98km) must come AFTER the stops right at Hoan Kiem!
        self.assertLess(dists[0], dists[-1])
        self.assertEqual(stops[-1]["poi"]["id"], "curated_006")


class TestFastAPIEndpoints(unittest.TestCase):
    def test_health_and_frontend(self) -> None:
        r_health = _asgi_request("GET", "/api/health")
        self.assertEqual(r_health.status_code, 200)
        body = r_health.json()
        self.assertEqual(body["status"], "ok")
        self.assertGreaterEqual(body["total_pois"], 40)

        r_home = _asgi_request("GET", "/")
        self.assertEqual(r_home.status_code, 200)
        self.assertIn("Hanoi", r_home.text)

    def test_recommend_and_optimize_endpoints(self) -> None:
        r_rec = _asgi_request(
            "POST",
            "/api/recommend",
            json={
                "query_text": "tour đêm di tích lịch sử và phở đêm phố cổ",
                "start_hour": 18.5,
                "end_hour": 23.0,
                "budget_vnd": 500000,
                "top_k": 5,
                "enforce_open_window": True,
            },
        )
        self.assertEqual(r_rec.status_code, 200)
        recs = r_rec.json()["recommendations"]
        self.assertEqual(len(recs), 5)

        r_opt = _asgi_request(
            "POST",
            "/api/optimize-itinerary",
            json={
                "query_text": "ẩm thực đêm Hoàn Kiếm và phố đi bộ",
                "start_hour": 18.0,
                "end_hour": 23.0,
                "budget_vnd": 450000,
                "max_stops": 4,
                "transport_mode": "walk",
            },
        )
        self.assertEqual(r_opt.status_code, 200)
        opt_body = r_opt.json()
        self.assertIn("optimized", opt_body)
        self.assertIn("baseline", opt_body)
        self.assertIn("comparison", opt_body)
        self.assertGreaterEqual(opt_body["optimized"]["summary"]["stop_count"], 2)

    def test_bilingual_chat_and_metrics(self) -> None:
        r_vi = _asgi_request("POST", "/api/chat", json={"message": "Tối nay ăn đêm ở đâu ngon tại Hoàn Kiếm?"})
        self.assertEqual(r_vi.status_code, 200)
        self.assertEqual(r_vi.json()["language"], "vi")

        # Strict midnight + 50k food scenario reported by user
        r_midnight = _asgi_request(
            "POST",
            "/api/chat",
            json={"message": "sau 0h đêm sau 0h đêm thì đi ăn gì với tài chính 50k"},
        )
        self.assertEqual(r_midnight.status_code, 200)
        m_body = r_midnight.json()
        self.assertEqual(m_body["parsed_context"]["start_hour"], 24.0)
        self.assertEqual(m_body["parsed_context"]["strict_max_cost_vnd"], 50000)
        self.assertIn("00:00 (+1)", m_body["reply"])
        self.assertGreaterEqual(len(m_body["retrieved_pois"]), 2)
        for item in m_body["retrieved_pois"]:
            p = item["poi"]
            self.assertEqual(p["category"], "night_food", f"Non-food POI leaked: {p['name']}")
            self.assertLessEqual(p["avg_cost_vnd"], 50000, f"Over-budget POI leaked: {p['name']}")

        r_en = _asgi_request("POST", "/api/chat", json={"message": "Best night street food and jazz in Hanoi"})
        self.assertEqual(r_en.status_code, 200)
        self.assertEqual(r_en.json()["language"], "en")

        r_metrics = _asgi_request("GET", "/api/metrics")
        self.assertEqual(r_metrics.status_code, 200)
        self.assertIn("model_evaluation", r_metrics.json())

    def test_admin_poi_crud_and_cold_start_embedding(self) -> None:
        # Create a brand-new Night Economy event -> encoded on-the-fly by Contrastive Tower
        r_create = _asgi_request(
            "POST",
            "/api/pois",
            json={
                "name": "Lễ hội Ánh sáng Đêm Hồ Gươm (Test)",
                "name_en": "Hoan Kiem Night Light Festival (Test)",
                "category": "heritage_night",
                "district": "Hoàn Kiếm",
                "lat": 21.0291,
                "lon": 105.8525,
                "open_hour": 19.0,
                "close_hour": 23.0,
                "is_night_economy": True,
                "avg_cost_vnd": 50000,
                "duration_min": 60,
                "description_vi": "Sự kiện trình chiếu ánh sáng nghệ thuật ban đêm tại Hồ Gươm.",
            },
        )
        self.assertEqual(r_create.status_code, 201)
        new_id = r_create.json()["poi"]["id"]

        # Retrieve & update
        r_get = _asgi_request("GET", f"/api/pois/{new_id}")
        self.assertEqual(r_get.status_code, 200)

        r_upd = _asgi_request("PUT", f"/api/pois/{new_id}", json={"avg_cost_vnd": 80000})
        self.assertEqual(r_upd.status_code, 200)
        self.assertEqual(r_upd.json()["poi"]["avg_cost_vnd"], 80000)

        # Clean up test POI
        r_del = _asgi_request("DELETE", f"/api/pois/{new_id}")
        self.assertEqual(r_del.status_code, 200)


if __name__ == "__main__":
    unittest.main()
