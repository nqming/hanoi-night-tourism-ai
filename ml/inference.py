"""Thread-safe inference engine for Two-Tower Contrastive Learning POI Retrieval.

Supports hot-updating POI embeddings whenever an administrator creates, edits,
or deletes a Hanoi Night Economy destination via the REST API, as well as strict
intent/category, opening-hour, and budget filtering for grounded RAG & planning.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from data.generate_training_pairs import time_overlap_hours
from ml.features import FeatureExtractor
from ml.model import TwoTowerContrastiveNet

ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT_DIR / "ml" / "checkpoints"


def poi_open_overlap(start_h: float, end_h: float, open_h: float, close_h: float) -> float:
    """Compute overlap hours between user window [start_h, end_h] and POI [open_h, close_h],
    properly handling 24/7 venues (0.0-24.0) and overnight schedules (e.g. 18.0-27.0 or 22.0-29.0).
    """
    if open_h == 0.0 and close_h >= 24.0:
        return max(0.0, end_h - start_h)
    return time_overlap_hours(start_h, end_h, open_h, close_h)


class ContrastiveRecommender:
    def __init__(self, checkpoint_dir: Path = CHECKPOINT_DIR):
        self.checkpoint_dir = checkpoint_dir
        self._lock = threading.RLock()
        self.device = torch.device("cpu")
        self.reload_checkpoints()

    def reload_checkpoints(self) -> None:
        with self._lock:
            self.fe: FeatureExtractor = FeatureExtractor.load(self.checkpoint_dir / "feature_extractor.pkl")
            ckpt = torch.load(
                self.checkpoint_dir / "contrastive_model.pt",
                map_location=self.device,
                weights_only=True,
            )
            self.model = TwoTowerContrastiveNet(
                context_dim=ckpt["context_dim"],
                poi_dim=ckpt["poi_dim"],
                hidden_dim=ckpt["hidden_dim"],
                embed_dim=ckpt["embed_dim"],
            ).to(self.device)
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()

            self._poi_embeddings: Dict[str, np.ndarray] = {}
            emb_path = self.checkpoint_dir / "poi_embeddings.npz"
            if emb_path.exists():
                data = np.load(emb_path, allow_pickle=False)
                for pid, vec in zip(data["poi_ids"].tolist(), data["embeddings"]):
                    self._poi_embeddings[str(pid)] = vec.astype(np.float32)

    def refresh_poi_embeddings(self, pois: List[Dict[str, Any]]) -> None:
        """Re-encode all POIs into the shared Contrastive Embedding Space."""
        if not pois:
            return
        with self._lock:
            poi_feats = self.fe.transform_pois(pois)
            with torch.no_grad():
                vecs = self.model.encode_poi(torch.from_numpy(poi_feats).to(self.device)).cpu().numpy()
            self._poi_embeddings = {
                p["id"]: vec.astype(np.float32) for p, vec in zip(pois, vecs)
            }

    def encode_single_poi(self, poi: Dict[str, Any]) -> np.ndarray:
        with self._lock:
            feat = self.fe.transform_pois([poi])
            with torch.no_grad():
                vec = self.model.encode_poi(torch.from_numpy(feat).to(self.device)).cpu().numpy()[0]
            self._poi_embeddings[poi["id"]] = vec.astype(np.float32)
            return vec

    def remove_poi(self, poi_id: str) -> None:
        with self._lock:
            self._poi_embeddings.pop(poi_id, None)

    def encode_context(self, ctx: Dict[str, Any]) -> np.ndarray:
        feat = self.fe.transform_contexts([ctx])
        with torch.no_grad():
            vec = self.model.encode_context(torch.from_numpy(feat).to(self.device)).cpu().numpy()[0]
        return vec.astype(np.float32)

    def recommend(
        self,
        ctx: Dict[str, Any],
        pois: List[Dict[str, Any]],
        top_k: int = 12,
        enforce_open_window: bool = False,
        only_night_economy: bool = False,
        enforce_categories: bool = False,
        max_cost_vnd: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Score & rank POIs using Two-Tower Contrastive Cosine Similarity + Explainable Reasoning."""
        with self._lock:
            missing = [p for p in pois if p["id"] not in self._poi_embeddings]
            if missing:
                feats = self.fe.transform_pois(missing)
                with torch.no_grad():
                    vecs = self.model.encode_poi(torch.from_numpy(feats).to(self.device)).cpu().numpy()
                for p, v in zip(missing, vecs):
                    self._poi_embeddings[p["id"]] = v.astype(np.float32)

            u = self.encode_context(ctx)
            start_h = float(ctx.get("start_hour", 18.0))
            end_h = float(ctx.get("end_hour", 23.0))
            budget = float(ctx.get("budget_vnd", 500000))
            weather = ctx.get("weather", "clear")
            pref_cats = set(ctx.get("preferred_categories", []))
            pref_dist = ctx.get("preferred_district", "all")

            scored: List[Tuple[float, float, Dict[str, Any], List[str]]] = []
            for p in pois:
                if only_night_economy and not p.get("is_night_economy", False):
                    continue
                if enforce_categories and pref_cats and p.get("category") not in pref_cats:
                    continue

                cost = int(p.get("avg_cost_vnd", 0))
                if max_cost_vnd is not None and cost > max_cost_vnd:
                    continue

                overlap = poi_open_overlap(start_h, end_h, float(p["open_hour"]), float(p["close_hour"]))
                if enforce_open_window and overlap < 0.45:
                    continue

                v = self._poi_embeddings[p["id"]]
                raw_sim = float(np.dot(u, v))

                reasons: List[str] = []
                adj_score = raw_sim
                if overlap >= 0.5:
                    reasons.append("Mở cửa đúng khung giờ")
                else:
                    adj_score -= 0.35
                    reasons.append("Cần lưu ý khung giờ mở cửa")

                if p.get("is_night_economy") and end_h >= 20.0:
                    reasons.append("Điểm nhấn Kinh tế đêm")
                if p.get("category") in pref_cats:
                    adj_score += 0.12
                    reasons.append("Đúng loại hình yêu cầu")
                elif pref_cats:
                    adj_score -= 0.25

                eff_cap = max_cost_vnd if max_cost_vnd is not None else budget * 0.8
                if cost <= eff_cap:
                    reasons.append(f"Trong ngân sách ({cost:,}đ)" if cost > 0 else "Miễn phí")
                else:
                    adj_score -= 0.25

                if weather == "rainy" and p.get("indoor"):
                    reasons.append("Không gian trong nhà tránh mưa")
                if pref_dist != "all" and p.get("district") == pref_dist:
                    adj_score += 0.10
                    reasons.append(f"Tại quận {pref_dist}")

                match_pct = max(15.0, min(99.5, round(((adj_score + 0.35) / 1.15) * 100.0, 1)))
                scored.append((adj_score, match_pct, p, reasons))

            scored.sort(key=lambda x: x[0], reverse=True)

            results: List[Dict[str, Any]] = []
            for adj_score, match_pct, p, reasons in scored[:top_k]:
                results.append(
                    {
                        "poi": p,
                        "cosine_similarity": round(adj_score, 4),
                        "match_percent": match_pct,
                        "reasons": reasons[:4],
                    }
                )
            return results

    def load_metrics(self) -> Dict[str, Any]:
        metrics_path = self.checkpoint_dir / "metrics.json"
        if not metrics_path.exists():
            return {}
        with metrics_path.open("r", encoding="utf-8") as f:
            return json.load(f)
