"""Feature extraction for User-Context and Hanoi Night-Economy POIs.

Combines:
1. Bilingual Vietnamese + English semantic text projection (Word + Char-wb TF-IDF -> TruncatedSVD).
2. Domain-specific structured features:
   - Circular time encodings (sin/cos of open/close/start/end hours across 24h cycle)
   - Overnight indicator & night-span ratio (18:00 - 06:00)
   - Log-scaled VND budget / average cost
   - Category multi-hot / one-hot, district one-hot, indoor/outdoor, weather, companion, GPS coords.
"""

from __future__ import annotations

import math
import pickle
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

CATEGORIES = [
    "heritage_night",
    "culture_art",
    "walking_market",
    "night_food",
    "nightlife_cafe",
    "scenic_view",
    "daytime_heritage",
]

DISTRICTS = [
    "Hoàn Kiếm",
    "Ba Đình",
    "Tây Hồ",
    "Đống Đa",
    "Hai Bà Trưng",
    "Cầu Giấy",
]

COMPANIONS = ["solo", "couple", "friends", "family"]
WEATHERS = ["clear", "cool", "rainy"]

# Hanoi reference center (Hoan Kiem Lake) for normalized coordinates
HANOI_LAT_CENTER = 21.0288
HANOI_LON_CENTER = 105.8522


def strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch)).replace("đ", "d").replace("Đ", "D")


def normalize_bilingual_text(text: str) -> str:
    """Keep both accented Vietnamese and ASCII-folded text so unaccented queries match cleanly."""
    clean = " ".join(str(text).lower().split())
    folded = strip_accents(clean)
    return f"{clean} {folded}" if folded != clean else clean


def poi_to_text(poi: Dict[str, Any]) -> str:
    parts = [
        poi.get("name", ""),
        poi.get("name_en", ""),
        poi.get("category", "").replace("_", " "),
        poi.get("district", ""),
        " ".join(poi.get("tags", [])),
        poi.get("description_vi", ""),
        poi.get("description_en", ""),
        poi.get("wiki_excerpt_vi", ""),
        poi.get("wiki_excerpt_en", ""),
    ]
    if poi.get("is_night_economy"):
        parts.append("kinh tế đêm du lịch đêm mở cửa tối night economy evening")
    if poi.get("indoor"):
        parts.append("trong nhà tránh mưa indoor cozy")
    else:
        parts.append("ngoài trời thoáng mát outdoor open air")
    return normalize_bilingual_text(" ".join(p for p in parts if p))


def context_to_text(ctx: Dict[str, Any]) -> str:
    cats = " ".join(c.replace("_", " ") for c in ctx.get("preferred_categories", []))
    parts = [
        ctx.get("query_text", ""),
        cats,
        ctx.get("companion", ""),
        ctx.get("weather", ""),
        "" if ctx.get("preferred_district", "all") == "all" else ctx.get("preferred_district", ""),
    ]
    if ctx.get("end_hour", 22.0) >= 20.0:
        parts.append("kinh tế đêm buổi tối night economy evening")
    if ctx.get("weather") == "rainy":
        parts.append("trong nhà tránh mưa indoor")
    return normalize_bilingual_text(" ".join(p for p in parts if p))


def _cyclic_hour(h: float) -> List[float]:
    angle = 2.0 * math.pi * (h % 24.0) / 24.0
    return [math.sin(angle), math.cos(angle)]


def _night_overlap_ratio(start_h: float, end_h: float) -> float:
    """Fraction of requested or open span falling inside Night Economy window (18:00 - 30:00)."""
    span = max(end_h - start_h, 0.5)
    lo = max(start_h, 18.0)
    hi = min(end_h, 30.0)
    overlap = max(0.0, hi - lo)
    return min(1.0, overlap / span)


class FeatureExtractor:
    """Fits TF-IDF + SVD on Hanoi tourism text and creates structured numeric vectors."""

    def __init__(self, text_dim: int = 96):
        self.text_dim = text_dim
        self.vectorizer = TfidfVectorizer(
            max_features=3500,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )
        self.svd: TruncatedSVD | None = None
        self.actual_text_dim: int = text_dim

    def fit(self, pois: List[Dict[str, Any]], contexts: List[Dict[str, Any]]) -> "FeatureExtractor":
        corpus = [poi_to_text(p) for p in pois] + [context_to_text(c) for c in contexts]
        tfidf_mat = self.vectorizer.fit_transform(corpus)
        n_comp = min(self.text_dim, tfidf_mat.shape[1] - 1, tfidf_mat.shape[0] - 1)
        n_comp = max(16, n_comp)
        self.actual_text_dim = n_comp
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.svd.fit(tfidf_mat)
        return self

    def encode_text(self, texts: List[str]) -> np.ndarray:
        assert self.svd is not None, "FeatureExtractor must be fitted before encoding."
        mat = self.vectorizer.transform(texts)
        vec = self.svd.transform(mat).astype(np.float32)
        norms = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8
        return vec / norms

    def encode_tfidf_raw(self, texts: List[str]):
        """Raw TF-IDF sparse matrix used for the Content-Based Baseline comparison."""
        return self.vectorizer.transform(texts)

    def encode_poi_structured(self, poi: Dict[str, Any]) -> np.ndarray:
        cat_oh = [1.0 if poi.get("category") == c else 0.0 for c in CATEGORIES]
        dist_oh = [1.0 if poi.get("district") == d else 0.0 for d in DISTRICTS]

        open_h = float(poi.get("open_hour", 9.0))
        close_h = float(poi.get("close_hour", 22.0))
        is_night = 1.0 if poi.get("is_night_economy", False) else 0.0
        is_late_night = 1.0 if close_h >= 23.0 else 0.0
        is_past_midnight = 1.0 if close_h > 24.0 else 0.0
        night_ratio = _night_overlap_ratio(open_h, close_h)

        cost = max(0.0, float(poi.get("avg_cost_vnd", 100000)))
        log_cost = math.log1p(cost) / math.log1p(1_000_000.0)
        is_free = 1.0 if cost == 0 else 0.0
        duration = min(float(poi.get("duration_min", 60)), 180.0) / 180.0
        rating = (float(poi.get("rating", 4.5)) - 3.5) / 1.5
        indoor = 1.0 if poi.get("indoor", False) else 0.0
        lat_norm = (float(poi.get("lat", HANOI_LAT_CENTER)) - HANOI_LAT_CENTER) * 20.0
        lon_norm = (float(poi.get("lon", HANOI_LON_CENTER)) - HANOI_LON_CENTER) * 20.0

        feats = (
            cat_oh
            + dist_oh
            + _cyclic_hour(open_h)
            + _cyclic_hour(close_h)
            + [
                open_h / 30.0,
                close_h / 30.0,
                is_night,
                is_late_night,
                is_past_midnight,
                night_ratio,
                log_cost,
                is_free,
                duration,
                rating,
                indoor,
                lat_norm,
                lon_norm,
            ]
        )
        return np.asarray(feats, dtype=np.float32)

    def encode_context_structured(self, ctx: Dict[str, Any]) -> np.ndarray:
        pref_cats = set(ctx.get("preferred_categories", []))
        cat_mh = [1.0 if c in pref_cats else 0.0 for c in CATEGORIES]
        pref_dist = ctx.get("preferred_district", "all")
        dist_oh = [1.0 if pref_dist == d else 0.0 for d in DISTRICTS]
        comp_oh = [1.0 if ctx.get("companion", "friends") == c else 0.0 for c in COMPANIONS]
        weather_oh = [1.0 if ctx.get("weather", "clear") == w else 0.0 for w in WEATHERS]

        start_h = float(ctx.get("start_hour", 18.0))
        end_h = float(ctx.get("end_hour", 23.0))
        wants_night = 1.0 if end_h >= 20.0 else 0.0
        wants_late = 1.0 if end_h >= 23.0 else 0.0
        wants_midnight = 1.0 if end_h > 24.0 else 0.0
        night_ratio = _night_overlap_ratio(start_h, end_h)

        budget = max(0.0, float(ctx.get("budget_vnd", 400000)))
        log_budget = math.log1p(budget) / math.log1p(1_500_000.0)
        is_budget_tight = 1.0 if budget <= 180000 else 0.0
        prefers_indoor = 1.0 if ctx.get("weather") == "rainy" else 0.0

        feats = (
            cat_mh
            + dist_oh
            + comp_oh
            + weather_oh
            + _cyclic_hour(start_h)
            + _cyclic_hour(end_h)
            + [
                start_h / 30.0,
                end_h / 30.0,
                wants_night,
                wants_late,
                wants_midnight,
                night_ratio,
                log_budget,
                is_budget_tight,
                prefers_indoor,
            ]
        )
        return np.asarray(feats, dtype=np.float32)

    def transform_pois(self, pois: List[Dict[str, Any]]) -> np.ndarray:
        text_vecs = self.encode_text([poi_to_text(p) for p in pois])
        struct_vecs = np.stack([self.encode_poi_structured(p) for p in pois], axis=0)
        return np.concatenate([text_vecs, struct_vecs], axis=1).astype(np.float32)

    def transform_contexts(self, contexts: List[Dict[str, Any]]) -> np.ndarray:
        text_vecs = self.encode_text([context_to_text(c) for c in contexts])
        struct_vecs = np.stack([self.encode_context_structured(c) for c in contexts], axis=0)
        return np.concatenate([text_vecs, struct_vecs], axis=1).astype(np.float32)

    @property
    def poi_dim(self) -> int:
        # 7 cat + 6 dist + 4 cyclic + 13 scalar = 30 structured dims
        return self.actual_text_dim + 30

    @property
    def context_dim(self) -> int:
        # 7 cat + 6 dist + 4 comp + 3 weather + 4 cyclic + 9 scalar = 33 structured dims
        return self.actual_text_dim + 33

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "FeatureExtractor":
        with path.open("rb") as f:
            return pickle.load(f)
