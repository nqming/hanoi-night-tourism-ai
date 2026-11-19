"""Thread-safe JSON repository for Hanoi POIs supporting Admin CRUD operations."""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_POIS_PATH = ROOT_DIR / "data" / "processed" / "hanoi_pois.json"


class POIStorage:
    def __init__(self, path: Path = DEFAULT_POIS_PATH):
        self.path = path
        self._lock = threading.RLock()
        self._pois: List[Dict[str, Any]] = []
        self.reload()

    def reload(self) -> None:
        with self._lock:
            if not self.path.exists():
                self._pois = []
                return
            with self.path.open("r", encoding="utf-8") as f:
                self._pois = json.load(f)

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self._pois, f, ensure_ascii=False, indent=2)
        tmp.replace(self.path)

    def list_all(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._pois)

    def get(self, poi_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            for p in self._pois:
                if p["id"] == poi_id:
                    return dict(p)
            return None

    def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            poi = dict(data)
            poi["id"] = f"custom_{uuid.uuid4().hex[:8]}"
            if not poi.get("name_en"):
                poi["name_en"] = poi["name"]
            poi["source"] = "admin_created"
            self._pois.insert(0, poi)
            self._persist()
            return dict(poi)

    def update(self, poi_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        with self._lock:
            for idx, p in enumerate(self._pois):
                if p["id"] == poi_id:
                    updated = dict(p)
                    for k, v in updates.items():
                        if v is not None:
                            updated[k] = v
                    self._pois[idx] = updated
                    self._persist()
                    return dict(updated)
            return None

    def delete(self, poi_id: str) -> bool:
        with self._lock:
            before = len(self._pois)
            self._pois = [p for p in self._pois if p["id"] != poi_id]
            if len(self._pois) < before:
                self._persist()
                return True
            return False

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            by_cat: Dict[str, int] = {}
            by_dist: Dict[str, int] = {}
            by_source: Dict[str, int] = {}
            night_count = 0
            indoor_count = 0
            for p in self._pois:
                by_cat[p["category"]] = by_cat.get(p["category"], 0) + 1
                by_dist[p["district"]] = by_dist.get(p["district"], 0) + 1
                src = p.get("source", "unknown")
                by_source[src] = by_source.get(src, 0) + 1
                if p.get("is_night_economy"):
                    night_count += 1
                if p.get("indoor"):
                    indoor_count += 1
            return {
                "total_pois": len(self._pois),
                "night_economy_pois": night_count,
                "daytime_only_pois": len(self._pois) - night_count,
                "indoor_pois": indoor_count,
                "outdoor_pois": len(self._pois) - indoor_count,
                "by_category": by_cat,
                "by_district": by_dist,
                "by_source": by_source,
            }
