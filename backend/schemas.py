"""Pydantic v2 request and response schemas for the Hanoi Night Tourism AI API."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

CategoryType = Literal[
    "heritage_night",
    "culture_art",
    "walking_market",
    "night_food",
    "nightlife_cafe",
    "scenic_view",
    "daytime_heritage",
]

CompanionType = Literal["solo", "couple", "friends", "family"]
WeatherType = Literal["clear", "cool", "rainy"]
TransportType = Literal["walk", "motorbike", "car"]


class POICreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=200)
    name_en: Optional[str] = Field(default=None, max_length=200)
    category: CategoryType = "heritage_night"
    district: str = Field(default="Hoàn Kiếm", min_length=2, max_length=80)
    address: str = Field(default="Trung tâm Hà Nội", max_length=260)
    lat: float = Field(..., ge=20.90, le=21.15)
    lon: float = Field(..., ge=105.70, le=105.98)
    open_hour: float = Field(default=18.0, ge=0.0, le=24.0)
    close_hour: float = Field(default=23.0, ge=0.5, le=30.0)
    is_night_economy: bool = True
    avg_cost_vnd: int = Field(default=100000, ge=0, le=10_000_000)
    duration_min: int = Field(default=60, ge=15, le=240)
    rating: float = Field(default=4.6, ge=1.0, le=5.0)
    review_count: int = Field(default=100, ge=0)
    indoor: bool = False
    tags: List[str] = Field(default_factory=list)
    description_vi: str = Field(default="Điểm trải nghiệm du lịch đêm tại Hà Nội.", max_length=1500)
    description_en: str = Field(default="Hanoi night tourism destination.", max_length=1500)


class POIUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=2, max_length=200)
    name_en: Optional[str] = Field(default=None, max_length=200)
    category: Optional[CategoryType] = None
    district: Optional[str] = None
    address: Optional[str] = None
    lat: Optional[float] = Field(default=None, ge=20.90, le=21.15)
    lon: Optional[float] = Field(default=None, ge=105.70, le=105.98)
    open_hour: Optional[float] = Field(default=None, ge=0.0, le=24.0)
    close_hour: Optional[float] = Field(default=None, ge=0.5, le=30.0)
    is_night_economy: Optional[bool] = None
    avg_cost_vnd: Optional[int] = Field(default=None, ge=0, le=10_000_000)
    duration_min: Optional[int] = Field(default=None, ge=15, le=240)
    rating: Optional[float] = Field(default=None, ge=1.0, le=5.0)
    indoor: Optional[bool] = None
    tags: Optional[List[str]] = None
    description_vi: Optional[str] = None
    description_en: Optional[str] = None


class RecommendRequest(BaseModel):
    query_text: str = Field(
        default="khám phá tour đêm di tích văn hóa và ẩm thực phố cổ Hà Nội",
        max_length=500,
    )
    start_hour: float = Field(default=18.0, ge=6.0, le=28.0)
    end_hour: float = Field(default=23.5, ge=7.0, le=30.0)
    budget_vnd: int = Field(default=500000, ge=0, le=20_000_000)
    companion: CompanionType = "friends"
    weather: WeatherType = "clear"
    preferred_categories: List[CategoryType] = Field(
        default_factory=lambda: ["heritage_night", "night_food", "walking_market"]
    )
    preferred_district: str = "all"
    top_k: int = Field(default=10, ge=1, le=50)
    only_night_economy: bool = False
    enforce_open_window: bool = False


class OptimizeItineraryRequest(BaseModel):
    query_text: str = Field(
        default="trải nghiệm di sản đêm, ẩm thực đường phố và không gian văn hóa Hà Nội",
        max_length=500,
    )
    start_hour: float = Field(default=17.5, ge=6.0, le=28.0)
    end_hour: float = Field(default=23.0, ge=7.0, le=30.0)
    budget_vnd: int = Field(default=500000, ge=50000, le=20_000_000)
    max_stops: int = Field(default=4, ge=1, le=8)
    companion: CompanionType = "friends"
    weather: WeatherType = "clear"
    preferred_categories: List[CategoryType] = Field(
        default_factory=lambda: ["heritage_night", "night_food", "walking_market", "nightlife_cafe"]
    )
    preferred_district: str = "all"
    transport_mode: TransportType = "motorbike"
    start_lat: float = Field(default=21.0313, ge=20.90, le=21.15)
    start_lon: float = Field(default=105.8520, ge=105.70, le=105.98)
    only_night_economy: bool = False


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=2, max_length=800)
    lang: Optional[Literal["vi", "en"]] = None
