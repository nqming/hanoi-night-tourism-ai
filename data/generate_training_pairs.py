"""Generate realistic bilingual User-Context <-> POI positive & hard-negative triplets
for training and evaluating the Two-Tower Contrastive Learning model.

Run:
    python hanoi-night-tourism-ai/data/generate_training_pairs.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
POIS_PATH = PROCESSED_DIR / "hanoi_pois.json"
PAIRS_PATH = PROCESSED_DIR / "training_pairs.json"

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

# Bilingual realistic query templates per persona
PERSONAS: List[Dict[str, Any]] = [
    {
        "name": "night_heritage_explorer",
        "categories": ["heritage_night", "culture_art"],
        "time_range": (18.0, 22.5),
        "budget_range": (250000, 700000),
        "companions": ["solo", "couple", "family"],
        "weathers": ["clear", "cool"],
        "queries": [
            "muốn đi tour đêm di tích lịch sử văn hóa ánh sáng 3D tại Hà Nội",
            "tìm trải nghiệm lịch sử Hoàng thành Hỏa Lò Văn Miếu buổi tối",
            "khám phá di sản văn hóa và biểu diễn nghệ thuật truyền thống ban đêm",
            "night heritage tour historical landmarks and 3D light show in Hanoi",
            "cultural night experience traditional theatre and illuminated temples",
            "tham quan di tích cổ kiến trúc đẹp lung linh ánh đèn tối nay",
        ],
    },
    {
        "name": "night_foodie",
        "categories": ["night_food", "walking_market"],
        "time_range": (18.5, 25.0),
        "budget_range": (100000, 400000),
        "companions": ["friends", "solo", "couple", "family"],
        "weathers": ["clear", "cool", "rainy"],
        "queries": [
            "tìm quán ăn đêm phở bò bún chả chả cá ẩm thực phố cổ Hà Nội",
            "khám phá phố ẩm thực đêm Tống Duy Tân Ngũ Xã đồ nướng vỉa hè",
            "ăn khuya ngon bổ rẻ món đặc sản Hà Nội mở cửa sau 22 giờ",
            "best Hanoi night street food pho bun cha cha ca late night dining",
            "food tour evening local Vietnamese noodles and night market snacks",
            "muốn ăn phở cuốn gà tần xôi đêm và kem Tràng Tiền buổi tối",
        ],
    },
    {
        "name": "social_nightlife",
        "categories": ["nightlife_cafe", "walking_market"],
        "time_range": (19.5, 26.0),
        "budget_range": (200000, 800000),
        "companions": ["friends", "couple", "solo"],
        "weathers": ["clear", "cool", "rainy"],
        "queries": [
            "đi chơi đêm phố bia Tạ Hiện nghe nhạc sống cocktail bar sôi động",
            "tìm quán pub bar speakeasy hoặc cà phê nhạc acoustic mở khuya",
            "không gian giải trí về đêm giao lưu bạn bè quốc tế tại Hoàn Kiếm",
            "vibrant Hanoi nightlife beer street craft cocktails live jazz bar",
            "late night pub cafe rooftop lounge and live acoustic music",
            "uống bia hơi cà phê trứng nghe nhạc jazz thư giãn tối muộn",
        ],
    },
    {
        "name": "romantic_evening",
        "categories": ["scenic_view", "nightlife_cafe", "culture_art"],
        "time_range": (17.5, 23.5),
        "budget_range": (300000, 900000),
        "companions": ["couple"],
        "weathers": ["clear", "cool"],
        "queries": [
            "lịch trình hẹn hò lãng mạn ngắm toàn cảnh Hồ Tây rooftop bar buổi tối",
            "nghe nhạc jazz hòa nhạc Nhà hát Lớn và ngắm cảnh đêm từ trên cao",
            "dạo hồ Gươm uống cocktail yên tĩnh không gian nghệ thuật cho cặp đôi",
            "romantic Hanoi night rooftop skyline view West Lake cocktails jazz",
            "evening date night concert opera house quiet speakeasy cafe",
            "ngắm hoàng hôn đường Thanh Niên và dạo cầu Long Biên mát mẻ",
        ],
    },
    {
        "name": "family_evening",
        "categories": ["culture_art", "walking_market", "night_food", "scenic_view"],
        "time_range": (17.0, 22.0),
        "budget_range": (200000, 600000),
        "companions": ["family"],
        "weathers": ["clear", "cool"],
        "queries": [
            "đi chơi tối cùng gia đình xem múa rối nước đi phố đi bộ Hồ Gươm",
            "trải nghiệm xe buýt 2 tầng ngắm phố đêm và ăn kem phố cổ cho gia đình",
            "hoạt động nhẹ nhàng an toàn buổi tối công viên phố đi bộ ăn phở cuốn",
            "family friendly Hanoi evening water puppet show walking street night bus",
            "safe evening activities for family kids lakeside stroll and local dinner",
            "xem lễ hạ cờ Ba Đình và dạo chợ đêm mua quà lưu niệm",
        ],
    },
    {
        "name": "budget_backpacker",
        "categories": ["walking_market", "night_food", "scenic_view", "heritage_night"],
        "time_range": (18.0, 24.5),
        "budget_range": (50000, 180000),
        "companions": ["solo", "friends"],
        "weathers": ["clear", "cool"],
        "queries": [
            "du lịch đêm tiết kiệm miễn phí phố bích họa cầu Long Biên trà chanh",
            "đi chơi tối Hà Nội dưới 150k ăn vặt chợ đêm phố đi bộ",
            "trải nghiệm Hà Nội về đêm giá rẻ bình dân cho sinh viên du khách ba lô",
            "budget friendly cheap Hanoi night walk free attractions street food",
            "free night sights Ba Dinh flag ceremony mural street lemon tea",
            "phở gánh đêm xôi đêm cà phê vỉa hè giá bình dân phố cổ",
        ],
    },
    {
        "name": "midnight_sleepless",
        "categories": ["night_food", "nightlife_cafe", "walking_market"],
        "time_range": (22.0, 28.0),
        "budget_range": (100000, 450000),
        "companions": ["friends", "solo", "couple"],
        "weathers": ["clear", "cool"],
        "queries": [
            "khám phá Hà Nội sau nửa đêm chợ hoa Quảng Bá phở gánh hàng Chiếu",
            "địa điểm mở cửa xuyên đêm từ 23h đến sáng Tống Duy Tân Tạ Hiện",
            "trải nghiệm thành phố không ngủ ăn khuya và chụp ảnh chợ hoa đêm",
            "midnight Hanoi tour Quang Ba flower market 2 AM pho and late night street",
            "open past midnight sleepless Hanoi late night food and bars",
            "đi đâu chơi lúc 1 giờ sáng ở Hà Nội ăn đêm cà phê khuya",
        ],
    },
    {
        "name": "rainy_indoor_evening",
        "categories": ["culture_art", "nightlife_cafe", "night_food", "heritage_night"],
        "time_range": (18.0, 23.0),
        "budget_range": (150000, 600000),
        "companions": ["solo", "couple", "friends", "family"],
        "weathers": ["rainy"],
        "queries": [
            "trời mưa buổi tối đi đâu trong nhà ở Hà Nội xem múa rối uống cà phê sách",
            "trải nghiệm tối trong nhà ấm cúng Nhà hát Lớn bảo tàng cafe trứng",
            "ăn tối ấm nóng chả cá phở bò và nghe nhạc jazz trong nhà khi trời mưa",
            "rainy night indoor Hanoi activities water puppet theatre egg coffee jazz",
            "cozy indoor evening dining museum night tour and speakeasy bar",
            "tránh mưa buổi tối tại quán cafe acoustic phòng triển lãm nghệ thuật",
        ],
    },
    {
        "name": "day_to_night_culture",
        "categories": ["daytime_heritage", "heritage_night", "culture_art", "night_food"],
        "time_range": (14.0, 22.0),
        "budget_range": (200000, 650000),
        "companions": ["solo", "couple", "family"],
        "weathers": ["clear", "cool", "rainy"],
        "queries": [
            "lịch trình kết hợp tham quan bảo tàng chiều và tour đêm ăn tối Hà Nội",
            "đi từ chiều thăm chùa Trấn Quốc bảo tàng mỹ thuật rồi ăn phở cuốn đêm",
            "kết nối du lịch ban ngày và kinh tế đêm trọn vẹn tại trung tâm Hà Nội",
            "afternoon to night Hanoi itinerary daytime museum sunset pagoda night dinner",
            "seamless day to evening cultural tour art gallery and night walking street",
            "tham quan di tích buổi chiều nối tiếp không gian phố đi bộ buổi tối",
        ],
    },
    {
        "name": "west_lake_chill",
        "categories": ["scenic_view", "walking_market", "night_food", "nightlife_cafe"],
        "time_range": (17.5, 23.5),
        "budget_range": (150000, 550000),
        "companions": ["couple", "friends", "family"],
        "weathers": ["clear", "cool"],
        "queries": [
            "hóng gió Hồ Tây tối đi phố đi bộ Trịnh Công Sơn ăn bánh tôm phở cuốn",
            "trải nghiệm du lịch đêm khu vực Tây Hồ Trúc Bạch ngắm hoàng hôn",
            "chụp ảnh chợ hoa Quảng Bá bảo tàng văn học và rooftop nhìn ra Hồ Tây",
            "West Lake Tay Ho evening breeze Tranh Nien sunset Ngũ Xã food",
            "chill lakeside Hanoi night Trinh Cong Son walking street west lake bar",
            "không gian thoáng mát ven hồ ít đông đúc khu Tây Hồ Ba Đình",
        ],
    },
]


def time_overlap_hours(start_a: float, end_a: float, open_b: float, close_b: float) -> float:
    """Compute overlap duration in hours, handling overnight intervals (up to 30.0 = 06:00 next day)."""
    best = 0.0
    for shift in (0.0, 24.0, -24.0):
        lo = max(start_a, open_b + shift)
        hi = min(end_a, close_b + shift)
        if hi > lo:
            best = max(best, hi - lo)
    return best


def compute_compatibility_score(ctx: Dict[str, Any], poi: Dict[str, Any]) -> Tuple[float, List[str]]:
    """Compute ground-truth compatibility score & violation tags for realistic pairing."""
    violations: List[str] = []
    score = 0.0

    # 1. Time window overlap check
    min_needed_h = max(0.5, poi["duration_min"] / 60.0)
    overlap = time_overlap_hours(ctx["start_hour"], ctx["end_hour"], poi["open_hour"], poi["close_hour"])
    if overlap >= min_needed_h:
        score += 3.0 + min(overlap / 4.0, 0.8)
    elif overlap >= 0.35:
        score += 0.8
        violations.append("tight_time")
    else:
        score -= 4.5
        violations.append("closed_in_window")

    # 2. Night-economy alignment
    wants_night = ctx["end_hour"] >= 20.0
    if wants_night and poi["is_night_economy"]:
        score += 1.5
    elif wants_night and not poi["is_night_economy"] and ctx["start_hour"] >= 18.0:
        score -= 3.0
        violations.append("not_night_economy")

    # 3. Category preference match
    if poi["category"] in ctx["preferred_categories"]:
        score += 2.8
    else:
        score -= 1.2
        violations.append("category_mismatch")

    # 4. Budget feasibility (per-stop reasonable budget <= 70% of total trip budget)
    budget_cap = max(ctx["budget_vnd"] * 0.75, 50000)
    if poi["avg_cost_vnd"] <= budget_cap:
        score += 1.5
    else:
        score -= 3.2
        violations.append("over_budget")

    # 5. Weather suitability
    if ctx["weather"] == "rainy":
        if poi["indoor"]:
            score += 2.0
        else:
            score -= 2.8
            violations.append("outdoor_in_rain")

    # 6. District preference bonus
    if ctx["preferred_district"] != "all":
        if poi["district"] == ctx["preferred_district"]:
            score += 1.2
        else:
            score -= 0.3

    # 7. Quality & authentic popularity prior
    score += (poi["rating"] - 4.0) * 0.8
    if poi["source"].startswith("curated"):
        score += 0.4

    return score, violations


def generate_dataset(num_queries: int = 1600, seed: int = 42) -> Dict[str, Any]:
    rng = random.Random(seed)
    with POIS_PATH.open("r", encoding="utf-8") as f:
        pois: List[Dict[str, Any]] = json.load(f)

    # Hold out 15% of POIs as Cold-Start evaluation POIs (never seen during training!)
    poi_ids = [p["id"] for p in pois]
    shuffled_ids = list(poi_ids)
    rng.shuffle(shuffled_ids)
    n_cold = max(6, int(len(pois) * 0.15))
    cold_start_poi_ids = set(shuffled_ids[:n_cold])
    train_pool_pois = [p for p in pois if p["id"] not in cold_start_poi_ids]

    samples: List[Dict[str, Any]] = []

    for q_idx in range(num_queries):
        persona = PERSONAS[q_idx % len(PERSONAS)]
        t_lo, t_hi = persona["time_range"]
        start_h = round(rng.uniform(t_lo, max(t_lo + 0.5, t_hi - 2.5)), 1)
        end_h = round(min(t_hi + 0.5, rng.uniform(start_h + 2.0, t_hi + 0.5)), 1)
        budget = int(round(rng.uniform(*persona["budget_range"]) / 10000) * 10000)
        companion = rng.choice(persona["companions"])
        weather = rng.choice(persona["weathers"])

        # Occasional district specificity
        if persona["name"] == "west_lake_chill":
            pref_dist = rng.choice(["Tây Hồ", "Ba Đình", "all"])
        elif rng.random() < 0.35:
            pref_dist = rng.choice(["Hoàn Kiếm", "Ba Đình", "Tây Hồ", "Đống Đa", "Hai Bà Trưng"])
        else:
            pref_dist = "all"

        base_query = rng.choice(persona["queries"])
        # Add contextual natural-language modifiers for rich semantic training
        modifiers = []
        if pref_dist != "all":
            modifiers.append(f"khu vực {pref_dist}")
        if weather == "rainy":
            modifiers.append("trời mưa ưu tiên trong nhà")
        if budget <= 180000:
            modifiers.append("ngân sách tiết kiệm")
        elif budget >= 600000:
            modifiers.append("trải nghiệm cao cấp")
        if start_h >= 21.5:
            modifiers.append("mở cửa khuya sau 22h")

        query_text = base_query + (" " + " ".join(modifiers) if modifiers else "")

        ctx = {
            "query_id": f"q_{q_idx:05d}",
            "persona": persona["name"],
            "query_text": query_text,
            "start_hour": start_h,
            "end_hour": end_h,
            "budget_vnd": budget,
            "companion": companion,
            "weather": weather,
            "preferred_categories": persona["categories"],
            "preferred_district": pref_dist,
        }

        # Determine split: 75% train, 10% val, 15% test
        if q_idx < int(num_queries * 0.75):
            split = "train"
            candidate_pois = train_pool_pois
        elif q_idx < int(num_queries * 0.85):
            split = "val"
            candidate_pois = train_pool_pois
        else:
            split = "test"
            candidate_pois = pois  # Includes Cold-Start POIs!

        scored = []
        for p in candidate_pois:
            s, viols = compute_compatibility_score(ctx, p)
            scored.append((s, viols, p))
        scored.sort(key=lambda x: x[0], reverse=True)

        positives = [item for item in scored if item[0] >= 5.2 and "closed_in_window" not in item[1] and "over_budget" not in item[1]]
        if not positives:
            positives = scored[:3]

        # Top positive(s) for retrieval evaluation
        top_pos = positives[: min(5, len(positives))]
        chosen_pos = rng.choice(top_pos[: min(3, len(top_pos))])[2]

        # Hard negatives: POIs that violate time, budget, weather, or category
        hard_negs = [
            item[2]
            for item in scored
            if item[2]["id"] != chosen_pos["id"]
            and (
                "closed_in_window" in item[1]
                or "over_budget" in item[1]
                or "outdoor_in_rain" in item[1]
                or "not_night_economy" in item[1]
                or "category_mismatch" in item[1]
            )
            and item[0] < 2.5
        ]
        if not hard_negs:
            hard_negs = [item[2] for item in scored[-10:]]
        chosen_hard_neg = rng.choice(hard_negs[: min(25, len(hard_negs))])

        samples.append(
            {
                "split": split,
                "context": ctx,
                "positive_poi_id": chosen_pos["id"],
                "relevant_poi_ids": [item[2]["id"] for item in top_pos],
                "hard_negative_poi_id": chosen_hard_neg["id"],
            }
        )

    dataset = {
        "meta": {
            "total_samples": len(samples),
            "train_samples": sum(1 for s in samples if s["split"] == "train"),
            "val_samples": sum(1 for s in samples if s["split"] == "val"),
            "test_samples": sum(1 for s in samples if s["split"] == "test"),
            "total_pois": len(pois),
            "cold_start_poi_ids": sorted(cold_start_poi_ids),
        },
        "samples": samples,
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with PAIRS_PATH.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print("[DataGen] Generated Contrastive Learning dataset:")
    print(json.dumps(dataset["meta"], ensure_ascii=False, indent=2))
    return dataset


if __name__ == "__main__":
    generate_dataset()
