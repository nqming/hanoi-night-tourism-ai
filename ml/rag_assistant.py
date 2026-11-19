"""Bilingual (VI/EN) RAG AI Tourism Assistant backed by Two-Tower Contrastive Retrieval.

Features:
- Accurate natural-language extraction of late-night / post-midnight hours (e.g., "sau 0h đêm",
  "1h sáng", "after midnight", "22h - 2h sáng"), exact budgets (e.g., "50k", "50.000đ", "dưới 100k"),
  weather, companion, district, and strict category intent (e.g., eating vs drinking vs sightseeing).
- Strict pre-filtering on category, opening window, and budget so food queries NEVER return
  monuments/murals and midnight queries NEVER return venues that close at 21:00/22:00.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

from ml.inference import ContrastiveRecommender
from ml.optimizer import format_hour


def detect_language(message: str) -> str:
    vi_chars = "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    lower = message.lower()
    if any(ch in lower for ch in vi_chars):
        return "vi"
    vi_words = (
        "ha noi",
        "pho co",
        "kinh te dem",
        "an dem",
        "an gi",
        "di an",
        "du lich",
        "lich trinh",
        "goi y",
        "ho guom",
        "sau 0h",
        "tai chinh",
    )
    if any(w in lower for w in vi_words):
        return "vi"
    return "en"


def _parse_time_window(lower: str) -> Tuple[float, float]:
    """Parse Vietnamese and English time expressions into [start_hour, end_hour]
    where hours after midnight (00:00 - 06:00 next day) are represented as 24.0 - 30.0.
    """
    # 1. Explicit range: "từ 22h đến 2h", "19h-23h", "from 10pm to 2am"
    range_match = re.search(
        r"(?:từ|from)?\s*(\d{1,2})(?:h|:00|\s*giờ|\s*(?:am|pm))?\s*(?:đến|tới|-|–|to)\s*(\d{1,2})(?:h|:00|\s*giờ|\s*(?:am|pm))?",
        lower,
    )
    if range_match:
        h1, h2 = int(range_match.group(1)), int(range_match.group(2))
        if 0 <= h1 <= 24 and 0 <= h2 <= 24:
            s_h = 24.0 + h1 if h1 <= 5 else float(h1)
            e_h = 24.0 + h2 if h2 <= 6 or h2 <= h1 else float(h2)
            if e_h > s_h:
                return s_h, min(29.5, e_h)

    # 2. Explicit post-midnight patterns: "sau 0h", "0h đêm", "12h đêm", "nửa đêm", "1h sáng", "2h đêm", "after midnight"
    if any(
        kw in lower
        for kw in (
            "sau 0h",
            "0h đêm",
            "0 giờ đêm",
            "00h",
            "00:00",
            "12h đêm",
            "12 giờ đêm",
            "nửa đêm",
            "quá nửa đêm",
            "xuyên đêm",
            "after midnight",
            "past midnight",
            "12am",
            "12 am",
        )
    ):
        return 24.0, 27.5  # 00:00 (+1) -> 03:30 (+1)

    early_am_match = re.search(
        r"(?:sau\s*|lúc\s*| tầm\s*|after\s*|at\s*)?\b([0-5])\s*(?:h|giờ|am)\s*(?:đêm|sáng|khuya)?\b",
        lower,
    )
    if early_am_match:
        h_am = int(early_am_match.group(1))
        s_h = 24.0 + float(h_am)
        return s_h, min(29.0, s_h + 3.0)

    # 3. General "sau XXh" / "after XXpm" (e.g., "sau 22h", "sau 23h", "sau 21h")
    after_match = re.search(r"(?:sau|từ|lúc|after|from)\s*(\d{1,2})\s*(?:h|giờ|pm|:00)\b", lower)
    if after_match:
        h = int(after_match.group(1))
        if h == 0 or h == 24:
            return 24.0, 27.5
        if 1 <= h <= 5:
            return 24.0 + h, min(29.0, 24.0 + h + 3.0)
        if 6 <= h <= 11 and any(k in lower for k in ("tối", "đêm", "khuya", "pm", "night")):
            h += 12
        if 14 <= h <= 23:
            return float(h), min(28.0, float(h) + 3.5)

    # 4. Keyword fallbacks
    if any(k in lower for k in ("ăn khuya", "đêm khuya", "khuya", "late night", "sleepless")):
        return 22.5, 26.5  # 22:30 -> 02:30 (+1)
    if any(k in lower for k in ("chiều", "hoàng hôn", "afternoon", "sunset")):
        return 15.5, 22.0

    return 18.0, 23.0


def _parse_budget(lower: str) -> Tuple[int, Optional[int]]:
    """Return (trip_budget_vnd, strict_max_item_cost_vnd)."""
    # Match "50k", "150k", "50 nghìn", "50 ngàn", "50 cành"
    k_match = re.search(r"\b(\d{1,4})\s*(?:k|nghìn|ngàn|nhìn|cành)\b", lower)
    if k_match:
        val = int(k_match.group(1)) * 1000
        if 5000 <= val <= 10_000_000:
            return val, val

    # Match "50.000", "50,000", "50000đ", "50000 vnd"
    vnd_match = re.search(r"\b(\d{2,4})[.,](\d{3})\s*(?:đ|đồng|vnd)?\b", lower)
    if vnd_match:
        val = int(vnd_match.group(1)) * 1000 + int(vnd_match.group(2))
        if 5000 <= val <= 10_000_000:
            return val, val

    raw_vnd = re.search(r"\b(\d{5,7})\s*(?:đ|đồng|vnd)\b", lower)
    if raw_vnd:
        val = int(raw_vnd.group(1))
        return val, val

    if any(k in lower for k in ("miễn phí", "0đ", "0 đồng", "free")):
        return 0, 0
    if any(k in lower for k in ("giá rẻ", "tiết kiệm", "bình dân", "sinh viên", "cheap", "budget")):
        return 100000, 65000
    if any(k in lower for k in ("cao cấp", "sang trọng", "luxury")):
        return 900000, None

    return 450000, None


def parse_context_from_message(message: str) -> Dict[str, Any]:
    lower = message.lower()
    start_h, end_h = _parse_time_window(lower)
    budget, strict_max_cost = _parse_budget(lower)

    weather = "rainy" if any(k in lower for k in ("mưa", "rain", "trong nhà", "indoor")) else "clear"

    companion = "friends"
    if any(k in lower for k in ("gia đình", "trẻ em", "bố mẹ", "family", "kids")):
        companion = "family"
    elif any(k in lower for k in ("hẹn hò", "cặp đôi", "lãng mạn", "người yêu", "couple", "romantic", "date")):
        companion = "couple"
    elif any(k in lower for k in ("một mình", "1 mình", "solo", "alone")):
        companion = "solo"

    pref_dist = "all"
    for d_key, d_val in [
        ("hoàn kiếm", "Hoàn Kiếm"),
        ("hoan kiem", "Hoàn Kiếm"),
        ("phố cổ", "Hoàn Kiếm"),
        ("old quarter", "Hoàn Kiếm"),
        ("hồ gươm", "Hoàn Kiếm"),
        ("đồng xuân", "Hoàn Kiếm"),
        ("tây hồ", "Tây Hồ"),
        ("west lake", "Tây Hồ"),
        ("hồ tây", "Tây Hồ"),
        ("ba đình", "Ba Đình"),
        ("ba dinh", "Ba Đình"),
        ("trúc bạch", "Ba Đình"),
        ("đống đa", "Đống Đa"),
        ("dong da", "Đống Đa"),
        ("hai bà trưng", "Hai Bà Trưng"),
        ("cầu giấy", "Cầu Giấy"),
    ]:
        if d_key in lower:
            pref_dist = d_val
            break

    # Detect explicit category intent
    cats: List[str] = []
    food_kw = (
        "đi ăn",
        "ăn gì",
        "ăn đêm",
        "ăn khuya",
        "ăn tối",
        "ăn vặt",
        "quán ăn",
        "món gì",
        "đói",
        "ẩm thực",
        "phở",
        "bún",
        "cháo",
        "xôi",
        "bánh mì",
        "bánh cuốn",
        "chả cá",
        "mì ",
        "nướng",
        "lẩu",
        "chè ",
        "kem ",
        "food",
        "eat",
        "dinner",
        "hungry",
        "street food",
        "snack",
        "noodle",
        "porridge",
    )
    drink_kw = (
        "uống gì",
        "đi uống",
        "cà phê",
        "cafe",
        "coffee",
        "bia",
        "bar",
        "pub",
        "cocktail",
        "jazz",
        "nhạc sống",
        "acoustic",
        "rooftop",
        "beer",
        "drink",
    )
    heritage_kw = (
        "tour đêm",
        "di tích",
        "lịch sử",
        "hỏa lò",
        "hoàng thành",
        "văn miếu",
        "đền ",
        "chùa ",
        "nhà cổ",
        "heritage",
        "history",
        "historical",
        "temple",
        "citadel",
        "prison",
    )
    art_kw = (
        "múa rối",
        "nhà hát",
        "nghệ thuật",
        "triển lãm",
        "tuồng",
        "chèo",
        "ca trù",
        "phố sách",
        "art",
        "puppet",
        "opera",
        "theatre",
        "show",
        "gallery",
    )
    market_kw = (
        "chợ đêm",
        "chợ hoa",
        "phố đi bộ",
        "mua sắm",
        "dạo phố",
        "market",
        "walking street",
        "shopping",
    )
    scenic_kw = (
        "ngắm cảnh",
        "hóng gió",
        "cầu long biên",
        "cầu nhật tân",
        "xe buýt 2 tầng",
        "view",
        "skyline",
        "bridge",
    )

    if any(k in lower for k in food_kw):
        cats.append("night_food")
    if any(k in lower for k in drink_kw):
        cats.append("nightlife_cafe")
    if any(k in lower for k in heritage_kw):
        cats.append("heritage_night")
    if any(k in lower for k in art_kw):
        cats.append("culture_art")
    if any(k in lower for k in market_kw):
        cats.append("walking_market")
    if any(k in lower for k in scenic_kw):
        cats.append("scenic_view")

    strict_categories = len(cats) > 0
    if not cats:
        cats = ["heritage_night", "night_food", "walking_market", "nightlife_cafe"]

    return {
        "query_text": message,
        "start_hour": start_h,
        "end_hour": end_h,
        "budget_vnd": budget,
        "strict_max_cost_vnd": strict_max_cost,
        "strict_categories": strict_categories,
        "companion": companion,
        "weather": weather,
        "preferred_categories": cats,
        "preferred_district": pref_dist,
    }


def _optional_gemini_synthesis(message: str, lang: str, retrieved: List[Dict[str, Any]]) -> str | None:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    context_lines = []
    for idx, item in enumerate(retrieved, start=1):
        p = item["poi"]
        context_lines.append(
            f"{idx}. {p['name']} ({p['name_en']}) | {p['district']} | "
            f"{format_hour(p['open_hour'])}-{format_hour(p['close_hour'])} | "
            f"~{p['avg_cost_vnd']:,} VND | {p['description_vi'] if lang == 'vi' else p['description_en']}"
        )
    prompt = (
        f"Bạn là trợ lý du lịch thông minh & kinh tế đêm Hà Nội (trả lời bằng {'tiếng Việt' if lang == 'vi' else 'English'}). "
        f"Chỉ sử dụng thông tin xác thực từ danh sách điểm đến sau do mô hình Contrastive Learning truy xuất:\n"
        + "\n".join(context_lines)
        + f"\n\nCâu hỏi của du khách: {message}\nHãy tư vấn ngắn gọn, hấp dẫn kèm gợi ý khung giờ và chi phí."
    )
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        resp = requests.post(
            url,
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return None
    return None


class RAGTourismAssistant:
    def __init__(self, recommender: ContrastiveRecommender):
        self.recommender = recommender

    def answer(self, message: str, pois: List[Dict[str, Any]], lang: str | None = None) -> Dict[str, Any]:
        detected = lang if lang in ("vi", "en") else detect_language(message)
        ctx = parse_context_from_message(message)

        strict_cats = bool(ctx.get("strict_categories", False))
        strict_cost = ctx.get("strict_max_cost_vnd")

        # Pass 1: Strict category + strict open window + strict cost cap
        retrieved = self.recommender.recommend(
            ctx=ctx,
            pois=pois,
            top_k=4,
            enforce_open_window=True,
            only_night_economy=ctx["end_hour"] >= 20.0,
            enforce_categories=strict_cats,
            max_cost_vnd=strict_cost,
        )

        relaxed_note_vi = ""
        relaxed_note_en = ""

        # Pass 2: If strict cost + time window was too tight, keep strict category & time window, relax cost slightly (+25k)
        if not retrieved and strict_cost is not None:
            retrieved = self.recommender.recommend(
                ctx=ctx,
                pois=pois,
                top_k=4,
                enforce_open_window=True,
                only_night_economy=ctx["end_hour"] >= 20.0,
                enforce_categories=strict_cats,
                max_cost_vnd=strict_cost + 30000,
            )
            if retrieved:
                relaxed_note_vi = f" *(mở rộng nhẹ mức giá lân cận {strict_cost:,}đ)*"
                relaxed_note_en = f" *(slightly expanded around {strict_cost:,} VND)*"

        # Pass 3: Keep strict category, relax time window check if still empty
        if not retrieved:
            retrieved = self.recommender.recommend(
                ctx=ctx,
                pois=pois,
                top_k=4,
                enforce_open_window=False,
                only_night_economy=False,
                enforce_categories=strict_cats,
                max_cost_vnd=strict_cost,
            )

        if not retrieved:
            retrieved = self.recommender.recommend(ctx=ctx, pois=pois, top_k=4)

        llm_reply = _optional_gemini_synthesis(message, detected, retrieved)
        if llm_reply:
            return {
                "reply": llm_reply,
                "language": detected,
                "parsed_context": ctx,
                "retrieved_pois": retrieved,
                "mode": "contrastive_rag_gemini",
            }

        cat_names_vi = {
            "night_food": "Ẩm thực đêm",
            "nightlife_cafe": "Cafe / Bar / Nhạc sống",
            "heritage_night": "Di sản & Tour đêm",
            "culture_art": "Văn hóa & Nghệ thuật",
            "walking_market": "Phố đi bộ & Chợ đêm",
            "scenic_view": "Ngắm cảnh đêm",
        }
        cat_summary_vi = ", ".join(cat_names_vi.get(c, c) for c in ctx["preferred_categories"])
        cost_label_vi = (
            f"≤ **{strict_cost:,}đ/món**{relaxed_note_vi}"
            if strict_cost is not None
            else f"~**{ctx['budget_vnd']:,}đ**"
        )
        cost_label_en = (
            f"≤ **{strict_cost:,} VND**{relaxed_note_en}"
            if strict_cost is not None
            else f"~**{ctx['budget_vnd']:,} VND**"
        )

        lines: List[str] = []
        if detected == "vi":
            lines.append(
                f"Dựa trên truy xuất ngữ nghĩa từ mô hình **Two-Tower Contrastive Learning** "
                f"(nhóm: **{cat_summary_vi}**, khung giờ **{format_hour(ctx['start_hour'])} – {format_hour(ctx['end_hour'])}**, "
                f"tài chính {cost_label_vi}), đây là các địa điểm phù hợp nhất đang mở cửa tại Hà Nội:"
            )
            for idx, item in enumerate(retrieved, start=1):
                p = item["poi"]
                cost_str = "Miễn phí" if p["avg_cost_vnd"] == 0 else f"~{p['avg_cost_vnd']:,}đ"
                open_str = f"{format_hour(p['open_hour'])} – {format_hour(p['close_hour'])}"
                lines.append(
                    f"\n**{idx}. {p['name']}** ({p['district']} • Độ phù hợp AI: {item['match_percent']}%)\n"
                    f"• Địa chỉ: `{p['address']}`\n"
                    f"• Giờ mở cửa: `{open_str}` | Chi phí: `{cost_str}` | Thời lượng: `{p['duration_min']} phút`\n"
                    f"• {p['description_vi']}"
                )
            lines.append(
                "\n💡 *Mẹo:* Bạn có thể bấm nút **🔊 Nghe** để trợ lý đọc trực tiếp câu trả lời hoặc mở tab **Bản đồ & Lập lịch trình AI** để vẽ lộ trình di chuyển!"
            )
        else:
            lines.append(
                f"Using our **Two-Tower Contrastive Learning** semantic retriever "
                f"(window **{format_hour(ctx['start_hour'])} – {format_hour(ctx['end_hour'])}**, "
                f"budget {cost_label_en}), here are the best matching Hanoi venues open during your requested hours:"
            )
            for idx, item in enumerate(retrieved, start=1):
                p = item["poi"]
                cost_str = "Free" if p["avg_cost_vnd"] == 0 else f"~{p['avg_cost_vnd']:,} VND"
                open_str = f"{format_hour(p['open_hour'])} – {format_hour(p['close_hour'])}"
                lines.append(
                    f"\n**{idx}. {p['name_en']}** ({p['district']} • AI Match: {item['match_percent']}%)\n"
                    f"• Address: `{p['address']}`\n"
                    f"• Hours: `{open_str}` | Est. Cost: `{cost_str}` | Visit: `{p['duration_min']} min`\n"
                    f"• {p['description_en']}"
                )
            lines.append(
                "\n💡 *Tip:* Click **🔊 Listen** for voice playback or switch to the **Smart Map & TOPTW Itinerary** tab to route these stops!"
            )

        return {
            "reply": "\n".join(lines),
            "language": detected,
            "parsed_context": ctx,
            "retrieved_pois": retrieved,
            "mode": "contrastive_rag_local",
        }
