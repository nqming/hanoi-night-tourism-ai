"""Bilingual (VI/EN) RAG AI Tourism Assistant backed by Two-Tower Contrastive Retrieval.

Works 100% offline/zero-cost out of the box using grounded semantic retrieval over the
crawled Hanoi POI database. Optionally enhances fluency via Gemini API if GEMINI_API_KEY
is set in environment variables.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List

import requests

from ml.inference import ContrastiveRecommender
from ml.optimizer import format_hour


def detect_language(message: str) -> str:
    vi_chars = "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    lower = message.lower()
    if any(ch in lower for ch in vi_chars):
        return "vi"
    vi_words = ("ha noi", "pho co", "kinh te dem", "an dem", "du lich", "lich trinh", "goi y", "ho guom")
    if any(w in lower for w in vi_words):
        return "vi"
    return "en"


def parse_context_from_message(message: str) -> Dict[str, Any]:
    lower = message.lower()

    # Infer time window
    start_h, end_h = 18.0, 23.0
    if any(k in lower for k in ("khuya", "nửa đêm", "midnight", "1h", "2h", "sau 22", "late night", "23h")):
        start_h, end_h = 22.0, 26.5
    elif any(k in lower for k in ("chiều", "hoàng hôn", "afternoon", "sunset", "cả ngày")):
        start_h, end_h = 15.0, 22.5

    # Infer budget (VND)
    budget = 450000
    k_match = re.search(r"(\d{2,4})\s*k\b", lower)
    if k_match:
        budget = int(k_match.group(1)) * 1000
    elif any(k in lower for k in ("miễn phí", "giá rẻ", "tiết kiệm", "free", "cheap", "budget")):
        budget = 150000
    elif any(k in lower for k in ("cao cấp", "sang trọng", "luxury", "rooftop")):
        budget = 900000

    # Infer weather
    weather = "rainy" if any(k in lower for k in ("mưa", "rain", "trong nhà", "indoor")) else "clear"

    # Infer companion
    companion = "friends"
    if any(k in lower for k in ("gia đình", "trẻ em", "family", "kids")):
        companion = "family"
    elif any(k in lower for k in ("hẹn hò", "cặp đôi", "lãng mạn", "couple", "romantic", "date")):
        companion = "couple"
    elif any(k in lower for k in ("một mình", "solo", "alone")):
        companion = "solo"

    # Infer district
    pref_dist = "all"
    for d_key, d_val in [
        ("hoàn kiếm", "Hoàn Kiếm"),
        ("hoan kiem", "Hoàn Kiếm"),
        ("phố cổ", "Hoàn Kiếm"),
        ("old quarter", "Hoàn Kiếm"),
        ("tây hồ", "Tây Hồ"),
        ("west lake", "Tây Hồ"),
        ("ba đình", "Ba Đình"),
        ("ba dinh", "Ba Đình"),
        ("đống đa", "Đống Đa"),
        ("hai bà trưng", "Hai Bà Trưng"),
    ]:
        if d_key in lower:
            pref_dist = d_val
            break

    # Infer categories
    cats: List[str] = []
    if any(k in lower for k in ("ăn", "phở", "bún", "ẩm thực", "chả cá", "food", "eat", "dinner", "street food")):
        cats.append("night_food")
    if any(k in lower for k in ("tour đêm", "di tích", "lịch sử", "hỏa lò", "hoàng thành", "văn miếu", "heritage", "history")):
        cats.append("heritage_night")
    if any(k in lower for k in ("múa rối", "nhà hát", "nghệ thuật", "triển lãm", "art", "puppet", "opera", "show")):
        cats.append("culture_art")
    if any(k in lower for k in ("bia", "bar", "pub", "cocktail", "cafe", "cà phê", "jazz", "nhạc sống", "beer", "coffee")):
        cats.append("nightlife_cafe")
    if any(k in lower for k in ("chợ đêm", "phố đi bộ", "mua sắm", "market", "walking street")):
        cats.append("walking_market")
    if any(k in lower for k in ("ngắm cảnh", "rooftop", "hồ", "cầu", "view", "skyline", "bus")):
        cats.append("scenic_view")

    if not cats:
        cats = ["heritage_night", "night_food", "walking_market", "nightlife_cafe"]

    return {
        "query_text": message,
        "start_hour": start_h,
        "end_hour": end_h,
        "budget_vnd": budget,
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

        retrieved = self.recommender.recommend(
            ctx=ctx,
            pois=pois,
            top_k=4,
            enforce_open_window=True,
            only_night_economy=ctx["end_hour"] >= 20.0,
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

        # Deterministic, rich grounded RAG response (zero external API cost)
        lines: List[str] = []
        if detected == "vi":
            lines.append(
                f"Dựa trên truy xuất ngữ nghĩa từ mô hình **Two-Tower Contrastive Learning** "
                f"(khung giờ **{format_hour(ctx['start_hour'])} – {format_hour(ctx['end_hour'])}**, "
                f"ngân sách dự kiến ~**{ctx['budget_vnd']:,}đ**), tôi gợi ý cho bạn các điểm đến nổi bật sau tại Hà Nội:"
            )
            for idx, item in enumerate(retrieved, start=1):
                p = item["poi"]
                cost_str = "Miễn phí" if p["avg_cost_vnd"] == 0 else f"~{p['avg_cost_vnd']:,}đ"
                open_str = f"{format_hour(p['open_hour'])} – {format_hour(p['close_hour'])}"
                lines.append(
                    f"\n**{idx}. {p['name']}** ({p['district']} • Độ phù hợp AI: {item['match_percent']}%)\n"
                    f"• Giờ mở cửa: `{open_str}` | Chi phí: `{cost_str}` | Thời lượng: `{p['duration_min']} phút`\n"
                    f"• {p['description_vi']}"
                )
            lines.append(
                "\n💡 *Mẹo:* Bạn có thể chuyển sang tab **Bản đồ & Lập lịch trình Tối ưu** để hệ thống tự động sắp xếp thứ tự di chuyển ngắn nhất cho các địa điểm này!"
            )
        else:
            lines.append(
                f"Using our **Two-Tower Contrastive Learning** semantic retriever "
                f"(window **{format_hour(ctx['start_hour'])} – {format_hour(ctx['end_hour'])}**, "
                f"budget ~**{ctx['budget_vnd']:,} VND**), here are the best grounded Hanoi Night Economy picks for you:"
            )
            for idx, item in enumerate(retrieved, start=1):
                p = item["poi"]
                cost_str = "Free" if p["avg_cost_vnd"] == 0 else f"~{p['avg_cost_vnd']:,} VND"
                open_str = f"{format_hour(p['open_hour'])} – {format_hour(p['close_hour'])}"
                lines.append(
                    f"\n**{idx}. {p['name_en']}** ({p['district']} • AI Match: {item['match_percent']}%)\n"
                    f"• Hours: `{open_str}` | Est. Cost: `{cost_str}` | Visit: `{p['duration_min']} min`\n"
                    f"• {p['description_en']}"
                )
            lines.append(
                "\n💡 *Tip:* Switch to the **Smart Map & TOPTW Itinerary** tab to auto-route these destinations with minimum travel time and zero opening-hour conflicts!"
            )

        return {
            "reply": "\n".join(lines),
            "language": detected,
            "parsed_context": ctx,
            "retrieved_pois": retrieved,
            "mode": "contrastive_rag_local",
        }
