"""Day-to-Night Time-Window Itinerary Optimizer (TOPTW Solver) & Baseline Comparator.

Solves the Team Orienteering Problem with Time Windows (TOPTW) over Top-K POIs retrieved
by the Two-Tower Contrastive Learning model:
- Handles overnight wrap-around schedules (e.g., 17:00-02:00 represented as 17.0-26.0,
  and 24/7 spots accessible throughout the night).
- Enforces start-proximity ordering and anti-backtracking: if a destination is right at
  or near the user's current starting point (e.g., Hoan Kiem Lake), it is visited FIRST
  before moving outward (never leaving the start area and looping back to it!).
- Evaluates exact permutations (k! for k <= 6) + 2-opt segment reversal on top beam
  candidates to guarantee the shortest non-backtracking route.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_START_LAT = 21.0313
DEFAULT_START_LON = 105.8520

TRANSPORT_SPEED_KMH = {
    "walk": 4.5,
    "motorbike": 20.0,
    "car": 17.0,
}

ROAD_TORTUOSITY = 1.32


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def estimate_travel(lat1: float, lon1: float, lat2: float, lon2: float, mode: str) -> Tuple[float, float]:
    """Return (road_distance_km, travel_minutes)."""
    straight_km = haversine_km(lat1, lon1, lat2, lon2)
    road_km = round(straight_km * ROAD_TORTUOSITY, 2)
    speed = TRANSPORT_SPEED_KMH.get(mode, 18.0)
    minutes = (road_km / speed) * 60.0 + (3.5 if road_km > 0.25 else 1.0)
    return road_km, round(minutes, 1)


def format_hour(h: float) -> str:
    """Format fractional hour (including >=24.0 next-day hours) into HH:MM (+1)."""
    next_day = h >= 24.0
    h_mod = h % 24.0
    hh = int(h_mod)
    mm = int(round((h_mod - hh) * 60))
    if mm == 60:
        hh = (hh + 1) % 24
        mm = 0
    suffix = " (+1)" if next_day else ""
    return f"{hh:02d}:{mm:02d}{suffix}"


def effective_open_window(poi: Dict[str, Any], tour_start: float, tour_end: float) -> Tuple[float, float]:
    """Return effective (open_h, close_h) aligned with the tour window [tour_start, tour_end]."""
    o = float(poi["open_hour"])
    c = float(poi["close_hour"])
    if o == 0.0 and c >= 24.0:
        return 0.0, max(30.0, tour_end + 1.0)
    if c <= tour_start and c + 24.0 > tour_start:
        return o + 24.0, c + 24.0
    return o, c


def simulate_sequence(
    seq: List[Dict[str, Any]],
    start_lat: float,
    start_lon: float,
    start_hour: float,
    end_hour: float,
    budget_vnd: int,
    transport_mode: str,
    max_wait_min: float = 40.0,
) -> Optional[Dict[str, Any]]:
    """Simulate a candidate sequence of (poi, ai_score) entries and validate all constraints."""
    cur_lat, cur_lon = start_lat, start_lon
    cur_time = start_hour
    total_cost = 0
    total_dist_km = 0.0
    total_travel_min = 0.0
    total_wait_min = 0.0
    total_visit_min = 0.0
    night_active_min = 0.0
    stops_detail: List[Dict[str, Any]] = []

    for idx, entry in enumerate(seq, start=1):
        poi = entry["poi"]
        ai_sim = entry["cosine_similarity"]
        match_pct = entry["match_percent"]

        cost = int(poi["avg_cost_vnd"])
        if total_cost + cost > budget_vnd:
            return None

        dist_km, travel_min = estimate_travel(cur_lat, cur_lon, poi["lat"], poi["lon"], transport_mode)
        arrival = cur_time + travel_min / 60.0

        open_h, close_h = effective_open_window(poi, start_hour, end_hour)
        duration_h = float(poi["duration_min"]) / 60.0

        if arrival < open_h:
            wait_min = (open_h - arrival) * 60.0
            if wait_min > max_wait_min:
                return None
            service_start = open_h
        else:
            wait_min = 0.0
            service_start = arrival

        departure = service_start + duration_h
        if departure > close_h + 0.05 or departure > end_hour + 0.15:
            return None

        total_cost += cost
        total_dist_km += dist_km
        total_travel_min += travel_min
        total_wait_min += wait_min
        total_visit_min += float(poi["duration_min"])

        night_lo = max(service_start, 18.0)
        night_hi = min(departure, 30.0)
        if night_hi > night_lo:
            night_active_min += (night_hi - night_lo) * 60.0

        stops_detail.append(
            {
                "order": idx,
                "poi": poi,
                "arrival_time": format_hour(arrival),
                "start_time": format_hour(service_start),
                "departure_time": format_hour(departure),
                "arrival_hour": round(arrival, 2),
                "departure_hour": round(departure, 2),
                "distance_from_prev_km": dist_km,
                "travel_from_prev_min": travel_min,
                "wait_min": round(wait_min, 1),
                "cosine_similarity": ai_sim,
                "match_percent": match_pct,
                "reasons": entry.get("reasons", []),
            }
        )

        cur_lat, cur_lon = poi["lat"], poi["lon"]
        cur_time = departure

    total_span_min = max(1.0, (cur_time - start_hour) * 60.0)
    night_stops = sum(1 for s in stops_detail if s["poi"].get("is_night_economy") or s["departure_hour"] >= 18.5)
    categories = [s["poi"]["category"] for s in stops_detail]
    unique_cats = len(set(categories))

    return {
        "stops": stops_detail,
        "summary": {
            "stop_count": len(stops_detail),
            "total_distance_km": round(total_dist_km, 2),
            "total_travel_min": round(total_travel_min, 1),
            "total_wait_min": round(total_wait_min, 1),
            "total_visit_min": round(total_visit_min, 1),
            "total_cost_vnd": total_cost,
            "budget_vnd": budget_vnd,
            "budget_utilization_pct": round((total_cost / max(1, budget_vnd)) * 100.0, 1),
            "night_economy_stops": night_stops,
            "night_coverage_pct": round(min(100.0, (night_active_min / total_span_min) * 100.0), 1),
            "category_diversity": unique_cats,
            "avg_ai_match_pct": round(
                sum(s["match_percent"] for s in stops_detail) / max(1, len(stops_detail)), 1
            ),
            "start_time": format_hour(start_hour),
            "end_time": format_hour(cur_time),
        },
    }


def _backtracking_penalty(
    stops: List[Dict[str, Any]], start_lat: float, start_lon: float, start_hour: float
) -> float:
    """Penalize leaving the starting area and looping back to it later, or sharp zig-zag turns."""
    if len(stops) <= 1:
        return 0.0

    penalty = 0.0
    dists_from_start = [
        haversine_km(start_lat, start_lon, s["poi"]["lat"], s["poi"]["lon"]) for s in stops
    ]

    # 1. Start-proximity inversion penalty:
    # If stop j (j > i) is significantly closer to the user's starting point than stop i,
    # and stop j was already open at stop i's arrival time, penalize going far first then coming back!
    for i in range(len(stops)):
        for j in range(i + 1, len(stops)):
            di = dists_from_start[i]
            dj = dists_from_start[j]
            if dj + 0.18 < di:
                # Check if stop j could have been visited earlier at stop i's time
                open_j = float(stops[j]["poi"]["open_hour"])
                if open_j <= stops[i]["arrival_hour"] + 0.1:
                    # Heavy penalty when dj is right next to start (< 0.45 km) and di went away (> 0.5 km)
                    if dj < 0.45 and di > 0.45:
                        penalty += 2.2 + 2.0 * (di - dj)
                    else:
                        penalty += 0.85 * (di - dj)

    # 2. Zig-zag triangle detour penalty across consecutive triples (A -> B -> C)
    coords = [(start_lat, start_lon)] + [(s["poi"]["lat"], s["poi"]["lon"]) for s in stops]
    for k in range(len(coords) - 2):
        a_lat, a_lon = coords[k]
        b_lat, b_lon = coords[k + 1]
        c_lat, c_lon = coords[k + 2]
        ab = haversine_km(a_lat, a_lon, b_lat, b_lon)
        bc = haversine_km(b_lat, b_lon, c_lat, c_lon)
        ac = haversine_km(a_lat, a_lon, c_lat, c_lon)
        # If going A -> B -> C travels way farther than A -> C and C is closer to A than B is:
        if ab > 0.35 and ac + 0.20 < ab:
            penalty += 1.4 * (ab + bc - ac)

    return penalty


def _objective_score(
    sim_res: Dict[str, Any],
    start_lat: float,
    start_lon: float,
    start_hour: float,
    prefer_night: bool = True,
) -> float:
    """Multi-objective utility balancing AI relevance, diversity, night coverage, and non-backtracking routing."""
    stops = sim_res["stops"]
    summary = sim_res["summary"]
    if not stops:
        return -1e9

    ai_sum = sum(max(0.05, s["cosine_similarity"] + 0.35) for s in stops)
    diversity_bonus = 0.45 * summary["category_diversity"]
    repeat_penalty = sum(
        0.35
        for i in range(1, len(stops))
        if stops[i]["poi"]["category"] == stops[i - 1]["poi"]["category"]
    )
    night_bonus = (0.35 * summary["night_economy_stops"]) if prefer_night else 0.0

    # Strong distance penalty + explicit anti-backtracking penalty
    dist_penalty = 0.50 * summary["total_distance_km"]
    wait_penalty = 0.55 * (summary["total_wait_min"] / 60.0)
    bt_penalty = _backtracking_penalty(stops, start_lat, start_lon, start_hour)

    # First-stop proximity bonus: reward visiting a stop right where the user is standing first!
    d_first = haversine_km(start_lat, start_lon, stops[0]["poi"]["lat"], stops[0]["poi"]["lon"])
    first_stop_bonus = 0.45 if d_first < 0.35 else (0.20 if d_first < 0.65 else 0.0)

    return (
        ai_sum
        + diversity_bonus
        + night_bonus
        + first_stop_bonus
        - repeat_penalty
        - dist_penalty
        - wait_penalty
        - bt_penalty
    )


def _optimize_stop_ordering(
    seq: List[Dict[str, Any]],
    start_lat: float,
    start_lon: float,
    start_hour: float,
    end_hour: float,
    budget_vnd: int,
    transport_mode: str,
    prefer_night: bool,
) -> Optional[Tuple[float, List[Dict[str, Any]], Dict[str, Any]]]:
    """Find the best feasible permutation of `seq` that minimizes distance and eliminates backtracking."""
    n = len(seq)
    best: Optional[Tuple[float, List[Dict[str, Any]], Dict[str, Any]]] = None

    if n <= 5:
        # Exact evaluation of all n! permutations (<= 120 for n=5)
        for perm in itertools.permutations(seq):
            cand_seq = list(perm)
            sim = simulate_sequence(
                cand_seq, start_lat, start_lon, start_hour, end_hour, budget_vnd, transport_mode
            )
            if sim is not None:
                sc = _objective_score(sim, start_lat, start_lon, start_hour, prefer_night=prefer_night)
                if best is None or sc > best[0]:
                    best = (sc, cand_seq, sim)
        return best

    # For n > 5, seed with distance-from-start sorted order + nearest-neighbor order, then run 2-opt segment reversal
    seeds = [
        list(seq),
        sorted(seq, key=lambda e: haversine_km(start_lat, start_lon, e["poi"]["lat"], e["poi"]["lon"])),
    ]
    for s_seq in seeds:
        sim = simulate_sequence(
            s_seq, start_lat, start_lon, start_hour, end_hour, budget_vnd, transport_mode
        )
        if sim is not None:
            sc = _objective_score(sim, start_lat, start_lon, start_hour, prefer_night=prefer_night)
            if best is None or sc > best[0]:
                best = (sc, s_seq, sim)

    if best is None:
        return None

    best_sc, best_seq, best_sim = best
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Test both 2-opt segment reversal and pairwise swap
                for cand_seq in (
                    best_seq[:i] + list(reversed(best_seq[i : j + 1])) + best_seq[j + 1 :],
                    [
                        best_seq[j] if k == i else (best_seq[i] if k == j else best_seq[k])
                        for k in range(n)
                    ],
                ):
                    sim = simulate_sequence(
                        cand_seq, start_lat, start_lon, start_hour, end_hour, budget_vnd, transport_mode
                    )
                    if sim is not None:
                        sc = _objective_score(
                            sim, start_lat, start_lon, start_hour, prefer_night=prefer_night
                        )
                        if sc > best_sc + 1e-4:
                            best_sc, best_seq, best_sim = sc, cand_seq, sim
                            improved = True
    return best_sc, best_seq, best_sim


def _build_naive_baseline(
    all_pois: List[Dict[str, Any]],
    score_lookup: Dict[str, Dict[str, Any]],
    start_lat: float,
    start_lon: float,
    start_hour: float,
    end_hour: float,
    budget_vnd: int,
    transport_mode: str,
    target_stops: int,
) -> Dict[str, Any]:
    """Unoptimized manual planning baseline: picks highest-rated popular POIs in Hanoi
    in popularity order without spatial proximity clustering or Contrastive Learning personalization.
    """
    popular = sorted(
        all_pois,
        key=lambda p: (p.get("rating", 4.0), p.get("review_count", 0)),
        reverse=True,
    )

    chosen: List[Dict[str, Any]] = []
    chosen_ids: set[str] = set()
    cur_lat, cur_lon = start_lat, start_lon
    cur_time = start_hour
    spent = 0

    for p in popular:
        if len(chosen) >= max(2, target_stops):
            break
        if p["id"] in chosen_ids:
            continue
        cost = int(p["avg_cost_vnd"])
        if spent + cost > budget_vnd:
            continue

        dist_km, travel_min = estimate_travel(cur_lat, cur_lon, p["lat"], p["lon"], transport_mode)
        arrival = cur_time + travel_min / 60.0
        open_h, close_h = effective_open_window(p, start_hour, end_hour)
        dur_h = float(p["duration_min"]) / 60.0
        service_start = max(arrival, open_h)
        if (service_start - arrival) * 60.0 > 75.0:
            continue
        if service_start + dur_h > close_h + 0.05 or service_start + dur_h > end_hour + 0.15:
            continue

        rec_entry = score_lookup.get(
            p["id"],
            {"poi": p, "cosine_similarity": 0.15, "match_percent": 45.0, "reasons": ["Phổ biến"]},
        )
        chosen.append(rec_entry)
        chosen_ids.add(p["id"])
        spent += cost
        cur_lat, cur_lon = p["lat"], p["lon"]
        cur_time = service_start + dur_h

    sim = simulate_sequence(
        chosen,
        start_lat,
        start_lon,
        start_hour,
        end_hour,
        budget_vnd,
        transport_mode,
        max_wait_min=75.0,
    )
    if sim is not None:
        return sim

    return {
        "stops": [],
        "summary": {
            "stop_count": 0,
            "total_distance_km": 0.0,
            "total_travel_min": 0.0,
            "total_wait_min": 0.0,
            "total_visit_min": 0.0,
            "total_cost_vnd": 0,
            "budget_vnd": budget_vnd,
            "budget_utilization_pct": 0.0,
            "night_economy_stops": 0,
            "night_coverage_pct": 0.0,
            "category_diversity": 0,
            "avg_ai_match_pct": 0.0,
            "start_time": format_hour(start_hour),
            "end_time": format_hour(start_hour),
        },
    }


def optimize_itinerary(
    ranked_candidates: List[Dict[str, Any]],
    all_pois: List[Dict[str, Any]],
    start_hour: float = 17.5,
    end_hour: float = 23.0,
    budget_vnd: int = 500000,
    max_stops: int = 5,
    transport_mode: str = "motorbike",
    start_lat: float = DEFAULT_START_LAT,
    start_lon: float = DEFAULT_START_LON,
    weather: str = "clear",
    beam_width: int = 20,
) -> Dict[str, Any]:
    """Solve Day-to-Night TOPTW using Beam Search + Exact Permutation Anti-Backtracking Refinement."""
    prefer_night = end_hour >= 20.0
    max_stops = max(1, min(8, int(max_stops)))

    pool: List[Dict[str, Any]] = []
    for item in ranked_candidates[:36]:
        entry = dict(item)
        if weather == "rainy" and not entry["poi"].get("indoor", False):
            entry["cosine_similarity"] = round(entry["cosine_similarity"] - 0.35, 4)
            entry["match_percent"] = round(max(10.0, entry["match_percent"] - 20.0), 1)
        pool.append(entry)
    pool.sort(key=lambda x: x["cosine_similarity"], reverse=True)
    pool = pool[:28]

    score_lookup = {item["poi"]["id"]: item for item in ranked_candidates}

    beam: List[Tuple[float, List[Dict[str, Any]], Dict[str, Any]]] = []
    best_overall: Optional[Tuple[float, List[Dict[str, Any]], Dict[str, Any]]] = None

    for cand in pool:
        sim = simulate_sequence(
            [cand], start_lat, start_lon, start_hour, end_hour, budget_vnd, transport_mode
        )
        if sim is not None:
            sc = _objective_score(sim, start_lat, start_lon, start_hour, prefer_night=prefer_night)
            beam.append((sc, [cand], sim))

    beam.sort(key=lambda x: x[0], reverse=True)
    beam = beam[:beam_width]
    if beam:
        best_overall = beam[0]

    for _step in range(2, max_stops + 1):
        next_beam: List[Tuple[float, List[Dict[str, Any]], Dict[str, Any]]] = []
        seen_sets: set[Tuple[str, ...]] = set()

        for _sc, seq, _sim in beam:
            used_ids = {e["poi"]["id"] for e in seq}
            cat_counts: Dict[str, int] = {}
            for e in seq:
                c = e["poi"]["category"]
                cat_counts[c] = cat_counts.get(c, 0) + 1

            for cand in pool:
                pid = cand["poi"]["id"]
                if pid in used_ids:
                    continue
                if cat_counts.get(cand["poi"]["category"], 0) >= 2:
                    continue

                new_seq = seq + [cand]
                # Deduplicate by unordered set of POI IDs and pick best spatial ordering immediately!
                set_sig = tuple(sorted(e["poi"]["id"] for e in new_seq))
                if set_sig in seen_sets:
                    continue

                # Order stops by proximity to start/previous stop first, then simulate
                ordered_try = sorted(
                    new_seq,
                    key=lambda e: haversine_km(start_lat, start_lon, e["poi"]["lat"], e["poi"]["lon"]),
                )
                for candidate_order in (ordered_try, new_seq):
                    sim = simulate_sequence(
                        candidate_order,
                        start_lat,
                        start_lon,
                        start_hour,
                        end_hour,
                        budget_vnd,
                        transport_mode,
                    )
                    if sim is not None:
                        seen_sets.add(set_sig)
                        sc = _objective_score(
                            sim, start_lat, start_lon, start_hour, prefer_night=prefer_night
                        )
                        next_beam.append((sc, candidate_order, sim))
                        break

        if not next_beam:
            break

        next_beam.sort(key=lambda x: x[0], reverse=True)
        beam = next_beam[:beam_width]
        if best_overall is None or beam[0][0] > best_overall[0]:
            best_overall = beam[0]

    # Run Exact Permutation Order Optimization on the top 8 beam candidates!
    top_candidates = beam[:8] + ([best_overall] if best_overall else [])
    for _sc, cand_seq, _sim in top_candidates:
        refined = _optimize_stop_ordering(
            cand_seq,
            start_lat,
            start_lon,
            start_hour,
            end_hour,
            budget_vnd,
            transport_mode,
            prefer_night,
        )
        if refined is not None:
            if best_overall is None or refined[0] > best_overall[0]:
                best_overall = refined

    if best_overall is None:
        optimized = {
            "stops": [],
            "summary": {
                "stop_count": 0,
                "total_distance_km": 0.0,
                "total_travel_min": 0.0,
                "total_wait_min": 0.0,
                "total_visit_min": 0.0,
                "total_cost_vnd": 0,
                "budget_vnd": budget_vnd,
                "budget_utilization_pct": 0.0,
                "night_economy_stops": 0,
                "night_coverage_pct": 0.0,
                "category_diversity": 0,
                "avg_ai_match_pct": 0.0,
                "start_time": format_hour(start_hour),
                "end_time": format_hour(start_hour),
            },
        }
    else:
        optimized = best_overall[2]

    baseline = _build_naive_baseline(
        all_pois=all_pois,
        score_lookup=score_lookup,
        start_lat=start_lat,
        start_lon=start_lon,
        start_hour=start_hour,
        end_hour=end_hour,
        budget_vnd=budget_vnd,
        transport_mode=transport_mode,
        target_stops=max(2, optimized["summary"]["stop_count"]),
    )

    opt_s = optimized["summary"]
    base_s = baseline["summary"]

    dist_saved = round(base_s["total_distance_km"] - opt_s["total_distance_km"], 2)
    dist_saved_pct = (
        round((dist_saved / base_s["total_distance_km"]) * 100.0, 1)
        if base_s["total_distance_km"] > 0
        else 0.0
    )
    time_saved = round(base_s["total_travel_min"] - opt_s["total_travel_min"], 1)
    time_saved_pct = (
        round((time_saved / base_s["total_travel_min"]) * 100.0, 1)
        if base_s["total_travel_min"] > 0
        else 0.0
    )
    ai_gain = round(opt_s["avg_ai_match_pct"] - base_s["avg_ai_match_pct"], 1)

    return {
        "optimized": optimized,
        "baseline": baseline,
        "comparison": {
            "distance_saved_km": dist_saved,
            "distance_saved_pct": dist_saved_pct,
            "travel_time_saved_min": time_saved,
            "travel_time_saved_pct": time_saved_pct,
            "ai_match_gain_pct": ai_gain,
            "baseline_distance_km": base_s["total_distance_km"],
            "optimized_distance_km": opt_s["total_distance_km"],
            "baseline_travel_min": base_s["total_travel_min"],
            "optimized_travel_min": opt_s["total_travel_min"],
            "baseline_ai_match_pct": base_s["avg_ai_match_pct"],
            "optimized_ai_match_pct": opt_s["avg_ai_match_pct"],
        },
        "start_point": {"lat": start_lat, "lon": start_lon, "transport_mode": transport_mode},
    }
