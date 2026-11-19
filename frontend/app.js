/* Hanoi Night-Tourism AI - Frontend Interactive Controller (v2.0) */

const CAT_META = {
  heritage_night: { label: "Di sản đêm", color: "#fbbf24", icon: "fa-landmark" },
  night_food: { label: "Ẩm thực đêm", color: "#fb7185", icon: "fa-utensils" },
  walking_market: { label: "Phố đi bộ / Chợ đêm", color: "#34d399", icon: "fa-store" },
  nightlife_cafe: { label: "Cafe / Bar / Jazz", color: "#a78bfa", icon: "fa-martini-glass-citrus" },
  culture_art: { label: "Văn hóa / Nghệ thuật", color: "#22d3ee", icon: "fa-masks-theater" },
  scenic_view: { label: "Ngắm cảnh đêm", color: "#38bdf8", icon: "fa-binoculars" },
  daytime_heritage: { label: "Di tích ban ngày", color: "#94a3b8", icon: "fa-building-columns" },
};

/* Distinct vibrant colors for each travel leg (S->1, 1->2, 2->3, 3->4, 4->5, 5->6) */
const LEG_PALETTE = [
  { hex: "#f59e0b", name: "Vàng Hổ phách" },
  { hex: "#06b6d4", name: "Xanh Ngọc Cyan" },
  { hex: "#ec4899", name: "Hồng Neon" },
  { hex: "#10b981", name: "Xanh Lục bảo" },
  { hex: "#a855f7", name: "Tím Violet" },
  { hex: "#f97316", name: "Cam San hô" },
  { hex: "#eab308", name: "Vàng Chanh" },
  { hex: "#3b82f6", name: "Xanh Lam" },
];

const PRESETS = {
  heritage_food: {
    query: "khám phá tour đêm di tích lịch sử, thưởng thức ẩm thực phố cổ và dạo không gian đi bộ Hà Nội",
    start: "17.5",
    end: "23.5",
    budget: 550000,
    stops: 4,
    transport: "motorbike",
    weather: "clear",
    companion: "friends",
    district: "all",
    cats: ["heritage_night", "night_food", "walking_market"],
    nightOnly: false,
  },
  sleepless: {
    query: "khám phá Hà Nội không ngủ sau 21h ăn đêm Tống Duy Tân bia Tạ Hiện phở gánh và chợ hoa Quảng Bá",
    start: "21",
    end: "26",
    budget: 450000,
    stops: 4,
    transport: "motorbike",
    weather: "clear",
    companion: "friends",
    district: "all",
    cats: ["night_food", "nightlife_cafe", "walking_market"],
    nightOnly: true,
  },
  romantic: {
    query: "hẹn hò lãng mạn ngắm toàn cảnh Hồ Tây rooftop bar nghe nhạc jazz Nhà hát Lớn và cocktail yên tĩnh",
    start: "17.5",
    end: "23.5",
    budget: 900000,
    stops: 4,
    transport: "car",
    weather: "cool",
    companion: "couple",
    district: "all",
    cats: ["scenic_view", "nightlife_cafe", "culture_art"],
    nightOnly: true,
  },
  rainy: {
    query: "trời mưa buổi tối đi chơi trong nhà ấm cúng xem múa rối nước ăn chả cá và uống cà phê trứng nhạc acoustic",
    start: "18.5",
    end: "22.5",
    budget: 500000,
    stops: 3,
    transport: "car",
    weather: "rainy",
    companion: "couple",
    district: "Hoàn Kiếm",
    cats: ["culture_art", "night_food", "nightlife_cafe", "heritage_night"],
    nightOnly: true,
  },
  budget: {
    query: "sau 0h đêm đi ăn đêm cháo sườn bánh mì dân tổ phở gánh xôi đêm giá rẻ bình dân",
    start: "24",
    end: "27.5",
    budget: 150000,
    stops: 3,
    transport: "motorbike",
    weather: "clear",
    companion: "friends",
    district: "Hoàn Kiếm",
    cats: ["night_food"],
    nightOnly: true,
  },
};

const state = {
  pois: [],
  lastOptimize: null,
  showBaselineRoute: false,
  darkTiles: true,
  pickingMapStart: false,
  map: null,
  startMarker: null,
  tileLayer: null,
  poiLayer: null,
  routeLayer: null,
  baselineLayer: null,
  charts: {},
  analyticsLoaded: false,
  recognition: null,
  isRecording: false,
};

function fmtVnd(amount) {
  if (!amount) return "Miễn phí";
  return `${Number(amount).toLocaleString("vi-VN")}đ`;
}

function fmtHour(h) {
  const next = h >= 24;
  const mod = h % 24;
  const hh = Math.floor(mod);
  const mm = Math.round((mod - hh) * 60);
  return `${String(hh).padStart(2, "0")}:${String(mm).padStart(2, "0")}${next ? " (+1)" : ""}`;
}

function escapeHtml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function simpleMarkdown(text) {
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, '<strong class="text-amber-300">$1</strong>')
    .replace(/\*(.+?)\*/g, '<em class="text-slate-300">$1</em>')
    .replace(/`(.+?)`/g, '<code class="px-1.5 py-0.5 rounded bg-slate-800 text-cyan-300 text-[11px]">$1</code>')
    .replace(/\n/g, "<br/>");
}

function stripMarkdownForSpeech(text) {
  return String(text || "")
    .replace(/\*\*(.+?)\*\*/g, "$1")
    .replace(/\*(.+?)\*/g, "$1")
    .replace(/`(.+?)`/g, "$1")
    .replace(/[•💡🤖🏛️🌙✨🌧️🎒]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/* ================= START LOCATION: GPS + ADDRESS SEARCH + MAP PICKER ================= */
function updateStartCoordinates(lat, lon, labelText, triggerOptimize = false) {
  const clampedLat = Math.max(20.90, Math.min(21.15, Number(lat)));
  const clampedLon = Math.max(105.70, Math.min(105.98, Number(lon)));

  document.getElementById("input-start-lat").value = clampedLat.toFixed(5);
  document.getElementById("input-start-lon").value = clampedLon.toFixed(5);
  document.getElementById("start-coord-badge").textContent = `${clampedLat.toFixed(4)}, ${clampedLon.toFixed(4)}`;

  if (labelText) {
    document.getElementById("start-address-input").value = labelText;
  }

  if (state.startMarker) {
    state.startMarker.setLatLng([clampedLat, clampedLon]);
  }
  if (triggerOptimize) {
    runOptimization();
  }
}

async function reverseGeocode(lat, lon) {
  try {
    const resp = await fetch(
      `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lon}&zoom=18&addressdetails=1`,
      { headers: { "Accept-Language": "vi" }, signal: AbortSignal.timeout(3500) }
    );
    if (resp.ok) {
      const data = await resp.json();
      if (data && data.display_name) {
        const shortName = data.display_name.split(",").slice(0, 3).join(",").trim();
        return shortName;
      }
    }
  } catch (_e) {
    // Fallback to nearest local POI name
  }
  if (state.pois.length > 0) {
    let bestPoi = state.pois[0];
    let bestD = 999;
    state.pois.forEach((p) => {
      const d = Math.hypot(p.lat - lat, p.lon - lon);
      if (d < bestD) {
        bestD = d;
        bestPoi = p;
      }
    });
    return `Gần ${bestPoi.name} (${bestPoi.district})`;
  }
  return `Vị trí (${lat.toFixed(4)}, ${lon.toFixed(4)})`;
}

async function searchStartAddress() {
  const query = document.getElementById("start-address-input").value.trim();
  const statusEl = document.getElementById("start-location-status");
  if (!query) return;

  statusEl.innerHTML = '<i class="fa-solid fa-spinner fa-spin mr-1 text-cyan-400"></i>Đang tìm tọa độ địa chỉ...';

  // 1. Check local 255 Hanoi POIs first for instant exact match
  const lower = query.toLowerCase();
  const matchedPoi = state.pois.find(
    (p) =>
      p.name.toLowerCase().includes(lower) ||
      (p.name_en && p.name_en.toLowerCase().includes(lower)) ||
      (p.address && p.address.toLowerCase().includes(lower))
  );

  if (matchedPoi) {
    updateStartCoordinates(matchedPoi.lat, matchedPoi.lon, `${matchedPoi.name} (${matchedPoi.district})`, true);
    state.map.setView([matchedPoi.lat, matchedPoi.lon], 15);
    statusEl.innerHTML = `<span class="text-emerald-400">✓ Đã đặt điểm xuất phát tại: <b>${escapeHtml(matchedPoi.name)}</b></span>`;
    return;
  }

  // 2. Query OpenStreetMap Nominatim Geocoder within Hanoi
  try {
    const qHanoi = query.toLowerCase().includes("hà nội") || query.toLowerCase().includes("hanoi")
      ? query
      : `${query}, Hà Nội, Việt Nam`;
    const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(qHanoi)}&limit=1`;
    const resp = await fetch(url, { headers: { "Accept-Language": "vi" }, signal: AbortSignal.timeout(4500) });
    if (resp.ok) {
      const results = await resp.json();
      if (results && results.length > 0) {
        const hit = results[0];
        const lat = Number(hit.lat);
        const lon = Number(hit.lon);
        if (lat >= 20.90 && lat <= 21.15 && lon >= 105.70 && lon <= 105.98) {
          const label = hit.display_name.split(",").slice(0, 3).join(",").trim();
          updateStartCoordinates(lat, lon, label, true);
          state.map.setView([lat, lon], 15);
          statusEl.innerHTML = `<span class="text-emerald-400">✓ Đã tìm thấy: <b>${escapeHtml(label)}</b></span>`;
          return;
        }
      }
    }
  } catch (_e) {
    // ignore network failure
  }

  statusEl.innerHTML = `<span class="text-amber-300">⚠️ Không tìm thấy địa chỉ chính xác, hãy thử tên đường/quận tại Hà Nội hoặc chấm trực tiếp trên bản đồ!</span>`;
}

function requestBrowserGeolocation(silentOnLoad = false) {
  const statusEl = document.getElementById("start-location-status");
  if (!navigator.geolocation) {
    if (!silentOnLoad) {
      statusEl.innerHTML = '<span class="text-rose-400">Trình duyệt của bạn không hỗ trợ định vị GPS.</span>';
    }
    return;
  }

  statusEl.innerHTML = '<i class="fa-solid fa-spinner fa-spin mr-1 text-emerald-400"></i>Đang yêu cầu quyền truy cập vị trí GPS của bạn...';

  navigator.geolocation.getCurrentPosition(
    async (pos) => {
      const { latitude, longitude } = pos.coords;
      // Check if user's physical GPS is within Hanoi area
      const inHanoi = latitude >= 20.90 && latitude <= 21.15 && longitude >= 105.70 && longitude <= 105.98;
      if (inHanoi) {
        const addr = await reverseGeocode(latitude, longitude);
        updateStartCoordinates(latitude, longitude, addr, true);
        state.map.setView([latitude, longitude], 14);
        statusEl.innerHTML = `<span class="text-emerald-400">✓ Đã lấy vị trí GPS hiện tại: <b>${escapeHtml(addr)}</b></span>`;
      } else {
        // User's GPS is outside Hanoi (e.g., testing from another province/ISP)
        statusEl.innerHTML = `<span class="text-amber-300">📍 Đã nhận GPS thực (${latitude.toFixed(3)}, ${longitude.toFixed(3)}) nhưng ngoài phạm vi nội đô Hà Nội nên giữ điểm xuất phát tại Trung tâm Hà Nội (bạn có thể tự nhập hoặc chấm điểm bất kỳ).</span>`;
      }
    },
    (err) => {
      if (!silentOnLoad) {
        statusEl.innerHTML = `<span class="text-amber-300">⚠️ Chưa cấp quyền GPS (${escapeHtml(err.message)}). Bạn có thể tự nhập địa chỉ hoặc bấm "Chấm trên bản đồ".</span>`;
      } else {
        statusEl.innerHTML = `💡 Bấm <b>Định vị GPS của tôi</b> để dùng vị trí thực tế, hoặc tự nhập địa chỉ / chấm trên bản đồ.`;
      }
    },
    { enableHighAccuracy: true, timeout: 6000, maximumAge: 60000 }
  );
}

/* ================= MAP SETUP & PARALLEL OFFSET MULTI-COLOR LEGS ================= */
function initMap() {
  state.map = L.map("hanoi-map", { zoomControl: true }).setView([21.0313, 105.849], 13);
  setMapTiles(true);
  state.poiLayer = L.layerGroup().addTo(state.map);
  state.baselineLayer = L.layerGroup().addTo(state.map);
  state.routeLayer = L.layerGroup().addTo(state.map);

  // Map click handler for picking start location
  state.map.on("click", async (e) => {
    if (!state.pickingMapStart) return;
    state.pickingMapStart = false;
    document.getElementById("hanoi-map").classList.remove("picking-location");
    const { lat, lng } = e.latlng;
    const addr = await reverseGeocode(lat, lng);
    updateStartCoordinates(lat, lng, addr, true);
    document.getElementById("start-location-status").innerHTML =
      `<span class="text-emerald-400">✓ Đã chọn điểm xuất phát trên bản đồ: <b>${escapeHtml(addr)}</b></span>`;
  });
}

function setMapTiles(dark) {
  state.darkTiles = dark;
  if (state.tileLayer) {
    state.map.removeLayer(state.tileLayer);
  }
  const url = dark
    ? "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
    : "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
  const attr = dark ? "&copy; OpenStreetMap &copy; CARTO" : "&copy; OpenStreetMap contributors";
  state.tileLayer = L.tileLayer(url, { attribution: attr, maxZoom: 19 }).addTo(state.map);
}

function renderAllPoisOnMap() {
  if (!state.poiLayer) return;
  state.poiLayer.clearLayers();
  state.pois.forEach((p) => {
    const meta = CAT_META[p.category] || { label: p.category, color: "#94a3b8" };
    const marker = L.circleMarker([p.lat, p.lon], {
      radius: p.source && p.source.startsWith("curated") ? 5.2 : 3.8,
      fillColor: meta.color,
      color: "#0f172a",
      weight: 1,
      opacity: 0.9,
      fillOpacity: 0.78,
    });
    marker.bindPopup(`
      <div style="min-width:215px">
        <div style="font-weight:700;color:#fbbf24;font-size:13px;margin-bottom:3px">${escapeHtml(p.name)}</div>
        <div style="font-size:11px;color:#94a3b8;margin-bottom:4px">${escapeHtml(meta.label)} • ${escapeHtml(p.district)}</div>
        <div style="font-size:11px;margin-bottom:4px">
          ⏰ ${fmtHour(p.open_hour)} – ${fmtHour(p.close_hour)} &nbsp;|&nbsp; 💰 ${fmtVnd(p.avg_cost_vnd)}
        </div>
        <div style="font-size:11px;color:#cbd5e1;line-height:1.4">${escapeHtml(p.description_vi)}</div>
      </div>
    `);
    marker.addTo(state.poiLayer);
  });
}

async function fetchOsrmLegGeometry(fromCoord, toCoord) {
  try {
    const coordStr = `${fromCoord[1]},${fromCoord[0]};${toCoord[1]},${toCoord[0]}`;
    const resp = await fetch(
      `https://router.project-osrm.org/route/v1/driving/${coordStr}?overview=full&geometries=geojson`,
      { signal: AbortSignal.timeout(3500) }
    );
    if (resp.ok) {
      const data = await resp.json();
      if (data.routes && data.routes.length > 0) {
        return data.routes[0].geometry.coordinates.map(([lon, lat]) => [lat, lon]);
      }
    }
  } catch (_e) {
    // Fallback to straight segment
  }
  return [fromCoord, toCoord];
}

/**
 * Parallel Offset Algorithm for Overlapping Route Segments:
 * Detects segments of leg `legIdx` that share the same street corridor with any earlier leg
 * (`distance < ~22m`) and shifts them laterally perpendicular to the canonical road bearing
 * so overlapping legs render side-by-side in parallel colored lanes instead of covering each other.
 */
function offsetOverlappingLegGeometry(coords, priorSegments) {
  if (coords.length < 2) return coords;

  // Densify long segments slightly for smooth parallel lane rendering
  const dense = [];
  for (let i = 0; i < coords.length - 1; i++) {
    const [lat1, lon1] = coords[i];
    const [lat2, lon2] = coords[i + 1];
    dense.push([lat1, lon1]);
    const distDeg = Math.hypot(lat2 - lat1, lon2 - lon1);
    if (distDeg > 0.0012) {
      const steps = Math.min(8, Math.ceil(distDeg / 0.0005));
      for (let s = 1; s < steps; s++) {
        const t = s / steps;
        dense.push([lat1 + (lat2 - lat1) * t, lon1 + (lon2 - lon1) * t]);
      }
    }
  }
  dense.push(coords[coords.length - 1]);

  const shifted = [];
  const CORRIDOR_THRESHOLD_DEG = 0.00024; // ~25 meters in Hanoi
  const LANE_OFFSET_DEG = 0.00011; // ~12 meters lateral lane separation

  for (let i = 0; i < dense.length; i++) {
    const [lat, lon] = dense[i];
    const prev = dense[Math.max(0, i - 1)];
    const next = dense[Math.min(dense.length - 1, i + 1)];

    let dLat = next[0] - prev[0];
    let dLon = next[1] - prev[1];
    const len = Math.hypot(dLat, dLon) || 1e-9;
    dLat /= len;
    dLon /= len;

    // Canonicalize road tangent direction so forward and reverse directions share the SAME normal axis!
    let cLat = dLat;
    let cLon = dLon;
    if (cLat < -1e-6 || (Math.abs(cLat) <= 1e-6 && cLon < 0)) {
      cLat = -cLat;
      cLon = -cLon;
    }
    // Unit normal vector perpendicular to canonical road direction
    const nLat = -cLon;
    const nLon = cLat;

    // Count how many prior legs pass through this road corridor
    const overlappingLegs = new Set();
    for (const seg of priorSegments) {
      if (Math.hypot(lat - seg.midLat, lon - seg.midLon) < CORRIDOR_THRESHOLD_DEG) {
        overlappingLegs.add(seg.legIdx);
      }
    }

    const overlapCount = overlappingLegs.size;
    if (overlapCount === 0) {
      shifted.push([lat, lon]);
    } else {
      // Alternate left/right parallel lanes (+1, -1, +2, -2...) relative to the original street centerline
      const laneMultiplier = (overlapCount % 2 === 1 ? 1 : -1) * Math.ceil(overlapCount / 2);
      shifted.push([
        lat + nLat * LANE_OFFSET_DEG * laneMultiplier,
        lon + nLon * LANE_OFFSET_DEG * laneMultiplier,
      ]);
    }
  }

  // Register this leg's original segments into priorSegments for subsequent legs
  for (let i = 0; i < dense.length - 1; i++) {
    priorSegments.push({
      legIdx: priorSegments.currentLegIdx || 0,
      midLat: (dense[i][0] + dense[i + 1][0]) * 0.5,
      midLon: (dense[i][1] + dense[i + 1][1]) * 0.5,
    });
  }

  return shifted;
}

async function renderRoutesOnMap(result) {
  state.routeLayer.clearLayers();
  state.baselineLayer.clearLayers();

  const start = result.start_point;
  const optStops = result.optimized.stops || [];
  const baseStops = result.baseline.stops || [];

  // Draggable Start Marker 'S'
  state.startMarker = L.marker([start.lat, start.lon], {
    draggable: true,
    icon: L.divIcon({
      className: "",
      html: '<div class="start-marker-icon" title="Kéo thả để đổi điểm xuất phát">S</div>',
      iconSize: [34, 34],
      iconAnchor: [17, 17],
    }),
  })
    .bindPopup("<b>Điểm xuất phát của bạn (S)</b><br/><span style='font-size:11px;color:#94a3b8'>Có thể kéo thả điểm S tới vị trí bất kỳ</span>")
    .addTo(state.routeLayer);

  state.startMarker.on("dragend", async (ev) => {
    const pos = ev.target.getLatLng();
    const addr = await reverseGeocode(pos.lat, pos.lng);
    updateStartCoordinates(pos.lat, pos.lng, addr, true);
    document.getElementById("start-location-status").innerHTML =
      `<span class="text-emerald-400">✓ Đã kéo điểm xuất phát tới: <b>${escapeHtml(addr)}</b></span>`;
  });

  if (state.showBaselineRoute && baseStops.length > 0) {
    const bCoords = [[start.lat, start.lon], ...baseStops.map((s) => [s.poi.lat, s.poi.lon])];
    L.polyline(bCoords, {
      color: "#64748b",
      weight: 3,
      dashArray: "7, 7",
      opacity: 0.75,
    }).addTo(state.baselineLayer);
  }

  const legendEl = document.getElementById("leg-color-legend");
  if (optStops.length === 0) {
    legendEl.innerHTML = "";
    return;
  }

  const waypoints = [
    { lat: start.lat, lon: start.lon, label: "Xuất phát (S)" },
    ...optStops.map((s) => ({
      lat: s.poi.lat,
      lon: s.poi.lon,
      label: `#${s.order} ${s.poi.name}`,
      stop: s,
    })),
  ];

  // Fetch each leg's street polyline in parallel
  const rawLegGeoms = await Promise.all(
    optStops.map((_s, idx) =>
      fetchOsrmLegGeometry(
        [waypoints[idx].lat, waypoints[idx].lon],
        [waypoints[idx + 1].lat, waypoints[idx + 1].lon]
      )
    )
  );

  const priorSegments = [];
  const legendItemsHtml = [];

  rawLegGeoms.forEach((geom, idx) => {
    const palette = LEG_PALETTE[idx % LEG_PALETTE.length];
    const stop = optStops[idx];
    const fromTag = idx === 0 ? "S" : `#${idx}`;
    const toTag = `#${stop.order}`;

    priorSegments.currentLegIdx = idx + 1;
    const offsetGeom = offsetOverlappingLegGeometry(geom, priorSegments);

    // Dark casing underneath so side-by-side parallel colored lines stand out sharply
    L.polyline(offsetGeom, {
      color: "#020617",
      weight: 7.5,
      opacity: 0.85,
      lineCap: "round",
      lineJoin: "round",
    }).addTo(state.routeLayer);

    // Colored leg line
    const legLine = L.polyline(offsetGeom, {
      color: palette.hex,
      weight: 4.8,
      opacity: 0.98,
      lineCap: "round",
      lineJoin: "round",
    }).addTo(state.routeLayer);

    legLine.bindTooltip(
      `<b>Chặng ${idx + 1} (${fromTag} → ${toTag})</b>: ${stop.distance_from_prev_km} km • ${stop.travel_from_prev_min} phút`,
      { sticky: true }
    );

    legendItemsHtml.push(`
      <span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-900/90 border border-slate-700/80">
        <span class="w-3.5 h-1.5 rounded-full" style="background:${palette.hex}"></span>
        <b style="color:${palette.hex}">Chặng ${idx + 1} (${fromTag} → ${toTag})</b>
        <span class="text-slate-400">${stop.distance_from_prev_km}km • ${stop.travel_from_prev_min}p</span>
      </span>
    `);
  });

  legendEl.innerHTML = legendItemsHtml.join("");

  // Draw numbered markers colored by their incoming leg color
  optStops.forEach((s, idx) => {
    const p = s.poi;
    const palette = LEG_PALETTE[idx % LEG_PALETTE.length];
    L.marker([p.lat, p.lon], {
      icon: L.divIcon({
        className: "",
        html: `<div class="route-marker-icon" style="background:${palette.hex}">${s.order}</div>`,
        iconSize: [32, 32],
        iconAnchor: [16, 16],
      }),
    })
      .bindPopup(`
        <div style="min-width:225px">
          <div style="font-weight:800;color:${palette.hex};font-size:13px">#${s.order} ${escapeHtml(p.name)}</div>
          <div style="font-size:11px;color:#22d3ee;margin:3px 0">
            ⏰ ${s.start_time} – ${s.departure_time} &nbsp;|&nbsp; AI Match: ${s.match_percent}%
          </div>
          <div style="font-size:11px;margin-bottom:4px">
            🚗 Chặng ${idx + 1}: ${s.distance_from_prev_km} km (${s.travel_from_prev_min}p) | 💰 ${fmtVnd(p.avg_cost_vnd)}
          </div>
          <div style="font-size:11px;color:#cbd5e1">${escapeHtml(p.description_vi)}</div>
        </div>
      `)
      .addTo(state.routeLayer);
  });

  const allBoundsCoords = waypoints.map((w) => [w.lat, w.lon]);
  state.map.fitBounds(L.latLngBounds(allBoundsCoords).pad(0.24));
}

/* ================= OPTIMIZATION & RECOMMENDATIONS ================= */
function collectPlannerPayload() {
  const lat = Number(document.getElementById("input-start-lat").value) || 21.0313;
  const lon = Number(document.getElementById("input-start-lon").value) || 105.8520;
  const selectedCats = [...document.querySelectorAll("#category-chips .chip-btn.selected")].map(
    (btn) => btn.dataset.cat
  );
  return {
    query_text: document.getElementById("input-query").value.trim() || "du lịch đêm Hà Nội",
    start_hour: Number(document.getElementById("input-start").value),
    end_hour: Number(document.getElementById("input-end").value),
    budget_vnd: Number(document.getElementById("input-budget").value),
    max_stops: Number(document.getElementById("input-stops").value),
    transport_mode: document.getElementById("input-transport").value,
    weather: document.getElementById("input-weather").value,
    companion: document.getElementById("input-companion").value,
    preferred_district: document.getElementById("input-district").value,
    preferred_categories: selectedCats.length ? selectedCats : ["heritage_night", "night_food"],
    start_lat: lat,
    start_lon: lon,
    only_night_economy: document.getElementById("input-night-only").checked,
  };
}

async function runOptimization() {
  const btn = document.getElementById("btn-optimize");
  const payload = collectPlannerPayload();
  if (payload.end_hour <= payload.start_hour) {
    alert("Giờ kết thúc phải lớn hơn giờ bắt đầu!");
    return;
  }

  btn.disabled = true;
  btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i><span>Đang chạy Contrastive Learning + TOPTW...</span>';

  try {
    const [optRes, recRes] = await Promise.all([
      fetch("/api/optimize-itinerary", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
      fetch("/api/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...payload, top_k: 8, enforce_open_window: true }),
      }),
    ]);

    if (!optRes.ok) throw new Error(await optRes.text());
    const optData = await optRes.json();
    const recData = await recRes.json();

    state.lastOptimize = optData;
    renderTimeline(optData);
    renderComparison(optData);
    renderMiniRecs(recData.recommendations || []);
    await renderRoutesOnMap(optData);
  } catch (err) {
    console.error(err);
    document.getElementById("timeline-container").innerHTML =
      `<div class="text-xs text-rose-400">Lỗi khi tối ưu lịch trình: ${escapeHtml(err.message)}</div>`;
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<i class="fa-solid fa-sparkles"></i><span>Tối ưu hóa Lộ trình AI (Contrastive + TOPTW)</span>';
  }
}

function renderTimeline(data) {
  const stops = data.optimized.stops || [];
  const s = data.optimized.summary;
  document.getElementById("itinerary-span-badge").textContent =
    `${s.start_time} → ${s.end_time} • ${s.stop_count} điểm • ${fmtVnd(s.total_cost_vnd)}`;

  const container = document.getElementById("timeline-container");
  if (!stops.length) {
    container.innerHTML = `<div class="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 text-xs text-amber-200">
      Không tìm thấy lịch trình thỏa mãn đồng thời khung giờ và ngân sách quá hẹp. Hãy thử nới rộng khung giờ hoặc ngân sách!
    </div>`;
    return;
  }

  container.innerHTML = stops
    .map((stop, idx) => {
      const p = stop.poi;
      const meta = CAT_META[p.category] || { label: p.category, color: "#f59e0b" };
      const legColor = LEG_PALETTE[idx % LEG_PALETTE.length];
      const fromTag = idx === 0 ? "S" : `#${idx}`;
      const reasons = (stop.reasons || [])
        .map((r) => `<span class="text-[10px] px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700">${escapeHtml(r)}</span>`)
        .join("");

      return `
        <div class="timeline-item">
          <div class="absolute left-0 top-1 w-8 h-8 rounded-full flex items-center justify-center font-extrabold text-xs text-slate-950 shadow-md" style="background:${legColor.hex}">
            ${stop.order}
          </div>
          <div class="glass-sub rounded-xl p-3.5 hover:border-amber-400/40 transition" style="border-left: 3px solid ${legColor.hex}">
            <div class="flex flex-wrap items-start justify-between gap-2 mb-1">
              <div>
                <div class="flex items-center gap-2">
                  <span class="text-[10px] font-bold px-2 py-0.5 rounded" style="background:${legColor.hex}22;color:${legColor.hex}">
                    Chặng ${idx + 1} (${fromTag} → #${stop.order})
                  </span>
                  <h4 class="text-sm font-bold text-white">${escapeHtml(p.name)}</h4>
                </div>
                <p class="text-[11px] text-slate-400 mt-0.5">${escapeHtml(meta.label)} • ${escapeHtml(p.district)} • ${escapeHtml(p.address)}</p>
              </div>
              <div class="text-right">
                <span class="inline-block text-[11px] font-bold px-2 py-0.5 rounded-lg bg-amber-500/20 text-amber-300 border border-amber-500/30">
                  AI Match ${stop.match_percent}%
                </span>
                <div class="text-[10px] text-slate-400 mt-0.5">cos = ${stop.cosine_similarity}</div>
              </div>
            </div>

            <div class="flex flex-wrap items-center gap-3 text-[11px] text-cyan-300 my-2">
              <span><i class="fa-regular fa-clock mr-1"></i><b>${stop.start_time} – ${stop.departure_time}</b> (${p.duration_min}p)</span>
              <span style="color:${legColor.hex}"><i class="fa-solid fa-route mr-1"></i>Di chuyển: <b>${stop.distance_from_prev_km} km</b> (${stop.travel_from_prev_min}p)</span>
              <span><i class="fa-solid fa-wallet mr-1 text-emerald-400"></i>${fmtVnd(p.avg_cost_vnd)}</span>
              ${stop.wait_min > 0 ? `<span class="text-amber-300"><i class="fa-solid fa-hourglass-half mr-1"></i>Chờ mở cửa: ${stop.wait_min}p</span>` : ""}
            </div>

            <p class="text-xs text-slate-300 leading-relaxed mb-2">${escapeHtml(p.description_vi)}</p>
            <div class="flex flex-wrap gap-1.5">${reasons}</div>
          </div>
        </div>
      `;
    })
    .join("");
}

function renderComparison(data) {
  const c = data.comparison;
  const opt = data.optimized.summary;
  const base = data.baseline.summary;

  const distSign = c.distance_saved_km >= 0 ? "-" : "+";
  const timeSign = c.travel_time_saved_min >= 0 ? "-" : "+";

  document.getElementById("comparison-cards").innerHTML = `
    <div class="glass-sub rounded-xl p-3">
      <div class="text-[11px] text-slate-400">Giảm quãng đường</div>
      <div class="text-lg font-extrabold ${c.distance_saved_km >= 0 ? "text-emerald-400" : "text-amber-300"}">
        ${distSign}${Math.abs(c.distance_saved_km)} km
      </div>
      <div class="text-[11px] text-slate-400">${c.distance_saved_pct >= 0 ? `Tiết kiệm ${c.distance_saved_pct}%` : "Tối ưu theo gu cá nhân"}</div>
    </div>
    <div class="glass-sub rounded-xl p-3">
      <div class="text-[11px] text-slate-400">Độ phù hợp AI tăng</div>
      <div class="text-lg font-extrabold text-amber-400">
        ${c.ai_match_gain_pct >= 0 ? "+" : ""}${c.ai_match_gain_pct}%
      </div>
      <div class="text-[11px] text-slate-400">So với chọn ngẫu nhiên</div>
    </div>
    <div class="glass-sub rounded-xl p-3">
      <div class="text-[11px] text-slate-400">Độ phủ Kinh tế đêm</div>
      <div class="text-lg font-extrabold text-cyan-400">${opt.night_coverage_pct}%</div>
      <div class="text-[11px] text-slate-400">${opt.night_economy_stops}/${opt.stop_count} điểm sau 18h</div>
    </div>
    <div class="glass-sub rounded-xl p-3">
      <div class="text-[11px] text-slate-400">Thời gian di chuyển</div>
      <div class="text-lg font-extrabold text-violet-300">${timeSign}${Math.abs(c.travel_time_saved_min)} phút</div>
      <div class="text-[11px] text-slate-400">Tổng ${opt.total_travel_min}p di chuyển</div>
    </div>
  `;

  document.getElementById("comparison-table-body").innerHTML = `
    <tr>
      <td class="py-2 pr-2">Tổng quãng đường</td>
      <td class="py-2 px-2 text-right">${base.total_distance_km} km</td>
      <td class="py-2 pl-2 text-right font-bold text-emerald-400">${opt.total_distance_km} km</td>
    </tr>
    <tr>
      <td class="py-2 pr-2">Thời gian di chuyển</td>
      <td class="py-2 px-2 text-right">${base.total_travel_min} phút</td>
      <td class="py-2 pl-2 text-right font-bold text-emerald-400">${opt.total_travel_min} phút</td>
    </tr>
    <tr>
      <td class="py-2 pr-2">Độ khớp sở thích AI</td>
      <td class="py-2 px-2 text-right">${base.avg_ai_match_pct}%</td>
      <td class="py-2 pl-2 text-right font-bold text-amber-300">${opt.avg_ai_match_pct}%</td>
    </tr>
    <tr>
      <td class="py-2 pr-2">Đa dạng loại hình</td>
      <td class="py-2 px-2 text-right">${base.category_diversity} nhóm</td>
      <td class="py-2 pl-2 text-right font-bold text-cyan-300">${opt.category_diversity} nhóm</td>
    </tr>
    <tr>
      <td class="py-2 pr-2">Chi phí / Ngân sách</td>
      <td class="py-2 px-2 text-right">${fmtVnd(base.total_cost_vnd)}</td>
      <td class="py-2 pl-2 text-right font-bold text-white">${fmtVnd(opt.total_cost_vnd)} (${opt.budget_utilization_pct}%)</td>
    </tr>
  `;
}

function renderMiniRecs(recs) {
  const el = document.getElementById("top-recs-mini");
  el.innerHTML = recs
    .map((r, i) => {
      const p = r.poi;
      return `
        <div class="glass-sub rounded-xl px-3 py-2 flex items-center justify-between gap-2 text-xs">
          <div class="min-w-0">
            <div class="font-semibold text-white truncate">${i + 1}. ${escapeHtml(p.name)}</div>
            <div class="text-[10px] text-slate-400">${escapeHtml(p.district)} • ${fmtHour(p.open_hour)}–${fmtHour(p.close_hour)} • ${fmtVnd(p.avg_cost_vnd)}</div>
          </div>
          <span class="shrink-0 text-[11px] font-bold text-amber-300 bg-amber-500/15 px-2 py-0.5 rounded-md">${r.match_percent}%</span>
        </div>
      `;
    })
    .join("");
}

/* ================= POI EXPLORER & ADMIN CRUD ================= */
async function loadPois() {
  const q = document.getElementById("poi-search").value.trim();
  const cat = document.getElementById("poi-filter-cat").value;
  const dist = document.getElementById("poi-filter-dist").value;
  const night = document.getElementById("poi-filter-night").checked;

  const params = new URLSearchParams({ limit: "350" });
  if (q) params.set("q", q);
  if (cat !== "all") params.set("category", cat);
  if (dist !== "all") params.set("district", dist);
  if (night) params.set("night_only", "true");

  const resp = await fetch(`/api/pois?${params.toString()}`);
  const data = await resp.json();

  if (!q && cat === "all" && dist === "all" && !night) {
    state.pois = data.items;
    renderAllPoisOnMap();
  }
  renderPoiCards(data.items, data.total);
}

function renderPoiCards(items, total) {
  document.getElementById("poi-count-label").textContent = `Hiển thị ${items.length} / ${total} địa điểm du lịch Hà Nội`;
  const grid = document.getElementById("poi-grid");
  grid.innerHTML = items
    .map((p) => {
      const meta = CAT_META[p.category] || { label: p.category, color: "#94a3b8" };
      const isCustom = p.source === "admin_created";
      return `
        <div class="glass-card rounded-2xl p-4 flex flex-col justify-between gap-2.5 hover:border-amber-400/40 transition">
          <div>
            <div class="flex items-start justify-between gap-2 mb-1">
              <span class="text-[10px] font-semibold px-2 py-0.5 rounded-full" style="background:${meta.color}22;color:${meta.color};border:1px solid ${meta.color}55">
                ${escapeHtml(meta.label)}
              </span>
              <span class="text-[11px] text-amber-300 font-semibold"><i class="fa-solid fa-star text-[10px] mr-0.5"></i>${p.rating}</span>
            </div>
            <h4 class="text-sm font-bold text-white mt-1">${escapeHtml(p.name)}</h4>
            <p class="text-[11px] text-slate-400 mb-2"><i class="fa-solid fa-location-dot mr-1"></i>${escapeHtml(p.address)}</p>
            <p class="text-xs text-slate-300 line-clamp-3 leading-relaxed">${escapeHtml(p.description_vi)}</p>
          </div>
          <div class="pt-2 border-t border-slate-800 flex flex-wrap items-center justify-between gap-2 text-[11px]">
            <span class="text-cyan-300">⏰ ${fmtHour(p.open_hour)}–${fmtHour(p.close_hour)}</span>
            <span class="text-emerald-300 font-semibold">${fmtVnd(p.avg_cost_vnd)}</span>
            <span class="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">${escapeHtml(p.source)}</span>
            ${
              isCustom
                ? `<button onclick="deletePoi('${p.id}')" class="text-rose-400 hover:text-rose-300 text-xs" title="Xóa điểm tùy chỉnh"><i class="fa-solid fa-trash"></i></button>`
                : ""
            }
          </div>
        </div>
      `;
    })
    .join("");
}

window.deletePoi = async function (poiId) {
  if (!confirm("Xóa điểm đến này khỏi cơ sở dữ liệu?")) return;
  const resp = await fetch(`/api/pois/${poiId}`, { method: "DELETE" });
  if (resp.ok) {
    await loadPois();
    state.analyticsLoaded = false;
  }
};

/* ================= VOICE CHAT ASSISTANT (SPEECH-TO-TEXT & TEXT-TO-SPEECH) ================= */
window.speakText = function (plainText, langCode = "vi") {
  if (!("speechSynthesis" in window)) {
    alert("Trình duyệt không hỗ trợ đọc giọng nói (Web Speech Synthesis).");
    return;
  }
  window.speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(stripMarkdownForSpeech(plainText));
  const targetLang = langCode === "en" ? "en-US" : "vi-VN";
  utter.lang = targetLang;
  utter.rate = 1.03;

  const voices = window.speechSynthesis.getVoices();
  const matchedVoice =
    voices.find((v) => v.lang.toLowerCase().startsWith(langCode.toLowerCase())) ||
    voices.find((v) => v.lang.toLowerCase().includes(langCode.toLowerCase()));
  if (matchedVoice) {
    utter.voice = matchedVoice;
  }
  window.speechSynthesis.speak(utter);
};

window.stopSpeaking = function () {
  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
};

function initVoiceRecognition() {
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  const micBtn = document.getElementById("btn-voice-mic");
  const statusBar = document.getElementById("voice-status-bar");
  const statusText = document.getElementById("voice-status-text");

  if (!SpeechRec) {
    micBtn.addEventListener("click", () => {
      alert("Trình duyệt này không hỗ trợ nhận diện giọng nói (hãy dùng Chrome hoặc Microsoft Edge).");
    });
    return;
  }

  const rec = new SpeechRec();
  rec.continuous = false;
  rec.interimResults = true;

  rec.onstart = () => {
    state.isRecording = true;
    micBtn.classList.add("mic-recording");
    statusBar.classList.remove("hidden");
    statusBar.classList.add("flex");
    statusText.textContent = "Đang lắng nghe bạn nói... Hãy nói yêu cầu du lịch đêm của bạn!";
  };

  rec.onresult = (event) => {
    let transcript = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      transcript += event.results[i][0].transcript;
    }
    document.getElementById("chat-input").value = transcript;
    statusText.textContent = `Đang nghe: "${transcript}"`;
    if (event.results[event.results.length - 1].isFinal && transcript.trim().length > 1) {
      setTimeout(() => {
        const msg = document.getElementById("chat-input").value.trim();
        if (msg) {
          document.getElementById("chat-input").value = "";
          sendChatMessage(msg);
        }
      }, 250);
    }
  };

  rec.onerror = (event) => {
    statusText.textContent = `Lỗi micro: ${event.error}`;
    setTimeout(() => {
      statusBar.classList.add("hidden");
      statusBar.classList.remove("flex");
    }, 2000);
  };

  rec.onend = () => {
    state.isRecording = false;
    micBtn.classList.remove("mic-recording");
    statusBar.classList.add("hidden");
    statusBar.classList.remove("flex");
  };

  state.recognition = rec;

  micBtn.addEventListener("click", () => {
    if (state.isRecording) {
      rec.stop();
    } else {
      window.stopSpeaking();
      rec.lang = document.getElementById("voice-lang-select").value || "vi-VN";
      rec.start();
    }
  });

  document.getElementById("btn-stop-voice").addEventListener("click", () => {
    if (state.isRecording) rec.stop();
  });
}

async function sendChatMessage(message) {
  const box = document.getElementById("chat-messages");
  box.innerHTML += `
    <div class="flex justify-end">
      <div class="rounded-2xl bg-amber-500/20 border border-amber-400/40 px-4 py-2.5 max-w-[80%] text-xs sm:text-sm text-amber-100">
        ${escapeHtml(message)}
      </div>
    </div>
  `;
  box.scrollTop = box.scrollHeight;

  const loadingId = `chat-load-${Date.now()}`;
  box.innerHTML += `
    <div id="${loadingId}" class="glass-sub rounded-2xl p-3 max-w-[70%] text-xs text-slate-400">
      <i class="fa-solid fa-spinner fa-spin mr-1.5 text-amber-400"></i>Đang lọc ngữ cảnh & truy xuất không gian vector Contrastive Learning...
    </div>
  `;
  box.scrollTop = box.scrollHeight;

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const data = await resp.json();
    document.getElementById(loadingId)?.remove();

    const safeSpeechText = stripMarkdownForSpeech(data.reply).replace(/'/g, "\\'").replace(/"/g, "&quot;");
    const lang = data.language || "vi";

    box.innerHTML += `
      <div class="glass-sub rounded-2xl p-4 max-w-[92%] text-xs sm:text-sm text-slate-200 leading-relaxed">
        <div class="flex flex-wrap items-center justify-between gap-2 mb-2">
          <span class="font-bold text-amber-300">🤖 Hanoi Night-Tourism AI</span>
          <div class="flex items-center gap-1.5">
            <button type="button" onclick="speakText('${safeSpeechText}', '${lang}')" class="text-[11px] px-2.5 py-0.5 rounded-lg bg-amber-500/20 border border-amber-500/40 text-amber-300 hover:bg-amber-500/30">
              <i class="fa-solid fa-volume-high mr-1"></i>Nghe
            </button>
            <button type="button" onclick="stopSpeaking()" class="text-[11px] px-2 py-0.5 rounded-lg bg-slate-800 text-slate-400 hover:text-white" title="Dừng đọc">
              <i class="fa-solid fa-stop"></i>
            </button>
            <span class="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-cyan-300">${escapeHtml(data.mode)}</span>
          </div>
        </div>
        <div>${simpleMarkdown(data.reply)}</div>
      </div>
    `;
    box.scrollTop = box.scrollHeight;

    if (document.getElementById("toggle-auto-tts")?.checked) {
      window.speakText(data.reply, lang);
    }
  } catch (err) {
    document.getElementById(loadingId)?.remove();
    box.innerHTML += `<div class="text-xs text-rose-400">Lỗi kết nối trợ lý AI: ${escapeHtml(err.message)}</div>`;
  }
}

/* ================= ANALYTICS DASHBOARD ================= */
async function loadAnalytics() {
  if (state.analyticsLoaded) return;
  const resp = await fetch("/api/metrics");
  const data = await resp.json();
  const ev = data.model_evaluation || {};
  const bench = ev.benchmark_comparison || {};
  const cl = bench.two_tower_contrastive_learning || {};
  const arch = ev.architecture || {};
  const stats = data.dataset_stats || {};

  document.getElementById("analytics-kpis").innerHTML = `
    <div class="glass-card rounded-2xl p-4">
      <div class="text-xs text-slate-400">NDCG@10 (Contrastive Learning)</div>
      <div class="text-2xl font-extrabold text-amber-400 mt-1">${((cl.ndcg_at_10 || 0) * 100).toFixed(1)}%</div>
      <div class="text-[11px] text-emerald-400 mt-1">MRR: ${((cl.mrr || 0) * 100).toFixed(1)}% | Recall@10: ${((cl.recall_at_10 || 0) * 100).toFixed(1)}%</div>
    </div>
    <div class="glass-card rounded-2xl p-4">
      <div class="text-xs text-slate-400">Thỏa mãn Ràng buộc Giờ & Giá @5</div>
      <div class="text-2xl font-extrabold text-emerald-400 mt-1">${((cl.constraint_satisfaction_at_5 || 0) * 100).toFixed(1)}%</div>
      <div class="text-[11px] text-slate-400 mt-1">Nhờ Hard-Negative Mining</div>
    </div>
    <div class="glass-card rounded-2xl p-4">
      <div class="text-xs text-slate-400">Zero-Shot Cold-Start Hit@10</div>
      <div class="text-2xl font-extrabold text-cyan-400 mt-1">${((cl.cold_start_hit_at_10 || 0) * 100).toFixed(1)}%</div>
      <div class="text-[11px] text-slate-400 mt-1">Trên ${(ev.dataset_summary?.cold_start_poi_ids || []).length} điểm mới chưa từng huấn luyện</div>
    </div>
    <div class="glass-card rounded-2xl p-4">
      <div class="text-xs text-slate-400">Quy mô Dữ liệu & Kiến trúc</div>
      <div class="text-2xl font-extrabold text-violet-300 mt-1">${stats.total_pois || 0} POIs</div>
      <div class="text-[11px] text-slate-400 mt-1">${stats.night_economy_pois || 0} điểm Kinh tế đêm • ${((arch.trainable_parameters || 0) / 1000).toFixed(0)}K params</div>
    </div>
  `;

  Object.values(state.charts).forEach((ch) => ch.destroy());
  state.charts = {};

  Chart.defaults.color = "#cbd5e1";
  Chart.defaults.borderColor = "rgba(148,163,184,0.12)";

  const pop = bench.popularity_baseline || {};
  const tfidf = bench.tfidf_content_baseline || {};
  state.charts.benchmark = new Chart(document.getElementById("chart-benchmark"), {
    type: "bar",
    data: {
      labels: ["Recall@5", "Recall@10", "MRR", "NDCG@10", "Constraint@5", "Cold-Start Hit@10"],
      datasets: [
        {
          label: "Popularity Baseline",
          data: [pop.recall_at_5, pop.recall_at_10, pop.mrr, pop.ndcg_at_10, pop.constraint_satisfaction_at_5, pop.cold_start_hit_at_10],
          backgroundColor: "rgba(148, 163, 184, 0.55)",
          borderRadius: 5,
        },
        {
          label: "TF-IDF Content-Based",
          data: [tfidf.recall_at_5, tfidf.recall_at_10, tfidf.mrr, tfidf.ndcg_at_10, tfidf.constraint_satisfaction_at_5, tfidf.cold_start_hit_at_10],
          backgroundColor: "rgba(34, 211, 238, 0.7)",
          borderRadius: 5,
        },
        {
          label: "Two-Tower Contrastive Learning (Ours)",
          data: [cl.recall_at_5, cl.recall_at_10, cl.mrr, cl.ndcg_at_10, cl.constraint_satisfaction_at_5, cl.cold_start_hit_at_10],
          backgroundColor: "rgba(245, 158, 11, 0.92)",
          borderRadius: 5,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: { y: { beginAtZero: true, max: 1.0 } },
    },
  });

  const hist = ev.training_history || [];
  state.charts.training = new Chart(document.getElementById("chart-training"), {
    type: "line",
    data: {
      labels: hist.map((h) => `Ep ${h.epoch}`),
      datasets: [
        {
          label: "InfoNCE Train Loss",
          data: hist.map((h) => h.train_loss),
          borderColor: "#fb7185",
          backgroundColor: "rgba(251,113,133,0.15)",
          yAxisID: "yLoss",
          tension: 0.3,
          pointRadius: 1.5,
        },
        {
          label: "Val NDCG@10",
          data: hist.map((h) => h.val_ndcg_at_10),
          borderColor: "#34d399",
          backgroundColor: "rgba(52,211,153,0.15)",
          yAxisID: "yMetric",
          tension: 0.3,
          pointRadius: 1.5,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        yLoss: { type: "linear", position: "left", title: { display: true, text: "InfoNCE Loss" } },
        yMetric: { type: "linear", position: "right", min: 0, max: 1, grid: { drawOnChartArea: false }, title: { display: true, text: "Val NDCG@10" } },
      },
    },
  });

  const byCat = stats.by_category || {};
  const catKeys = Object.keys(byCat);
  state.charts.category = new Chart(document.getElementById("chart-category"), {
    type: "doughnut",
    data: {
      labels: catKeys.map((k) => (CAT_META[k] ? CAT_META[k].label : k)),
      datasets: [
        {
          data: catKeys.map((k) => byCat[k]),
          backgroundColor: catKeys.map((k) => (CAT_META[k] ? CAT_META[k].color : "#94a3b8")),
          borderWidth: 0,
        },
      ],
    },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "right" } } },
  });

  const byDist = stats.by_district || {};
  state.charts.district = new Chart(document.getElementById("chart-district"), {
    type: "bar",
    data: {
      labels: Object.keys(byDist),
      datasets: [
        {
          label: "Số điểm du lịch & kinh tế đêm",
          data: Object.values(byDist),
          backgroundColor: "rgba(167, 139, 250, 0.85)",
          borderRadius: 6,
        },
      ],
    },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } },
  });

  state.analyticsLoaded = true;
}

/* ================= EVENT LISTENERS & BOOTSTRAP ================= */
function applyPreset(key) {
  const p = PRESETS[key];
  if (!p) return;
  document.getElementById("input-query").value = p.query;
  document.getElementById("input-start").value = p.start;
  document.getElementById("input-end").value = p.end;
  document.getElementById("input-budget").value = p.budget;
  document.getElementById("budget-label").textContent = fmtVnd(p.budget);
  document.getElementById("input-stops").value = p.stops;
  document.getElementById("stops-label").textContent = `${p.stops} điểm`;
  document.getElementById("input-transport").value = p.transport;
  document.getElementById("input-weather").value = p.weather;
  document.getElementById("input-companion").value = p.companion;
  document.getElementById("input-district").value = p.district;
  document.getElementById("input-night-only").checked = p.nightOnly;

  document.querySelectorAll("#category-chips .chip-btn").forEach((btn) => {
    btn.classList.toggle("selected", p.cats.includes(btn.dataset.cat));
  });
  runOptimization();
}

async function checkHealth() {
  try {
    const resp = await fetch("/api/health");
    const h = await resp.json();
    document.getElementById("health-pill").innerHTML =
      `<span class="w-2 h-2 rounded-full bg-emerald-400"></span><span>AI Ready • ${h.total_pois} POIs • τ=${h.temperature}</span>`;
  } catch (_e) {
    document.getElementById("health-pill").innerHTML =
      `<span class="w-2 h-2 rounded-full bg-rose-400"></span><span>Mất kết nối API</span>`;
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  initMap();
  initVoiceRecognition();

  // Tabs
  document.querySelectorAll(".nav-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".nav-tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.add("hidden"));
      const target = tab.dataset.tab;
      document.getElementById(`tab-${target}`).classList.remove("hidden");
      if (target === "planner" && state.map) setTimeout(() => state.map.invalidateSize(), 150);
      if (target === "analytics") loadAnalytics();
    });
  });

  // Start Location controls: GPS, Address Search, Pick on Map
  document.getElementById("btn-gps-start").addEventListener("click", () => requestBrowserGeolocation(false));
  document.getElementById("btn-search-start").addEventListener("click", searchStartAddress);
  document.getElementById("start-address-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      searchStartAddress();
    }
  });
  document.getElementById("btn-pick-map-start").addEventListener("click", () => {
    state.pickingMapStart = true;
    document.getElementById("hanoi-map").classList.add("picking-location");
    document.getElementById("start-location-status").innerHTML =
      `<span class="text-amber-300 font-semibold">🎯 Hãy click trực tiếp vào một vị trí bất kỳ trên bản đồ bên phải để chọn điểm xuất phát!</span>`;
  });

  // Sliders
  document.getElementById("input-budget").addEventListener("input", (e) => {
    document.getElementById("budget-label").textContent = fmtVnd(e.target.value);
  });
  document.getElementById("input-stops").addEventListener("input", (e) => {
    document.getElementById("stops-label").textContent = `${e.target.value} điểm`;
  });

  // Category chips
  document.querySelectorAll("#category-chips .chip-btn").forEach((btn) => {
    btn.addEventListener("click", () => btn.classList.toggle("selected"));
  });

  // Presets
  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => applyPreset(btn.dataset.preset));
  });

  // Optimize button & map toggles
  document.getElementById("btn-optimize").addEventListener("click", runOptimization);
  document.getElementById("btn-toggle-map-style").addEventListener("click", () => setMapTiles(!state.darkTiles));
  document.getElementById("btn-toggle-baseline").addEventListener("click", () => {
    state.showBaselineRoute = !state.showBaselineRoute;
    if (state.lastOptimize) renderRoutesOnMap(state.lastOptimize);
  });

  // POI filters
  ["poi-search", "poi-filter-cat", "poi-filter-dist", "poi-filter-night"].forEach((id) => {
    document.getElementById(id).addEventListener("input", loadPois);
    document.getElementById(id).addEventListener("change", loadPois);
  });

  // Add POI Modal
  const modal = document.getElementById("modal-add-poi");
  const openModal = () => { modal.classList.remove("hidden"); modal.classList.add("flex"); };
  const closeModal = () => { modal.classList.add("hidden"); modal.classList.remove("flex"); };
  document.getElementById("btn-open-add-poi").addEventListener("click", openModal);
  document.getElementById("btn-close-modal").addEventListener("click", closeModal);
  document.getElementById("btn-cancel-modal").addEventListener("click", closeModal);

  document.getElementById("form-add-poi").addEventListener("submit", async (e) => {
    e.preventDefault();
    const fd = new FormData(e.target);
    const payload = {
      name: fd.get("name"),
      name_en: fd.get("name_en") || fd.get("name"),
      category: fd.get("category"),
      district: fd.get("district"),
      address: `${fd.get("district")}, Hà Nội`,
      lat: Number(fd.get("lat")),
      lon: Number(fd.get("lon")),
      open_hour: Number(fd.get("open_hour")),
      close_hour: Number(fd.get("close_hour")),
      avg_cost_vnd: Number(fd.get("avg_cost_vnd")),
      duration_min: Number(fd.get("duration_min")),
      description_vi: fd.get("description_vi"),
      description_en: fd.get("description_vi"),
      is_night_economy: fd.get("is_night_economy") === "on",
      indoor: fd.get("indoor") === "on",
      tags: [fd.get("category"), "sự kiện mới", "kinh tế đêm"],
    };
    const resp = await fetch("/api/pois", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (resp.ok) {
      closeModal();
      e.target.reset();
      state.analyticsLoaded = false;
      await loadPois();
      await checkHealth();
    } else {
      const err = await resp.json();
      alert(err.detail || "Không thể thêm điểm đến");
    }
  });

  // Chat
  document.getElementById("chat-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const input = document.getElementById("chat-input");
    const msg = input.value.trim();
    if (!msg) return;
    input.value = "";
    sendChatMessage(msg);
  });
  document.querySelectorAll(".chat-quick").forEach((btn) => {
    btn.addEventListener("click", () => sendChatMessage(btn.textContent.trim()));
  });

  // Initial load & prompt browser geolocation permission automatically
  await checkHealth();
  await loadPois();
  await runOptimization();
  requestBrowserGeolocation(true);
});
