# 🌙 Hanoi Night-Tourism AI Platform

> **Nhiệm vụ KH&CN – Bài toán số 15:** Phát triển du lịch thông minh và kinh tế đêm tại Hà Nội  
> **Sinh viên thực hiện:** Nguyễn Quang Minh • Lớp `65TTNT` • MSV `2351260672`

Hệ thống nền tảng Web du lịch thông minh và quản lý kinh tế đêm tại Hà Nội tích hợp **Two-Tower Contrastive Learning (PyTorch InfoNCE + Hard-Negative Mining)**, **Thuật toán tối ưu hóa lộ trình có ràng buộc khung thời gian (TOPTW)**, **Bản đồ số tương tác (Leaflet + OpenStreetMap + OSRM)** và **Trợ lý ảo RAG đa ngôn ngữ (Việt / Anh)**.

---

## ✨ Tính năng chính (Đáp ứng đủ 4 Sản phẩm nghiên cứu)

1. **Nền tảng Web Du lịch Thông minh & Quản trị Dữ liệu Kinh tế Đêm (Sản phẩm 1):**
   - Bộ dữ liệu **160 địa điểm thực tế tại 6 quận trung tâm Hà Nội** được crawl tự động từ **OpenStreetMap Overpass API**, **Wikipedia REST API** và chuẩn hóa khung giờ kinh tế đêm (bao gồm khung giờ qua nửa đêm `17:00 – 02:00`, `22:30 – 05:00`).
   - Hỗ trợ tìm kiếm, lọc theo quận/danh mục/kinh tế đêm và **Admin CRUD** (thêm/sửa/xóa sự kiện đêm mới và mã hóa ngay vào không gian vector AI — *Zero-Shot Cold Start*).
2. **Mô hình Two-Tower Contrastive Learning & Trợ lý ảo RAG (Sản phẩm 2):**
   - Kiến trúc Dual-Encoder bằng **PyTorch** gồm **User-Context Tower** và **POI Tower** chiếu vào cùng không gian vector L2-normalized $\mathbb{R}^{64}$.
   - Huấn luyện bằng hàm mất mát **Symmetric InfoNCE (CLIP-style)** kết hợp **Hard-Negative Margin Mining** (ép mô hình phân biệt các địa điểm đóng cửa ban đêm, vượt ngân sách hoặc ngoài trời khi trời mưa).
   - Trợ lý ảo song ngữ Việt/Anh truy xuất trực tiếp từ không gian vector Contrastive Learning (hoạt động miễn phí 100% offline, có thể gắn thêm `GEMINI_API_KEY` nếu muốn).
3. **Bản đồ số & Thuật toán Tối ưu Lộ trình Ngày – Đêm TOPTW (Sản phẩm 3):**
   - Kết hợp điểm tương đồng từ Contrastive Learning với thuật toán **Beam Search + 2-Opt Local Search** giải bài toán *Team Orienteering Problem with Time Windows (TOPTW)*.
   - Tự động vẽ tuyến đường phố thực tế trên bản đồ **OpenStreetMap / OSRM**, tuân thủ giờ mở cửa ban đêm, ngân sách và tối ưu đa dạng trải nghiệm.
4. **Thực nghiệm & Đánh giá Định lượng (Sản phẩm 4):**
   - Dashboard trực quan so sánh **Two-Tower Contrastive Learning vs. TF-IDF Content-Based vs. Popularity Baseline** trên tập Test (gồm 24 điểm Cold-Start chưa từng xuất hiện trong tập Train).
   - So sánh trực tiếp **Trước & Sau Tối ưu** (Baseline thủ công vs. AI + TOPTW) về quãng đường, thời gian di chuyển, độ phủ kinh tế đêm và độ khớp sở thích.

---

## 📊 Kết quả Thực nghiệm Mô hình Contrastive Learning

| Mô hình | Recall@5 | Recall@10 | MRR | NDCG@10 | Thỏa mãn Giờ & Giá @5 | Cold-Start Hit@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Popularity Baseline | 0.1308 | 0.2217 | 0.3055 | 0.1924 | 62.9% | 2.3% |
| TF-IDF Content-Based | 0.2342 | 0.3867 | 0.4509 | 0.3310 | 77.8% | 55.9% |
| **Two-Tower Contrastive Learning (Ours)** | **0.6133** | **0.8458** | **0.8972** | **0.7888** | **92.5%** | **74.6%** |

---

## 🏗️ Cấu trúc dự án

```text
hanoi-night-tourism-ai/
├── .github/workflows/ci-cd.yml     # GitHub Actions CI/CD (Lint -> Test -> Docker Build -> Smoke Test -> Push GHCR)
├── Dockerfile                      # Production container (Python 3.12 + CPU PyTorch + Non-root + Healthcheck)
├── docker-compose.yml              # Khởi chạy 1 lệnh kèm lưu trữ dữ liệu bền vững
├── data/
│   ├── curated_hanoi_night.py      # 40 địa điểm kinh tế đêm biểu tượng Hà Nội
│   ├── crawl_hanoi_pois.py         # Script crawl tự động OpenStreetMap + Wikipedia
│   ├── generate_training_pairs.py  # Sinh cặp dữ liệu Positive & Hard-Negative Contrastive
│   └── processed/                  # hanoi_pois.json, hanoi_pois.csv, training_pairs.json
├── ml/
│   ├── features.py                 # Trích xuất đặc trưng song ngữ + thời gian tuần hoàn (sin/cos)
│   ├── model.py                    # PyTorch Two-Tower Contrastive Net + InfoNCE Loss
│   ├── train.py                    # Huấn luyện, đánh giá Benchmark & xuất checkpoint
│   ├── inference.py                # Engine truy xuất vector (hỗ trợ Zero-Shot Cold-Start POI mới)
│   ├── optimizer.py                # Bộ giải TOPTW (Beam Search + 2-Opt) & so sánh Baseline
│   ├── rag_assistant.py            # Trợ lý ảo RAG song ngữ VI/EN
│   └── checkpoints/                # contrastive_model.pt, feature_extractor.pkl, metrics.json
├── backend/
│   ├── main.py                     # FastAPI REST API + phục vụ Web UI & Swagger /docs
│   ├── schemas.py                  # Pydantic v2 schemas
│   └── storage.py                  # Quản lý dữ liệu điểm đến hỗ trợ CRUD
├── frontend/                       # Giao diện Web (Tailwind + Leaflet.js + Chart.js)
└── tests/test_suite.py             # Bộ kiểm thử tự động (Unit + Integration Tests)
```

---

## 🚀 Hướng dẫn Chạy Local

```bash
# 1. (Tùy chọn) Chạy lại Crawler lấy dữ liệu mới từ OpenStreetMap & Wikipedia
python data/crawl_hanoi_pois.py

# 2. (Tùy chọn) Sinh lại dữ liệu và huấn luyện lại mô hình Contrastive Learning
python data/generate_training_pairs.py
python ml/train.py

# 3. Chạy bộ kiểm thử tự động
python -m unittest discover -s tests -v

# 4. Khởi chạy Web App & API Server
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Mở trình duyệt truy cập:
- **Giao diện ứng dụng:** `http://localhost:8000`
- **Tài liệu API (Swagger UI):** `http://localhost:8000/docs`

---

## 🐳 Build & Chạy bằng Docker

```bash
# Build Docker Image
docker build -t hanoi-night-tourism-ai:latest .

# Chạy container
docker run -d -p 8000:8000 --name hanoi-ai hanoi-night-tourism-ai:latest

# Hoặc dùng Docker Compose
docker compose up -d --build
```

---

## ⚙️ Triển khai CI/CD với GitHub Actions

File pipeline đã được cấu hình sẵn tại `.github/workflows/ci-cd.yml`. Khi bạn đẩy mã nguồn lên GitHub (`git push origin main`), GitHub Actions sẽ tự động:
1. **Lint & Test:** Kiểm tra mã nguồn bằng `ruff`, chạy toàn bộ bộ kiểm thử `pytest` và kiểm tra ngưỡng chất lượng mô hình (`NDCG@10 > 0.65`).
2. **Docker Build & Smoke Test:** Build Docker image thực tế, khởi chạy container trên runner CI và gọi kiểm tra `/api/health`, `/api/recommend`, `/api/optimize-itinerary`.
3. **Publish Docker Image:** Tự động push image có gắn phiên bản (`latest`, `sha-...`) lên **GitHub Container Registry (`ghcr.io/<username>/<repo>:latest`)** bằng `GITHUB_TOKEN` mặc định.
