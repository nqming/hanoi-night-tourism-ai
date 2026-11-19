"""End-to-end training & benchmark evaluation for Two-Tower Contrastive Learning.

Compares:
1. Popularity Baseline (rating-based ranking)
2. Content-Based Baseline (raw bilingual TF-IDF cosine similarity)
3. Two-Tower Contrastive Learning (PyTorch Dual-Encoder with InfoNCE + Hard Negatives)

Saves:
- ml/checkpoints/contrastive_model.pt
- ml/checkpoints/feature_extractor.pkl
- ml/checkpoints/poi_embeddings.npz
- ml/checkpoints/metrics.json

Run:
    python hanoi-night-tourism-ai/ml/train.py
"""

from __future__ import annotations

import datetime as dt
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.generate_training_pairs import time_overlap_hours
from ml.features import FeatureExtractor, context_to_text, poi_to_text
from ml.model import TwoTowerContrastiveNet, contrastive_infonce_loss

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
CHECKPOINT_DIR = ROOT_DIR / "ml" / "checkpoints"
POIS_PATH = PROCESSED_DIR / "hanoi_pois.json"
PAIRS_PATH = PROCESSED_DIR / "training_pairs.json"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _evaluate_rankings(
    score_matrix: np.ndarray,
    samples: List[Dict[str, Any]],
    pois: List[Dict[str, Any]],
    poi_id_to_idx: Dict[str, int],
) -> Dict[str, float]:
    """Compute Recall@5, Recall@10, MRR, NDCG@10, and Top-5 Constraint Satisfaction Rate."""
    rec5, rec10, mrrs, ndcgs, csr_list = [], [], [], [], []

    for i, sample in enumerate(samples):
        scores = score_matrix[i]
        ranked_indices = np.argsort(-scores)
        relevant = set(sample["relevant_poi_ids"])
        ctx = sample["context"]

        top5_ids = [pois[idx]["id"] for idx in ranked_indices[:5]]
        top10_ids = [pois[idx]["id"] for idx in ranked_indices[:10]]

        # Recall@K (fraction of relevant items captured in top-K)
        denom = max(1, min(len(relevant), 5))
        rec5.append(len(relevant.intersection(top5_ids)) / denom)
        rec10.append(len(relevant.intersection(top10_ids)) / max(1, len(relevant)))

        # MRR
        rr = 0.0
        for rank, idx in enumerate(ranked_indices[:25], start=1):
            if pois[idx]["id"] in relevant:
                rr = 1.0 / rank
                break
        mrrs.append(rr)

        # NDCG@10
        dcg = 0.0
        for rank, idx in enumerate(ranked_indices[:10], start=1):
            if pois[idx]["id"] in relevant:
                dcg += 1.0 / math.log2(rank + 1)
        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(10, len(relevant)) + 1))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

        # Constraint Satisfaction Rate @5 (open in requested window AND within budget)
        sat = 0
        budget_cap = max(ctx["budget_vnd"] * 0.85, 50000)
        for idx in ranked_indices[:5]:
            p = pois[idx]
            ov = time_overlap_hours(ctx["start_hour"], ctx["end_hour"], p["open_hour"], p["close_hour"])
            if ov >= 0.45 and p["avg_cost_vnd"] <= budget_cap:
                sat += 1
        csr_list.append(sat / 5.0)

    return {
        "recall_at_5": round(float(np.mean(rec5)), 4),
        "recall_at_10": round(float(np.mean(rec10)), 4),
        "mrr": round(float(np.mean(mrrs)), 4),
        "ndcg_at_10": round(float(np.mean(ndcgs)), 4),
        "constraint_satisfaction_at_5": round(float(np.mean(csr_list)), 4),
    }


def evaluate_cold_start(
    score_matrix: np.ndarray,
    test_samples: List[Dict[str, Any]],
    pois: List[Dict[str, Any]],
    cold_ids: set[str],
) -> Dict[str, float]:
    """Measure retrieval quality specifically on queries whose relevant set includes unseen Cold-Start POIs."""
    cold_hits_10 = []
    cold_ndcg = []
    for i, sample in enumerate(test_samples):
        cold_rel = set(sample["relevant_poi_ids"]).intersection(cold_ids)
        if not cold_rel:
            continue
        ranked = np.argsort(-score_matrix[i])
        top10 = {pois[idx]["id"] for idx in ranked[:10]}
        cold_hits_10.append(1.0 if cold_rel.intersection(top10) else 0.0)
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, idx in enumerate(ranked[:10], start=1)
            if pois[idx]["id"] in cold_rel
        )
        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, len(cold_rel) + 1))
        cold_ndcg.append(dcg / idcg if idcg > 0 else 0.0)

    if not cold_hits_10:
        return {"cold_start_queries": 0, "cold_start_hit_at_10": 0.0, "cold_start_ndcg_at_10": 0.0}

    return {
        "cold_start_queries": len(cold_hits_10),
        "cold_start_hit_at_10": round(float(np.mean(cold_hits_10)), 4),
        "cold_start_ndcg_at_10": round(float(np.mean(cold_ndcg)), 4),
    }


def train_model(
    epochs: int = 45,
    batch_size: int = 64,
    lr: float = 2.5e-3,
    hidden_dim: int = 192,
    embed_dim: int = 64,
    seed: int = 42,
) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Running Two-Tower Contrastive Learning on device={device}")

    with POIS_PATH.open("r", encoding="utf-8") as f:
        pois: List[Dict[str, Any]] = json.load(f)
    with PAIRS_PATH.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    samples = dataset["samples"]
    cold_ids = set(dataset["meta"]["cold_start_poi_ids"])
    poi_id_to_idx = {p["id"]: i for i, p in enumerate(pois)}

    train_samples = [s for s in samples if s["split"] == "train"]
    val_samples = [s for s in samples if s["split"] == "val"]
    test_samples = [s for s in samples if s["split"] == "test"]

    # Fit FeatureExtractor ONLY on non-cold POIs & train contexts (no data leakage!)
    train_pois = [p for p in pois if p["id"] not in cold_ids]
    train_contexts = [s["context"] for s in train_samples]
    fe = FeatureExtractor(text_dim=96).fit(train_pois, train_contexts)

    # Encode all POIs and all split contexts
    all_poi_feats = fe.transform_pois(pois)  # [N_pois, D_poi]
    train_ctx_feats = fe.transform_contexts([s["context"] for s in train_samples])
    val_ctx_feats = fe.transform_contexts([s["context"] for s in val_samples])
    test_ctx_feats = fe.transform_contexts([s["context"] for s in test_samples])

    train_pos_idx = np.array([poi_id_to_idx[s["positive_poi_id"]] for s in train_samples], dtype=np.int64)
    train_neg_idx = np.array([poi_id_to_idx[s["hard_negative_poi_id"]] for s in train_samples], dtype=np.int64)

    train_pos_feats = all_poi_feats[train_pos_idx]
    train_neg_feats = all_poi_feats[train_neg_idx]

    train_ds = TensorDataset(
        torch.from_numpy(train_ctx_feats),
        torch.from_numpy(train_pos_feats),
        torch.from_numpy(train_neg_feats),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = TwoTowerContrastiveNet(
        context_dim=fe.context_dim,
        poi_dim=fe.poi_dim,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        dropout=0.15,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    all_poi_tensor = torch.from_numpy(all_poi_feats).to(device)
    val_ctx_tensor = torch.from_numpy(val_ctx_feats).to(device)

    history: List[Dict[str, float]] = []
    best_val_ndcg = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for bx_ctx, bx_pos, bx_neg in train_loader:
            bx_ctx = bx_ctx.to(device)
            bx_pos = bx_pos.to(device)
            bx_neg = bx_neg.to(device)

            optimizer.zero_grad()
            u, v_pos, v_neg, scale = model(bx_ctx, bx_pos, bx_neg)
            loss = contrastive_infonce_loss(u, v_pos, v_neg, scale)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(1, n_batches)

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            u_val = model.encode_context(val_ctx_tensor).cpu().numpy()
            v_all = model.encode_poi(all_poi_tensor).cpu().numpy()
            val_scores = u_val @ v_all.T
        val_metrics = _evaluate_rankings(val_scores, val_samples, pois, poi_id_to_idx)

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(avg_loss, 4),
                "val_ndcg_at_10": val_metrics["ndcg_at_10"],
                "val_recall_at_5": val_metrics["recall_at_5"],
                "temperature": round(model.temperature, 4),
            }
        )

        if val_metrics["ndcg_at_10"] > best_val_ndcg:
            best_val_ndcg = val_metrics["ndcg_at_10"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:02d}/{epochs} | Loss: {avg_loss:.4f} | "
                f"Val NDCG@10: {val_metrics['ndcg_at_10']:.4f} | "
                f"Val Recall@5: {val_metrics['recall_at_5']:.4f} | tau: {model.temperature:.3f}"
            )

    # Restore best validation weights
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()

    # === Final Test Evaluation (including Cold-Start POIs!) ===
    with torch.no_grad():
        u_test = model.encode_context(torch.from_numpy(test_ctx_feats).to(device)).cpu().numpy()
        v_all = model.encode_poi(all_poi_tensor).cpu().numpy()
        cl_test_scores = u_test @ v_all.T

    cl_metrics = _evaluate_rankings(cl_test_scores, test_samples, pois, poi_id_to_idx)
    cl_cold = evaluate_cold_start(cl_test_scores, test_samples, pois, cold_ids)

    # Baseline 1: Popularity (rating + log review_count)
    pop_scores_1d = np.array(
        [p["rating"] + 0.05 * math.log1p(p.get("review_count", 100)) for p in pois],
        dtype=np.float32,
    )
    pop_matrix = np.tile(pop_scores_1d, (len(test_samples), 1))
    pop_metrics = _evaluate_rankings(pop_matrix, test_samples, pois, poi_id_to_idx)
    pop_cold = evaluate_cold_start(pop_matrix, test_samples, pois, cold_ids)

    # Baseline 2: Raw TF-IDF Content-Based Cosine Similarity
    poi_tfidf = fe.encode_tfidf_raw([poi_to_text(p) for p in pois])
    ctx_tfidf = fe.encode_tfidf_raw([context_to_text(s["context"]) for s in test_samples])
    tfidf_matrix = cosine_similarity(ctx_tfidf, poi_tfidf)
    tfidf_metrics = _evaluate_rankings(tfidf_matrix, test_samples, pois, poi_id_to_idx)
    tfidf_cold = evaluate_cold_start(tfidf_matrix, test_samples, pois, cold_ids)

    # Save checkpoints
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "context_dim": fe.context_dim,
            "poi_dim": fe.poi_dim,
            "hidden_dim": hidden_dim,
            "embed_dim": embed_dim,
        },
        CHECKPOINT_DIR / "contrastive_model.pt",
    )
    fe.save(CHECKPOINT_DIR / "feature_extractor.pkl")
    np.savez_compressed(
        CHECKPOINT_DIR / "poi_embeddings.npz",
        poi_ids=np.array([p["id"] for p in pois]),
        embeddings=v_all.astype(np.float32),
    )

    report = {
        "trained_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "architecture": {
            "name": "Two-Tower Contrastive Learning (Dual-Encoder InfoNCE + Hard Negatives)",
            "context_input_dim": fe.context_dim,
            "poi_input_dim": fe.poi_dim,
            "hidden_dim": hidden_dim,
            "embed_dim": embed_dim,
            "learned_temperature": round(model.temperature, 4),
            "epochs": epochs,
            "batch_size": batch_size,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        },
        "dataset_summary": dataset["meta"],
        "benchmark_comparison": {
            "popularity_baseline": {**pop_metrics, **pop_cold},
            "tfidf_content_baseline": {**tfidf_metrics, **tfidf_cold},
            "two_tower_contrastive_learning": {**cl_metrics, **cl_cold},
        },
        "training_history": history,
    }
    with (CHECKPOINT_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== Benchmark Comparison (Test Set + Cold-Start POIs) ===")
    print(json.dumps(report["benchmark_comparison"], ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    train_model()
