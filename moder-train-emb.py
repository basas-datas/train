#!/usr/bin/env python3
import os
import json
import time
import joblib
import numpy as np
from tqdm import tqdm
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_recall_fscore_support

# ================= ПАРАМЕТРЫ ==================
TRAIN_PATH = "train.jsonl"
VAL_PATH   = "val.jsonl"
OUT_DIR    = "embed_model_topics"
MODEL_NAME = "distiluse-base-multilingual-cased-v1"

# где будут храниться вектора
TRAIN_EMB_PATH = "train_emb.npy"
VAL_EMB_PATH   = "val_emb.npy"

W_TITLE = 3
W_LINK  = 2
W_DESC  = 1

TOP_KS = [1, 3, 5, 10]
# ==============================================

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def build_dataset(records):
    texts, labels, triples = [], [], []
    for r in records:
        title = r.get("title") or ""
        link  = r.get("short_link") or ""
        desc  = r.get("orig_description") or ""

        labs = []
        if r.get("main_topic"): labs.append(r["main_topic"])
        if r.get("main_topic_old"): labs.append(r["main_topic_old"])
        if r.get("possible_topics"): labs.extend(r["possible_topics"])
        labs = [l.lower().strip() for l in labs if l]

        if (title or link or desc) and labs:
            triples.append((title, link, desc))
            labels.append(labs)
    return triples, labels


def embed_weighted(embedder, triples, batch_size=140):
    all_vecs = []
    total = len(triples)
    num_batches = (total + batch_size - 1) // batch_size
    for i in range(0, total, batch_size):
        batch_id = i // batch_size + 1
        log(f"Эмбеддинги: батч {batch_id}/{num_batches} (обработано {i}/{total}, осталось ~{total - i})")

        batch = triples[i:i+batch_size]
        titles = [t[0] for t in batch]
        links  = [t[1] for t in batch]
        descs  = [t[2] for t in batch]

        e_title = embedder.encode(titles, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=False)
        e_link  = embedder.encode(links, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=False)
        e_desc  = embedder.encode(descs, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=False)

        emb = (W_TITLE*e_title + W_LINK*e_link + W_DESC*e_desc) / (W_TITLE+W_LINK+W_DESC)
        all_vecs.append(emb)
    return np.vstack(all_vecs)


def build_classifier():
    base = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        penalty="l2",
        max_iter=80,
        tol=1e-3,
        n_jobs=-1,
        verbose=1,
        class_weight="balanced"
    )
    return OneVsRestClassifier(base, n_jobs=-1)


def fit_thresholds(y_true, y_proba):
    thresholds = []
    for j in tqdm(range(y_true.shape[1]), desc="Подбор порогов", unit="label"):
        yt, yp = y_true[:, j], y_proba[:, j]
        best_t, best_f1 = 0.5, -1
        for t in np.linspace(0.05, 0.95, 19):
            f1 = f1_score(yt, (yp >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_t, best_f1 = t, f1
        thresholds.append(best_t)
    return thresholds


# ---------- Метрики Top-k ----------
def precision_recall_at_k(y_true, y_proba, k=5):
    precisions, recalls = [], []
    for yt, yp in zip(y_true, y_proba):
        topk = np.argsort(-yp)[:k]
        hits = yt[topk].sum()
        precisions.append(hits / k)
        recalls.append(hits / yt.sum() if yt.sum() > 0 else 0)
    return np.mean(precisions), np.mean(recalls)


def dcg_at_k(y_true_row, y_proba_row, k=10):
    topk = np.argsort(-y_proba_row)[:k]
    gains = y_true_row[topk]
    discounts = 1.0 / np.log2(np.arange(2, k+2))
    return np.sum(gains * discounts)


def ndcg_at_k(y_true, y_proba, k=10):
    ndcgs = []
    for yt, yp in zip(y_true, y_proba):
        dcg = dcg_at_k(yt, yp, k)
        ideal = dcg_at_k(yt, yt, min(k, int(yt.sum())))
        if ideal == 0:
            continue
        ndcgs.append(dcg / ideal)
    return np.mean(ndcgs) if ndcgs else 0.0


def evaluate(y_true, y_proba, thresholds, mlb):
    log("=== Метрики с порогами ===")
    y_pred = np.zeros_like(y_true)
    for j, t in enumerate(thresholds):
        y_pred[:, j] = (y_proba[:, j] >= t).astype(int)

    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    log(f"Macro-F1={macro:.4f} Micro-F1={micro:.4f}")

    counts = y_true.sum(axis=0)
    top_idx = np.argsort(-counts)[:20]
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    for j in top_idx:
        log(f"{mlb.classes_[j]:<20s}: prec={prec[j]:.3f}, rec={rec[j]:.3f}, f1={f1[j]:.3f}, thr={thresholds[j]:.2f}, sup={counts[j]}")

    log("=== Метрики по Top-k ===")
    for k in TOP_KS:
        p, r = precision_recall_at_k(y_true, y_proba, k)
        n = ndcg_at_k(y_true, y_proba, k)
        log(f"Top-{k}: Precision@{k}={p:.4f}, Recall@{k}={r:.4f}, nDCG@{k}={n:.4f}")


def train():
    log("Загрузка датасета...")
    tr = read_jsonl(TRAIN_PATH)
    va = read_jsonl(VAL_PATH)
    Xtr_triples, Ytr_labels = build_dataset(tr)
    Xva_triples, Yva_labels = build_dataset(va)

    mlb = MultiLabelBinarizer()
    Ytr = mlb.fit_transform(Ytr_labels)
    Yva = mlb.transform(Yva_labels)

    log(f"Количество уникальных лейблов: {len(mlb.classes_)}")

    log(f"Загрузка эмбеддинговой модели: {MODEL_NAME}")
    embedder = SentenceTransformer(MODEL_NAME)

    # --- эмбеддинги train ---
    if os.path.exists(TRAIN_EMB_PATH):
        log(f"Загрузка сохранённых эмбеддингов train из {TRAIN_EMB_PATH}")
        Xtr = np.load(TRAIN_EMB_PATH)
    else:
        log("Вычисление эмбеддингов train...")
        Xtr = embed_weighted(embedder, Xtr_triples)
        np.save(TRAIN_EMB_PATH, Xtr)
        log(f"Сохранено в {TRAIN_EMB_PATH}")

    # --- эмбеддинги val ---
    if os.path.exists(VAL_EMB_PATH):
        log(f"Загрузка сохранённых эмбеддингов val из {VAL_EMB_PATH}")
        Xva = np.load(VAL_EMB_PATH)
    else:
        log("Вычисление эмбеддингов val...")
        Xva = embed_weighted(embedder, Xva_triples)
        np.save(VAL_EMB_PATH, Xva)
        log(f"Сохранено в {VAL_EMB_PATH}")

    log("Обучение классификатора...")
    clf = build_classifier()
    clf.fit(Xtr, Ytr)

    log("Предсказания на валидации...")
    proba = clf.predict_proba(Xva)
    thresholds = fit_thresholds(Yva, proba)
    evaluate(Yva, proba, thresholds, mlb)

    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump(embedder, os.path.join(OUT_DIR, "embedder.joblib"))
    joblib.dump(clf, os.path.join(OUT_DIR, "ovr_clf.joblib"))
    joblib.dump(mlb, os.path.join(OUT_DIR, "mlb.joblib"))
    with open(os.path.join(OUT_DIR, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump({"thresholds": thresholds}, f, ensure_ascii=False, indent=2)

    log(f"Модель сохранена в {OUT_DIR}")


if __name__ == "__main__":
    train()
