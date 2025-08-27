#!/usr/bin/env python3
import os
import json
import time
import joblib
import numpy as np
from tqdm import tqdm
from collections import Counter
from sentence_transformers import SentenceTransformer
from pecos.xmc.xlinear.model import XLinearModel
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix
import logging

# ================= ПАРАМЕТРЫ ==================
TRAIN_PATH = "train.jsonl"
VAL_PATH   = "val.jsonl"
OUT_DIR    = "xlinear_model_topics"
MODEL_NAME = "distiluse-base-multilingual-cased-v1"

TRAIN_EMB_PATH = "train_emb.npy"
VAL_EMB_PATH   = "val_emb.npy"

W_TITLE = 4
W_LINK  = 2
W_DESC  = 1

TOP_KS = [1, 3, 5, 10]
# ==============================================

# === Настройка логов ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                log(f"⚠️ Ошибка JSON в строке {i}: {e}")
    return data

def build_dataset(records):
    triples, labels = [], []
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

def embed_weighted(embedder, triples, batch_size=400):
    all_vecs = []
    total = len(triples)
    for i in range(0, total, batch_size):
        log(f"Эмбеддинги: {i}/{total}")
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

# === Построение матрицы Y с весами ===
def build_label_matrix(labels, classes):
    label_to_idx = {l: i for i, l in enumerate(classes)}
    rows, cols, data = [], [], []
    for row_id, labs in enumerate(labels):
        counts = Counter(labs)  # считаем дубликаты
        for lab, freq in counts.items():
            if lab in label_to_idx:
                rows.append(row_id)
                cols.append(label_to_idx[lab])
                data.append(float(freq))  # вес = число повторов (float!)
    return csr_matrix((data, (rows, cols)), shape=(len(labels), len(classes)), dtype=np.float32)

def precision_recall_at_k(y_true, y_score, k=5):
    precisions, recalls = [], []
    for yt, yp in zip(y_true, y_score):
        topk = np.argsort(-yp)[:k]
        hits = yt[topk].sum()
        precisions.append(hits / k)
        recalls.append(hits / yt.sum() if yt.sum() > 0 else 0)
    return np.mean(precisions), np.mean(recalls)

def dcg_at_k(y_true_row, y_score_row, k=10):
    topk = np.argsort(-y_score_row)[:k]
    gains = y_true_row[topk]
    discounts = 1.0 / np.log2(np.arange(2, k+2))
    return np.sum(gains * discounts)

def ndcg_at_k(y_true, y_score, k=10):
    ndcgs = []
    for yt, yp in zip(y_true, y_score):
        dcg = dcg_at_k(yt, yp, k)
        ideal = dcg_at_k(yt, yt, min(k, int(yt.sum())))
        if ideal > 0:
            ndcgs.append(dcg / ideal)
    return np.mean(ndcgs) if ndcgs else 0.0

def evaluate(y_true, y_score):
    log("=== Метрики по Top-k ===")
    for k in TOP_KS:
        p, r = precision_recall_at_k(y_true, y_score, k)
        n = ndcg_at_k(y_true, y_score, k)
        log(f"Top-{k}: Precision@{k}={p:.4f}, Recall@{k}={r:.4f}, nDCG@{k}={n:.4f}")

def train():
    log("Загрузка датасета...")
    tr = read_jsonl(TRAIN_PATH)
    va = read_jsonl(VAL_PATH)
    Xtr_triples, Ytr_labels = build_dataset(tr)
    Xva_triples, Yva_labels = build_dataset(va)

    mlb = MultiLabelBinarizer()
    mlb.fit(Ytr_labels + Yva_labels)
    classes = mlb.classes_

    Ytr = build_label_matrix(Ytr_labels, classes)
    Yva = build_label_matrix(Yva_labels, classes)

    log(f"Количество уникальных лейблов: {len(classes)}")

    embedder = SentenceTransformer(MODEL_NAME)

    if os.path.exists(TRAIN_EMB_PATH):
        Xtr = np.load(TRAIN_EMB_PATH)
    else:
        Xtr = embed_weighted(embedder, Xtr_triples)
        np.save(TRAIN_EMB_PATH, Xtr)

    if os.path.exists(VAL_EMB_PATH):
        Xva = np.load(VAL_EMB_PATH)
    else:
        Xva = embed_weighted(embedder, Xva_triples)
        np.save(VAL_EMB_PATH, Xva)

    log("⚡ Начало обучения XLinear (PECOS)...")
    start = time.time()

    # Включаем параметр verbosity для логов
    train_params = {
        "threads": os.cpu_count(),  # использовать все CPU
        "max_leaf_size": 100,
        "verbosity": 2              # уровень логирования (0=тихо, 1=основное, 2=подробно)
    }

    model = XLinearModel.train(
        csr_matrix(Xtr.astype(np.float32)),
        Ytr,
        train_params=train_params
    )

    elapsed = time.time() - start
    log(f"✅ Обучение завершено за {elapsed:.2f} сек.")

    log("Предсказания на валидации...")
    pred_csr = model.predict(csr_matrix(Xva.astype(np.float32)), only_topk=100)
    y_score = pred_csr.toarray()

    evaluate(Yva.toarray(), y_score)

    os.makedirs(OUT_DIR, exist_ok=True)
    model.save(OUT_DIR)
    joblib.dump(mlb, os.path.join(OUT_DIR, "mlb.joblib"))
    joblib.dump(embedder, os.path.join(OUT_DIR, "embedder.joblib"))

    log(f"💾 Модель сохранена в {OUT_DIR}")


if __name__ == "__main__":
    train()
