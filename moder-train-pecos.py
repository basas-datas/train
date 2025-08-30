#!/usr/bin/env python3
import os
import re
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
TRAIN_PATH = "train.jsonl"                  # путь к обучающей выборке
VAL_PATH   = "val.jsonl"                    # путь к валидационной выборке
OUT_DIR    = "xlinear_model_topics"         # папка для сохранения модели
MODEL_NAME = "intfloat/multilingual-e5-base"   # 🔄 используем e5-base эмбеддер

TRAIN_EMB_PATH = "train_emb_e5.npy"         # файл для кэша эмбеддингов train
VAL_EMB_PATH   = "val_emb_e5.npy"           # файл для кэша эмбеддингов val

# веса для разных частей текста
W_TITLE = 3
W_LINK  = 2
W_DESC  = 1

# метрики будем считать для этих k
TOP_KS = [1, 3, 5, 10]
# ==============================================

# === Настройка логов ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

def log(msg: str):
    """Удобный принт с таймштампом"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# === Очистка текста ===

def clean_desc(text: str) -> str:
    """Очищает описание от ссылок, username, email и длинных чисел"""
    if not text:
        return ""

    # Любые ссылки (http/https + всё до пробела или конца строки)
    text = re.sub(r"https?://\S+|www\.\S+", "<URL>", text)

    # @username
    text = re.sub(r"@\w+", "<USER>", text)

    # Email
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "<EMAIL>", text)

    # длинные числа (например телефоны, id, счета)
    text = re.sub(r"\b\d{5,}\b", "<NUM>", text)

    return text.strip()

def clean_link(link: str) -> str:
    """Проверяет ссылку: если она приватная (рандомная абракадабра) → заменяет"""
    if not link:
        return ""

    # Маска приватной ссылки: только буквы/цифры/-/_ , длина >= 10
    if re.fullmatch(r"[A-Za-z0-9\-_]{10,}", link):
        return "<PRIV_LINK>"

    return link.strip()

# === Вспомогательные функции ===

def read_jsonl(path):
    """Читает JSONL файл построчно"""
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
    """
    Собирает список (title, link, desc) и список меток для каждой записи.
    Метки берутся из main_topic, main_topic_old, possible_topics.
    """
    triples, labels = [], []
    for r in records:
        title = r.get("title") or ""
        link  = clean_link(r.get("short_link") or "")
        desc  = clean_desc(r.get("orig_description") or "")

        labs = []
        if r.get("main_topic"): labs.append(r["main_topic"])
        if r.get("main_topic_old"): labs.append(r["main_topic_old"])
        if r.get("possible_topics"): labs.extend(r["possible_topics"])
        labs = [l.lower().strip() for l in labs if l]

        if (title or link or desc) and labs:
            triples.append((title, link, desc))
            labels.append(labs)
    return triples, labels

def embed_weighted(embedder, triples, batch_size=1024):
    """
    Считает эмбеддинги с весами: title*W_TITLE + link*W_LINK + desc*W_DESC.
    E5-base не требует специальных префиксов.
    """
    all_vecs = []
    total = len(triples)
    for i in range(0, total, batch_size):
        log(f"Эмбеддинги: {i}/{total}")
        batch = triples[i:i+batch_size]
        titles = [t[0] for t in batch]
        links  = [t[1] for t in batch]
        descs  = [t[2] for t in batch]

        e_title = embedder.encode(titles, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=False)
        e_link  = embedder.encode(links,  convert_to_numpy=True, batch_size=batch_size, show_progress_bar=False)
        e_desc  = embedder.encode(descs,  convert_to_numpy=True, batch_size=batch_size, show_progress_bar=False)

        # линейная комбинация с весами
        emb = (W_TITLE*e_title + W_LINK*e_link + W_DESC*e_desc) / (W_TITLE+W_LINK+W_DESC)
        all_vecs.append(emb)
    return np.vstack(all_vecs)

def build_label_matrix(labels, classes):
    """
    Строим бинарную матрицу Y (samples x labels).
    Если у примера несколько меток — ставим 1 (или вес) для каждой.
    """
    label_to_idx = {l: i for i, l in enumerate(classes)}
    rows, cols, data = [], [], []
    for row_id, labs in enumerate(labels):
        counts = Counter(labs)  # если есть дубликаты меток
        for lab, freq in counts.items():
            if lab in label_to_idx:
                rows.append(row_id)
                cols.append(label_to_idx[lab])
                data.append(float(freq))
    return csr_matrix((data, (rows, cols)), shape=(len(labels), len(classes)), dtype=np.float32)

def precision_recall_at_k(y_true, y_score, k=5):
    """
    Считает Precision@k и Recall@k.
    y_true — матрица [samples x labels], бинарная.
    y_score — предсказанные вероятности.
    """
    precisions, recalls = [], []
    for yt, yp in zip(y_true, y_score):
        topk = np.argsort(-yp)[:k]
        hits = yt[topk].sum()
        precisions.append(hits / k)
        recalls.append(hits / yt.sum() if yt.sum() > 0 else 0)
    return np.mean(precisions), np.mean(recalls)

def dcg_at_k(y_true_row, y_score_row, k=10):
    """Discounted Cumulative Gain для одной строки"""
    topk = np.argsort(-y_score_row)[:k]
    gains = y_true_row[topk]
    discounts = 1.0 / np.log2(np.arange(2, k+2))
    return np.sum(gains * discounts)

def ndcg_at_k(y_true, y_score, k=10):
    """Normalized DCG"""
    ndcgs = []
    for yt, yp in zip(y_true, y_score):
        dcg = dcg_at_k(yt, yp, k)
        ideal = dcg_at_k(yt, yt, min(k, int(yt.sum())))
        if ideal > 0:
            ndcgs.append(dcg / ideal)
    return np.mean(ndcgs) if ndcgs else 0.0

def evaluate(y_true, y_score):
    """Считает и выводит Precision/Recall/nDCG для каждого K"""
    log("=== Метрики по Top-k ===")
    for k in TOP_KS:
        p, r = precision_recall_at_k(y_true, y_score, k)
        n = ndcg_at_k(y_true, y_score, k)
        log(f"Top-{k}: Precision@{k}={p:.4f}, Recall@{k}={r:.4f}, nDCG@{k}={n:.4f}")

def train():
    """Основной пайплайн: загрузка данных, эмбеддинг, обучение, валидация"""
    log("Загрузка датасета...")
    tr = read_jsonl(TRAIN_PATH)
    va = read_jsonl(VAL_PATH)
    Xtr_triples, Ytr_labels = build_dataset(tr)
    Xva_triples, Yva_labels = build_dataset(va)

    # кодируем метки в бинарную матрицу
    mlb = MultiLabelBinarizer()
    mlb.fit(Ytr_labels + Yva_labels)
    classes = mlb.classes_

    Ytr = build_label_matrix(Ytr_labels, classes)
    Yva = build_label_matrix(Yva_labels, classes)

    log(f"Количество уникальных лейблов: {len(classes)}")

    # эмбеддер
    embedder = SentenceTransformer(MODEL_NAME)

    # кэшируем эмбеддинги
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

    # обучение PECOS
    log("⚡ Начало обучения XLinear (PECOS)...")
    start = time.time()

    train_params = {
        "threads": os.cpu_count(),       # все CPU
        "max_leaf_size": 100,            # размер листа
        "negative_sampling": "tfn",      # улучшает топ-K метрики
        "verbosity": 2                   # уровень логов
    }

    model = XLinearModel.train(
        csr_matrix(Xtr.astype(np.float32)),
        Ytr,
        train_params=train_params
    )

    elapsed = time.time() - start
    log(f"✅ Обучение завершено за {elapsed:.2f} сек.")

    # предсказания на валидации
    log("Предсказания на валидации...")
    pred_csr = model.predict(csr_matrix(Xva.astype(np.float32)), only_topk=100)
    y_score = pred_csr.toarray()

    # метрики
    evaluate(Yva.toarray(), y_score)

    # сохраняем всё
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save(OUT_DIR)
    joblib.dump(mlb, os.path.join(OUT_DIR, "mlb.joblib"))
    joblib.dump(embedder, os.path.join(OUT_DIR, "embedder.joblib"))

    log(f"💾 Модель сохранена в {OUT_DIR}")


if __name__ == "__main__":
    train()
