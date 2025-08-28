#!/usr/bin/env python3
import os
import json
import random
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from pecos.xmc.xlinear.model import XLinearModel
from scipy.sparse import csr_matrix

# ========= ПАРАМЕТРЫ =========
VAL_PATH = "val.jsonl"
MODEL_DIR = "xlinear_model_topics"
SAMPLES = 100
TOP_K = 5
LANG_FILTERS = {"en_Latn"}  # языки, которые берём
OUT_FILE = "pred_samples.txt"
# =============================

def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def embed_weighted(embedder, triples, W_TITLE=4, W_LINK=2, W_DESC=1):
    titles = [t[0] for t in triples]
    links  = [t[1] for t in triples]
    descs  = [t[2] for t in triples]

    e_title = embedder.encode(titles, convert_to_numpy=True, show_progress_bar=False)
    e_link  = embedder.encode(links, convert_to_numpy=True, show_progress_bar=False)
    e_desc  = embedder.encode(descs, convert_to_numpy=True, show_progress_bar=False)

    emb = (W_TITLE*e_title + W_LINK*e_link + W_DESC*e_desc) / (W_TITLE+W_LINK+W_DESC)
    return emb

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

        triples.append((title, link, desc))
        labels.append(labs)
    return triples, labels

def main():
    print("Загрузка данных...")
    val_data = read_jsonl(VAL_PATH)

    # фильтруем по языку
    val_filtered = [r for r in val_data if r.get("lang2") in LANG_FILTERS]
    print(f"Всего найдено {len(val_filtered)} примеров с языками {LANG_FILTERS}")

    if len(val_filtered) > SAMPLES:
        val_filtered = random.sample(val_filtered, SAMPLES)

    triples, labels = build_dataset(val_filtered)

    print("Загрузка модели и энкодера...")
    model = XLinearModel.load(MODEL_DIR)
    mlb = joblib.load(os.path.join(MODEL_DIR, "mlb.joblib"))
    embedder = joblib.load(os.path.join(MODEL_DIR, "embedder.joblib"))

    print("Вычисление эмбеддингов...")
    Xva = embed_weighted(embedder, triples)

    print("Предсказания...")
    pred_csr = model.predict(csr_matrix(Xva.astype(np.float32)), only_topk=TOP_K)
    y_score = pred_csr.toarray()

    with open(OUT_FILE, "w", encoding="utf-8") as fout:
        for i, (sample, true_labels) in enumerate(zip(triples, labels)):
            scores = y_score[i]
            top_idx = np.argsort(-scores)[:TOP_K]
            pred_labels = [(mlb.classes_[j], scores[j]) for j in top_idx]

            block = []
            block.append("="*80)
            block.append(f"Пример {i+1}")
            block.append(f"Тайтл: {sample[0]}")
            block.append(f"Линк: {sample[1]}")
            block.append(f"Описание: {sample[2]}")
            block.append(f"Истинные метки: {true_labels}")
            block.append("Предсказанные метки:")
            for lab, sc in pred_labels:
                block.append(f"  {lab:<20s} {sc:.4f}")

            text_block = "\n".join(block)
            print(text_block)
            fout.write(text_block + "\n")

    print(f"\n✅ Результаты сохранены в {OUT_FILE}")

if __name__ == "__main__":
    main()
