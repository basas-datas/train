#!/usr/bin/env python3
import argparse
import json
import os
import random
import joblib
import numpy as np
from scipy.sparse import hstack

LABEL_ORDER = [
    "is_illegal_drugs","is_weapons","is_escort_prostitution","is_gambling","is_betting",
    "is_fraud_scam","is_lgbt","is_adult","is_sexual_content","is_illegal","is_vpn",
    "is_dating","is_extremism","is_political","is_war_military","is_piracy","is_cybercrime"
]

def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def dict_labels_to_array(d, order):
    return np.array([1 if d[k] else 0 for k in order], dtype=np.int32)

def main(val_path, model_dir, n_samples=100):
    # Загружаем val
    data = read_jsonl(val_path)
    if len(data) > n_samples:
        data = random.sample(data, n_samples)

    # Загружаем модель
    vec_w = joblib.load(os.path.join(model_dir, "tfidf_word.joblib"))
    vec_c = joblib.load(os.path.join(model_dir, "tfidf_char.joblib"))
    clf   = joblib.load(os.path.join(model_dir, "ovr_clf.joblib"))
    with open(os.path.join(model_dir, "thresholds.json"), "r", encoding="utf-8") as f:
        thresholds = json.load(f)["thresholds"]

    # Готовим данные
    texts = [r["text"] for r in data]
    Ytrue = np.vstack([dict_labels_to_array(r["labels"], LABEL_ORDER) for r in data])

    Xw = vec_w.transform(texts)
    Xc = vec_c.transform(texts)
    X  = hstack([Xw,Xc], format="csr")

    # Предсказания
    proba = clf.predict_proba(X)
    Ypred = np.zeros_like(Ytrue)
    for j,t in enumerate(thresholds):
        Ypred[:,j] = (proba[:,j] >= t).astype(int)

    # Выводим
    for i, r in enumerate(data):
        print("="*80)
        print(f"TEXT:\n{r['text']}\n")
        true_labels = [LABEL_ORDER[j] for j in range(len(LABEL_ORDER)) if Ytrue[i,j]==1]
        pred_labels = [LABEL_ORDER[j] for j in range(len(LABEL_ORDER)) if Ypred[i,j]==1]
        print(f"TRUE : {true_labels}")
        print(f"PRED : {pred_labels}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=str, required=True, help="val.jsonl")
    ap.add_argument("--model_dir", type=str, required=True, help="папка с моделью")
    ap.add_argument("--n", type=int, default=100, help="кол-во примеров")
    args = ap.parse_args()
    main(args.val, args.model_dir, args.n)
