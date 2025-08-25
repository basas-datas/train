#!/usr/bin/env python3
import argparse
import os
import json
import time
import joblib
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_recall_fscore_support

LABEL_ORDER = [
    "is_illegal_drugs","is_weapons","is_escort_prostitution","is_gambling","is_betting",
    "is_fraud_scam","is_lgbt","is_adult","is_sexual_content","is_illegal","is_vpn",
    "is_dating","is_extremism","is_political","is_war_military","is_piracy","is_cybercrime"
]

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def read_jsonl(path):
    log(f"Загрузка {path} ...")
    data = []
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    log(f"Загружено {len(data)} строк из {path}")
    return data

def dict_labels_to_array(d, order):
    return np.array([1 if d[k] else 0 for k in order], dtype=np.int32)

def to_Xy(records):
    texts = [r["text"] for r in records]
    Y = np.vstack([dict_labels_to_array(r["labels"], LABEL_ORDER) for r in records])
    return texts, Y

def build_classifier():
    base = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        penalty="l2",
        max_iter=50,
        tol=1e-3,
        n_jobs=-1,
        verbose=1,
        class_weight="balanced"
    )
    return OneVsRestClassifier(base, n_jobs=-1)

def fit_thresholds(y_true, y_proba):
    thresholds = []
    log("Подбор порогов по F1 для каждого класса...")
    for j in tqdm(range(y_true.shape[1]), desc="Порог по классам", unit="class"):
        yt, yp = y_true[:, j], y_proba[:, j]
        best_t, best_f1 = 0.5, -1
        for t in np.linspace(0.05,0.95,19):
            f1 = f1_score(yt, (yp>=t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_t, best_f1 = t, f1
        thresholds.append(best_t)
    return thresholds

def evaluate(y_true, y_proba, thresholds):
    log("Оценка качества модели...")
    y_pred = np.zeros_like(y_true)
    for j,t in enumerate(thresholds):
        y_pred[:,j] = (y_proba[:,j]>=t).astype(int)

    macro = f1_score(y_true,y_pred,average="macro",zero_division=0)
    micro = f1_score(y_true,y_pred,average="micro",zero_division=0)
    log(f"Macro-F1={macro:.4f} Micro-F1={micro:.4f}")

    prec,rec,f1,_ = precision_recall_fscore_support(y_true,y_pred,average=None,zero_division=0)
    for j,name in enumerate(LABEL_ORDER):
        log(f"{name:>20s}: prec={prec[j]:.3f}, rec={rec[j]:.3f}, f1={f1[j]:.3f}, thr={thresholds[j]:.2f}, sup={y_true[:,j].sum()}")

def train(train_path, val_path, out_dir, model_name="distiluse-base-multilingual-cased-v1"):
    log("Загрузка датасета...")
    tr = read_jsonl(train_path)
    va = read_jsonl(val_path)
    Xtr_texts,Ytr = to_Xy(tr)
    Xva_texts,Yva = to_Xy(va)

    log(f"Загрузка модели эмбеддингов: {model_name}")
    embedder = SentenceTransformer(model_name)

    log("Вычисление эмбеддингов train...")
    Xtr = embedder.encode(Xtr_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    log("Вычисление эмбеддингов val...")
    Xva = embedder.encode(Xva_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)

    log("Обучение классификатора...")
    clf = build_classifier()
    clf.fit(Xtr, Ytr)

    log("Предсказания на валидации...")
    proba = clf.predict_proba(Xva)
    thresholds = fit_thresholds(Yva, proba)
    evaluate(Yva, proba, thresholds)

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(embedder, os.path.join(out_dir,"embedder.joblib"))
    joblib.dump(clf, os.path.join(out_dir,"ovr_clf.joblib"))
    with open(os.path.join(out_dir,"thresholds.json"),"w",encoding="utf-8") as f:
        json.dump({"thresholds":thresholds},f,ensure_ascii=False,indent=2)

    log(f"Модель сохранена в {out_dir}")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, required=True)
    ap.add_argument("--val", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="embed_model")
    ap.add_argument("--embedder", type=str, default="distiluse-base-multilingual-cased-v1", help="название модели эмбеддингов")
    args = ap.parse_args()
    train(args.train, args.val, args.out_dir, args.embedder)
