import argparse
import json
import os
import sys
import time
from typing import List, Dict, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_recall_fscore_support
from scipy.sparse import hstack
import joblib
from tqdm import tqdm   # прогресс-бары

LABEL_ORDER = [
    "is_illegal_drugs","is_weapons","is_escort_prostitution","is_gambling","is_betting",
    "is_fraud_scam","is_lgbt","is_adult","is_sexual_content","is_illegal","is_vpn",
    "is_dating","is_extremism","is_political","is_war_military","is_piracy","is_cybercrime"
]

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def read_jsonl(path: str) -> List[Dict]:
    log(f"Загрузка {path}")
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data.append(json.loads(line))
    log(f"Загружено {len(data)} записей")
    return data

def dict_labels_to_array(d: Dict[str,bool], order: List[str]) -> np.ndarray:
    return np.array([1 if d[k] else 0 for k in order], dtype=np.int32)

def to_Xy(records: List[Dict], order: List[str]) -> Tuple[List[str], np.ndarray]:
    texts = [r["text"] for r in records]
    Y = np.vstack([dict_labels_to_array(r["labels"], order) for r in records])
    return texts, Y

def build_vectorizers():
    vec_word = TfidfVectorizer(analyzer="word", ngram_range=(1,2), max_features=500_000,
                               dtype=np.float32, sublinear_tf=True)
    vec_char = TfidfVectorizer(analyzer="char", ngram_range=(3,5), max_features=500_000,
                               dtype=np.float32, sublinear_tf=True)
    return vec_word, vec_char

def build_classifier():
    base = LogisticRegression(solver="saga", max_iter=200, C=1.0, n_jobs=-1, class_weight="balanced")
    return OneVsRestClassifier(base, n_jobs=-1)

def fit_thresholds(y_true, y_proba):
    thresholds = []
    log("Подбор порогов по F1...")
    for j in tqdm(range(y_true.shape[1])):
        yt, yp = y_true[:, j], y_proba[:, j]
        best_t, best_f1 = 0.5, -1
        for t in np.linspace(0.05,0.95,19):
            f1 = f1_score(yt, (yp>=t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_t, best_f1 = t, f1
        thresholds.append(best_t)
    return thresholds

def evaluate(y_true, y_proba, thresholds):
    y_pred = np.zeros_like(y_true)
    for j,t in enumerate(thresholds):
        y_pred[:,j] = (y_proba[:,j]>=t).astype(int)
    macro = f1_score(y_true,y_pred,average="macro",zero_division=0)
    micro = f1_score(y_true,y_pred,average="micro",zero_division=0)
    log(f"Macro-F1={macro:.4f} Micro-F1={micro:.4f}")
    prec,rec,f1,_ = precision_recall_fscore_support(y_true,y_pred,average=None,zero_division=0)
    for j,name in enumerate(LABEL_ORDER):
        log(f"{name:>20s}: f1={f1[j]:.3f}, thr={thresholds[j]:.2f}, sup={y_true[:,j].sum()}")

def train(train_path, val_path, out_dir):
    # load
    tr = read_jsonl(train_path); va = read_jsonl(val_path)
    Xtr_texts,Ytr = to_Xy(tr,LABEL_ORDER)
    Xva_texts,Yva = to_Xy(va,LABEL_ORDER)

    # vectorize
    log("Обучение векторизаторов...")
    vec_w, vec_c = build_vectorizers()
    Xtr_w = vec_w.fit_transform(Xtr_texts); Xtr_c = vec_c.fit_transform(Xtr_texts)
    Xtr = hstack([Xtr_w,Xtr_c],format="csr")
    Xva_w = vec_w.transform(Xva_texts); Xva_c = vec_c.transform(Xva_texts)
    Xva = hstack([Xva_w,Xva_c],format="csr")
    log(f"Размер матриц: train={Xtr.shape}, val={Xva.shape}")

    # clf
    log("Обучение классификатора...")
    clf = build_classifier()
    clf.fit(Xtr,Ytr)
    log("Модель обучена")

    # eval
    log("Предсказания на валидации...")
    proba = clf.predict_proba(Xva)
    thresholds = fit_thresholds(Yva,proba)
    evaluate(Yva,proba,thresholds)

    # save
    os.makedirs(out_dir,exist_ok=True)
    joblib.dump(vec_w, os.path.join(out_dir,"tfidf_word.joblib"))
    joblib.dump(vec_c, os.path.join(out_dir,"tfidf_char.joblib"))
    joblib.dump(clf,   os.path.join(out_dir,"ovr_clf.joblib"))
    with open(os.path.join(out_dir,"thresholds.json"),"w",encoding="utf-8") as f:
        json.dump({"thresholds":thresholds},f,ensure_ascii=False,indent=2)
    log(f"Сохранено в {out_dir}")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str)
    ap.add_argument("--val", type=str)
    ap.add_argument("--out_dir", type=str, default="tfidf_model")
    args = ap.parse_args()
    if args.train and args.val:
        train(args.train,args.val,args.out_dir)
    else:
        log("Укажите --train и --val")
