#!/usr/bin/env python3
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers
from pecos.xmc.xlinear.model import XLinearModel
from scipy.sparse import csr_matrix
import numpy as np
import joblib

# ==== ПАРАМЕТРЫ ====
INDEX_NAME = "list_index"
EMBEDDING_FIELD = "embedding_e5_base"           # уже готовые вектора
UPDATED_FIELD = "topic10_updated_at"            # поле для времени обновления
TOPIC_FIELD = "topic10"                         # поле для сохранения меток
MAX_DOCS = 200
MAX_RUNTIME = 55  # секунд
BATCH_SIZE = 20
TOP_K = 10
MODEL_DIR = "xlinear_model_topics"
# ====================

load_dotenv()

# Elasticsearch клиент
es = Elasticsearch(
    hosts=["http://localhost:9200"],
    basic_auth=(os.getenv("ELASTIC_USER"), os.getenv("ELASTIC_PASS"))
)

print("Загрузка модели...")
model = XLinearModel.load(MODEL_DIR)
mlb = joblib.load(os.path.join(MODEL_DIR, "mlb.joblib"))


def fetch_docs(limit):
    """Берём документы, у которых есть эмбеддинг, но нет меток"""
    query = {
        "bool": {
            "must": {"exists": {"field": EMBEDDING_FIELD}},
            "must_not": {"exists": {"field": TOPIC_FIELD}}
        }
    }
    res = es.search(
        index=INDEX_NAME,
        size=limit,
        query=query,
        sort=[{"subscribers": {"order": "desc"}}],
        _source=[EMBEDDING_FIELD]
    )
    return res["hits"]["hits"]


def predict_topics(batch_vectors):
    """Запускаем XLinear и получаем топ-10 меток"""
    X = csr_matrix(np.array(batch_vectors, dtype=np.float32))
    pred_csr = model.predict(X, only_topk=TOP_K)
    y_score = pred_csr.toarray()

    all_preds = []
    for scores in y_score:
        top_idx = np.argsort(-scores)[:TOP_K]
        labels = [mlb.classes_[j] for j in top_idx]
        all_preds.append(labels)
    return all_preds


def update_bulk(batch_docs, batch_labels):
    """Отправляем bulk update в Elasticsearch"""
    now = datetime.utcnow().isoformat()
    actions = []
    for doc, labels in zip(batch_docs, batch_labels):
        actions.append({
            "_op_type": "update",
            "_index": INDEX_NAME,
            "_id": doc["_id"],
            "doc": {
                TOPIC_FIELD: labels,
                UPDATED_FIELD: now
            }
        })
    helpers.bulk(es, actions)


def main():
    docs = fetch_docs(MAX_DOCS)
    if not docs:
        print("✅ Все документы уже содержат топики.")
        return

    start = time.time()
    total = 0

    for i in range(0, len(docs), BATCH_SIZE):
        elapsed = time.time() - start
        if elapsed > MAX_RUNTIME:
            print(f"⏱ Время вышло ({elapsed:.1f} сек). Завершаем.")
            break

        batch_docs = docs[i:i + BATCH_SIZE]
        vectors = [d["_source"][EMBEDDING_FIELD] for d in batch_docs]

        try:
            pred_labels = predict_topics(vectors)
            update_bulk(batch_docs, pred_labels)
            total += len(batch_docs)
            print(f"[{elapsed:.1f} сек] ✅ Обновлено {len(batch_docs)} документов (всего {total})")
        except Exception as e:
            print(f"[{elapsed:.1f} сек] ⚠️ Ошибка в батче {i//BATCH_SIZE}: {e}")

    print(f"🔁 Всего обновлено документов: {total}")


if __name__ == "__main__":
    main()
