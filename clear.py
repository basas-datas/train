#!/usr/bin/env python3
import json
from collections import Counter

# входные и выходные файлы
FILES = [
    ("train0.jsonl", "train.jsonl"),
    ("val0.jsonl", "val.jsonl"),
]

THRESHOLD = 300
STATS_FILE = "labels_stats.json"

def extract_labels(item):
    labels = []
    if item.get("main_topic"):
        labels.append(item["main_topic"])
    if item.get("main_topic_old"):
        labels.append(item["main_topic_old"])
    if isinstance(item.get("possible_topics"), list):
        labels.extend(item["possible_topics"])
    return labels

def count_labels(files):
    counter = Counter()
    for in_path, _ in files:
        with open(in_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                labels = extract_labels(item)
                counter.update(labels)
    return counter

def filter_labels(files, allowed_labels):
    for in_path, out_path in files:
        total, kept, removed_rows = 0, 0, 0
        with open(in_path, "r", encoding="utf-8") as fin, \
             open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                total += 1
                item = json.loads(line)

                # фильтруем метки
                main = item.get("main_topic")
                old = item.get("main_topic_old")
                poss = item.get("possible_topics", [])

                if main not in allowed_labels:
                    item["main_topic"] = ""
                if old not in allowed_labels:
                    item["main_topic_old"] = ""
                item["possible_topics"] = [p for p in poss if p in allowed_labels]

                # проверяем, осталась ли хоть одна метка
                labels_after = extract_labels(item)
                if not labels_after:
                    removed_rows += 1
                    continue

                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                kept += 1

        print(f"{in_path} → {out_path}: всего={total}, сохранено={kept}, удалено={removed_rows}")

def main():
    # шаг 1: собираем статистику
    counter = count_labels(FILES)
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(counter.most_common(), f, ensure_ascii=False, indent=2)
    print(f"Статистика сохранена в {STATS_FILE}, всего уникальных меток: {len(counter)}")

    # шаг 2: фильтруем по порогу
    allowed = {lbl for lbl, freq in counter.items() if freq >= THRESHOLD}
    print(f"Разрешённых меток (частота ≥{THRESHOLD}): {len(allowed)}")

    # шаг 3: фильтрация файлов
    filter_labels(FILES, allowed)

if __name__ == "__main__":
    main()
