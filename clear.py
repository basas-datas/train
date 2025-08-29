#!/usr/bin/env python3
import json
import os
import re
from collections import Counter

# входные и выходные файлы
FILES = [
    ("train0.jsonl", "train.jsonl"),
    ("val0.jsonl", "val.jsonl"),
]

THRESHOLD = 150
STATS_RAW_FILE   = "labels_stats_raw.json"    # ДО замен/удалений
STATS_NORM_FILE  = "labels_stats_norm.json"   # ПОСЛЕ замен/удалений (dry-run)
FINAL_LABELS_FILE = "final_labels.txt"
REPLACEMENT_FILE = "label_replacement.txt"

# метки для удаления (всегда в нижнем регистре!)
DELETION = {
    "promotion","advertising","social","updates","bot","links",
    "russia","russian","ukrainian","ukrain","ukrainians", "watch","watches","chatting","commerce","group_chat", "ban","chat", "ban", "channel"
}

def load_replacements(path):
    """Парсим строки вида: "main": "alias"  (каждая пара на отдельной строке, запятая опциональна)"""
    if not os.path.exists(path):
        return {}, (lambda s: (s or "").lower().strip())

    canon_map = {}
    line_re = re.compile(r'"\s*([^"]+)\s*"\s*:\s*"\s*([^"]+)\s*"')

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip().rstrip(",")
            if not line:
                continue
            m = line_re.match(line)
            if not m:
                continue
            main, alias = m.group(1), m.group(2)
            main = (main or "").lower().strip()
            alias = (alias or "").lower().strip()
            if main and alias:
                # alias -> main
                canon_map[alias] = main

    def canon_fn(label: str) -> str:
        x = (label or "").lower().strip()
        seen = set()
        while x in canon_map and x not in seen:
            seen.add(x)
            x = canon_map[x]
        return x

    # «сплющим» возможные цепочки alias -> alias -> main
    for v in list(canon_map.keys()):
        canon_map[v] = canon_fn(v)

    return canon_map, canon_fn

def extract_labels_raw(item):
    """Сбор сырых меток из записи (как есть), БЕЗ изменений"""
    labels = []
    if item.get("main_topic"):       labels.append(item["main_topic"])
    if item.get("main_topic_old"):   labels.append(item["main_topic_old"])
    if isinstance(item.get("possible_topics"), list):
        labels.extend([x for x in item["possible_topics"] if x])
    return labels

def normalize_label(lbl, canon_fn):
    """Нормализация одного лейбла: lower, trim, replacement, deletion"""
    if not lbl:
        return ""
    x = canon_fn(lbl)  # приведение к канонической форме (с учётом файла замен)
    if x in DELETION:
        return ""
    return x

def normalize_item(item, canon_fn):
    """Применяем нормализацию к полям записи"""
    if item.get("main_topic"):
        item["main_topic"] = normalize_label(item["main_topic"], canon_fn)
    if item.get("main_topic_old"):
        item["main_topic_old"] = normalize_label(item["main_topic_old"], canon_fn)
    if isinstance(item.get("possible_topics"), list):
        norm_poss = []
        for x in item["possible_topics"]:
            nx = normalize_label(x, canon_fn)
            if nx:
                norm_poss.append(nx)
        item["possible_topics"] = norm_poss
    return item

def extract_labels_norm(item):
    """Сбор меток из уже нормализованной записи"""
    labels = []
    if item.get("main_topic"):       labels.append(item["main_topic"])
    if item.get("main_topic_old"):   labels.append(item["main_topic_old"])
    if isinstance(item.get("possible_topics"), list):
        labels.extend([x for x in item["possible_topics"] if x])
    # уберём дубликаты внутри одной записи
    return sorted(set([x for x in labels if x]))

def count_raw(files):
    """Подсчёт частот по сырым данным (до нормализации) — для отладки/сравнения"""
    counter = Counter()
    total_rows = 0
    for in_path, _ in files:
        with open(in_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                total_rows += 1
                item = json.loads(line)
                labels = extract_labels_raw(item)
                # приводим к нижнему регистру тут, чтобы не плодить дублей по регистру
                labels = [(lbl or "").lower().strip() for lbl in labels if lbl]
                counter.update(labels)
    return counter, total_rows

def count_norm_dry(files, canon_fn):
    """Dry-run нормализация + подсчет частот после замен/удалений (без записи файлов)"""
    counter = Counter()
    total_rows = 0
    rows_with_any_label = 0
    for in_path, _ in files:
        with open(in_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                total_rows += 1
                item = json.loads(line)
                item = normalize_item(item, canon_fn)
                labels = extract_labels_norm(item)
                if labels:
                    rows_with_any_label += 1
                    counter.update(labels)
    return counter, total_rows, rows_with_any_label

def write_filtered(files, allowed_set, canon_fn):
    """Запись финальных файлов с нормализацией и фильтром по allowed_set"""
    stats = []
    for in_path, out_path in files:
        total, kept, removed_rows = 0, 0, 0
        with open(in_path, "r", encoding="utf-8") as fin, \
             open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                total += 1
                item = json.loads(line)
                item = normalize_item(item, canon_fn)

                # фильтрация по allowed_set
                mt  = item.get("main_topic")
                mto = item.get("main_topic_old")
                poss = item.get("possible_topics", [])

                if mt not in allowed_set:
                    item["main_topic"] = ""
                if mto not in allowed_set:
                    item["main_topic_old"] = ""
                item["possible_topics"] = [p for p in poss if p in allowed_set]

                labels_after = extract_labels_norm(item)
                if not labels_after:
                    removed_rows += 1
                    continue

                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                kept += 1

        print(f"{in_path} → {out_path}: всего={total}, сохранено={kept}, удалено={removed_rows}")
        stats.append((in_path, total, kept, removed_rows))
    return stats

def recount_final(files_out):
    """Пересчёт частот на уже записанных нормализованных файлах"""
    counter = Counter()
    uniq_rows = 0
    for path_out in files_out:
        with open(path_out, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                uniq_rows += 1
                item = json.loads(line)
                labels = extract_labels_norm(item)
                counter.update(labels)
    return counter, uniq_rows

def main():
    canon_map, canon_fn = load_replacements(REPLACEMENT_FILE)
    print(f"Загружено правил замен: {len(canon_map)}, удаляемых меток: {len(DELETION)}")

    # 1) Сырые частоты (для понимания, откуда стартуем)
    raw_counter, raw_rows = count_raw(FILES)
    with open(STATS_RAW_FILE, "w", encoding="utf-8") as f:
        json.dump(raw_counter.most_common(), f, ensure_ascii=False, indent=2)
    print(f"[RAW] Строк: {raw_rows}, уникальных меток: {len(raw_counter)} (см. {STATS_RAW_FILE})")

    # 2) Dry-run нормализация и частоты после замен/удалений
    norm_counter, norm_rows, norm_rows_with_labels = count_norm_dry(FILES, canon_fn)
    with open(STATS_NORM_FILE, "w", encoding="utf-8") as f:
        json.dump(norm_counter.most_common(), f, ensure_ascii=False, indent=2)
    print(f"[NORM] Строк: {norm_rows}, со строками с метками: {norm_rows_with_labels}, уникальных меток: {len(norm_counter)} (см. {STATS_NORM_FILE})")

    # 3) Allowed по порогу — ВАЖНО: по нормализованным частотам
    allowed = {lbl for lbl, freq in norm_counter.items() if freq >= THRESHOLD}
    print(f"[ALLOWED] Порог={THRESHOLD}, разрешённых меток: {len(allowed)}")

    # 4) Пишем финальные файлы
    write_filtered(FILES, allowed, canon_fn)

    # 5) Пересчитываем частоты по финальным файлам
    final_paths = [FILES[0][1], FILES[1][1]]
    final_counter, final_rows = recount_final(final_paths)
    print(f"[FINAL] Строк в финальных файлах: {final_rows}, уникальных меток: {len(final_counter)}")

    # сортировка меток по алфавиту
    final_labels = sorted(final_counter.keys())

    # формируем список "метка + частота"
    final_labels_with_freq = [f"{lbl}" for lbl in final_labels]

    with open(FINAL_LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(final_labels_with_freq, f, ensure_ascii=False, indent=2)
    print(f"[FINAL] Итоговые метки с частотами записаны в {FINAL_LABELS_FILE} (всего {len(final_labels)})")

    # 6) Для удобства — первые 50 меток и top-20 по частоте
    print("\n=== Пример (первые 50 финальных меток) ===")
    for lbl in final_labels_with_freq[:10]:
        print(lbl)

    print("\n=== Топ-20 меток по частоте в финальных файлах ===")
    for lbl, cnt in final_counter.most_common(150):
        print(f"{lbl:<30} {cnt}")


if __name__ == "__main__":
    main()
