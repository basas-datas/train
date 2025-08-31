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
    "russia","russian","ukrainian","ukrain","ukrainians",
    "watch","watches","chatting","commerce","group_chat",
    "ban","chat","channel", "communities"
}

# === Очистка текста ===

def clean_desc(text: str) -> str:
    """Очищает описание от любых URL, username, email и длинных чисел.
       Телефоны с + заменяются на +<NUM>."""
    if not text:
        return ""

    # Любые URL (http/https/ftp/… + www)
    text = re.sub(r"\b(?:[a-z][a-z0-9+\-.]*://\S+|www\.\S+)\b", "<URL>", text)

    # @username
    text = re.sub(r"@\w+", "<USER>", text)

    # Email
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "<EMAIL>", text)

    # длинные числа (например телефоны, id, счета)
    # сначала телефоны с + впереди
    text = re.sub(r"\+\d{5,}", "+<NUM>", text)
    # остальные длинные числа
    text = re.sub(r"\b\d{5,}\b", "<NUM>", text)

    return text.strip()


def clean_link(link: str):
    """
    Проверяет ссылку и заменяет на токен <PRIV_LINK>, если она приватная.
    Возвращает (новая_ссылка, old_link) — чтобы можно было проверить подмены.
    
    Приватная ссылка определяется по правилам:
    1. Ровно 16 символов (латиница/цифры).
    2. 16 символов, затем "_" или "-" или "–", затем 5–6 символов.
    3. Если строка содержит "-", "_" , "+", "–" — то сразу приватная.
    """
    if not link:
        return "", None

    link = link.strip()

    # Правило 3: наличие спецсимволов → приватная
    if any(ch in link for ch in "-+–"):
        return "<PRIV_LINK>", link

    # Правило 1: ровно 16 символов
    if re.fullmatch(r"[A-Za-z0-9]{16}", link):
        return "<PRIV_LINK>", link

    # Правило 2: 16 + "_" или "-" или "–" + 5–6 символов
    if re.fullmatch(r"[A-Za-z0-9]{16}[_\-–][A-Za-z0-9]{5,6}", link):
        return "<PRIV_LINK>", link

    return link, None


# === Работа с метками ===

def load_replacements(path):
    """Парсим строки вида: "main": "alias"  (каждая пара на отдельной строке)"""
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
                canon_map[alias] = main

    def canon_fn(label: str) -> str:
        x = (label or "").lower().strip()
        seen = set()
        while x in canon_map and x not in seen:
            seen.add(x)
            x = canon_map[x]
        return x

    for v in list(canon_map.keys()):
        canon_map[v] = canon_fn(v)

    return canon_map, canon_fn

def extract_labels_raw(item):
    labels = []
    if item.get("main_topic"):       labels.append(item["main_topic"])
    if item.get("main_topic_old"):   labels.append(item["main_topic_old"])
    if isinstance(item.get("possible_topics"), list):
        labels.extend([x for x in item["possible_topics"] if x])
    return labels

def normalize_label(lbl, canon_fn):
    if not lbl:
        return ""
    x = canon_fn(lbl)
    if x in DELETION:
        return ""
    return x

def normalize_item(item, canon_fn):
    """Применяем нормализацию к меткам + очистку текстовых полей"""
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

    # 🔄 очистка текстов
    if item.get("orig_description"):
        item["orig_description"] = clean_desc(item["orig_description"])

    if item.get("short_link"):
        new_link, old_link = clean_link(item["short_link"])
        item["short_link"] = new_link
        if old_link:
            item["old_link"] = old_link  # сохраняем оригинал, если была замена

    # title не трогаем

    return item

def extract_labels_norm(item):
    labels = []
    if item.get("main_topic"):       labels.append(item["main_topic"])
    if item.get("main_topic_old"):   labels.append(item["main_topic_old"])
    if isinstance(item.get("possible_topics"), list):
        labels.extend([x for x in item["possible_topics"] if x])
    return sorted(set([x for x in labels if x]))

# === Подсчёты и фильтрация ===

def count_raw(files):
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
                labels = [(lbl or "").lower().strip() for lbl in labels if lbl]
                counter.update(labels)
    return counter, total_rows

def count_norm_dry(files, canon_fn):
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

# === main ===

def main():
    canon_map, canon_fn = load_replacements(REPLACEMENT_FILE)
    print(f"Загружено правил замен: {len(canon_map)}, удаляемых меток: {len(DELETION)}")

    raw_counter, raw_rows = count_raw(FILES)
    with open(STATS_RAW_FILE, "w", encoding="utf-8") as f:
        json.dump(raw_counter.most_common(), f, ensure_ascii=False, indent=2)
    print(f"[RAW] Строк: {raw_rows}, уникальных меток: {len(raw_counter)} (см. {STATS_RAW_FILE})")

    norm_counter, norm_rows, norm_rows_with_labels = count_norm_dry(FILES, canon_fn)
    with open(STATS_NORM_FILE, "w", encoding="utf-8") as f:
        json.dump(norm_counter.most_common(), f, ensure_ascii=False, indent=2)
    print(f"[NORM] Строк: {norm_rows}, со строками с метками: {norm_rows_with_labels}, уникальных меток: {len(norm_counter)} (см. {STATS_NORM_FILE})")

    allowed = {lbl for lbl, freq in norm_counter.items() if freq >= THRESHOLD}
    print(f"[ALLOWED] Порог={THRESHOLD}, разрешённых меток: {len(allowed)}")

    write_filtered(FILES, allowed, canon_fn)

    final_paths = [FILES[0][1], FILES[1][1]]
    final_counter, final_rows = recount_final(final_paths)
    print(f"[FINAL] Строк в финальных файлах: {final_rows}, уникальных меток: {len(final_counter)}")

    final_labels = sorted(final_counter.keys())
    final_labels_with_freq = [f"{lbl}" for lbl in final_labels]

    with open(FINAL_LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(final_labels_with_freq, f, ensure_ascii=False, indent=2)
    print(f"[FINAL] Итоговые метки с частотами записаны в {FINAL_LABELS_FILE} (всего {len(final_labels)})")

    print("\n=== Пример (первые 50 финальных меток) ===")
    for lbl in final_labels_with_freq[:10]:
        print(lbl)

    print("\n=== Топ-20 меток по частоте в финальных файлах ===")
    for lbl, cnt in final_counter.most_common(20):
        print(f"{lbl:<30} {cnt}")


if __name__ == "__main__":
    main()
