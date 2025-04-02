import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from datasets import load_dataset
from time import time

# Путь к уже сохранённой модели
model_path = "./mt5-large-ru-test2"

# Загружаем модель и токенизатор
model = MT5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = MT5Tokenizer.from_pretrained(model_path)

# Загружаем только валидационную часть датасета
data_files = {
    "validation": "mt5_ru_gen_eval.jsonl"
}
eval_dataset = load_dataset("json", data_files=data_files)["validation"]

# Переводим модель в режим оценки
model.eval()

# Отправляем на GPU, если доступен
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Параметры генерации
generation_kwargs = {
    "max_length": 256,
    "num_beams": 3,
    "length_penalty": 1,
    "early_stopping": True,
    "no_repeat_ngram_size": 3,
    "repetition_penalty": 1.2,
    "pad_token_id": tokenizer.pad_token_id
}

# Путь к файлу вывода
output_file = "generated_outputs.txt"

# Настройки батча
batch_size = 100  # Можно увеличить до 32–64, если точно влезает в VRAM

# Генерация
start_time = time()

with open(output_file, "w", encoding="utf-8") as f_out:
    total = min(1000, len(eval_dataset))  # Можно увеличить лимит
    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch_texts = [eval_dataset[i]["text"].strip().replace("\n", " ") for i in range(start_idx, end_idx)]

        # Токенизация с паддингом
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)

        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for i, (inp, out) in enumerate(zip(batch_texts, decoded_outputs), start=start_idx + 1):
            f_out.write(f"=== Пример {i} ===\n")
            f_out.write("Вход:\n")
            f_out.write(inp + "\n")
            f_out.write("Выход:\n")
            f_out.write(out + "\n")
            f_out.write("-" * 40 + "\n")

            print(f"=== Пример {i} ===")
            print("Вход:")
            print(inp)
            print("Выход:")
            print(out)
            print("-" * 40)

# Время генерации
end_time = time()
elapsed = end_time - start_time
avg_per_sample = elapsed / total

print(f"\n✅ Генерация завершена. Результаты сохранены в: {output_file}")
print(f"⏱️ Общее время: {elapsed:.2f} сек на {total} примеров")
print(f"⏱️ Среднее время на строку: {avg_per_sample:.3f} сек")
