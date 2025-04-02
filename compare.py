import time
import torch
from transformers import MT5ForConditionalGeneration, T5ForConditionalGeneration, AutoTokenizer, MT5Tokenizer
from datasets import load_dataset

# Пути к моделям
mt5_path = "./mt5"              # Локальная MT5 модель
byt5_path = "./unzipped_model"            # Локальная или скачанная ByT5 модель

# Путь к данным
validation_file = "mt5_validation_data-1.jsonl"

# Загрузка моделей и токенизаторов
mt5_tokenizer = MT5Tokenizer.from_pretrained(mt5_path)
mt5_model = MT5ForConditionalGeneration.from_pretrained(mt5_path).eval()

byt5_tokenizer = AutoTokenizer.from_pretrained(byt5_path)
byt5_model = T5ForConditionalGeneration.from_pretrained(byt5_path).eval()

# Выбор устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mt5_model.to(device)
byt5_model.to(device)

# Загрузка валидационной выборки
dataset = load_dataset("json", data_files={"validation": validation_file})
val_data = dataset["validation"]

# Функция предсказания
def predict(model, tokenizer, text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_length=64,
        num_beams=5,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Статистика
country_match = 0
city_match = 0
full_match = 0
mismatch_samples = []

start_time = time.time()

for example in val_data:
    text = example["text"]

    # Предсказания от MT5 и ByT5
    mt5_pred = predict(mt5_model, mt5_tokenizer, text)
    byt5_pred = predict(byt5_model, byt5_tokenizer, text)

    def split_prediction(pred):
        parts = pred.split(":", 1)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
        else:
            return parts[0].strip(), "unknown"

    mt5_country, mt5_city = split_prediction(mt5_pred)
    byt5_country, byt5_city = split_prediction(byt5_pred)

    if mt5_country == byt5_country:
        country_match += 1
    if mt5_city == byt5_city:
        city_match += 1
    if mt5_country == byt5_country and mt5_city == byt5_city:
        full_match += 1
    else:
        mismatch_samples.append({
            "text": text,
            "mt5_prediction": f"{mt5_country}:{mt5_city}",
            "byt5_prediction": f"{byt5_country}:{byt5_city}"
        })

end_time = time.time()
total_time = end_time - start_time
num_examples = len(val_data)
time_per_example = total_time / num_examples if num_examples > 0 else 0

# Вывод различий
print("Примеры, где хотя бы что-то не совпало между MT5 и ByT5 (макс. 80):")
for i, item in enumerate(mismatch_samples[:80]):
    print(f"\nПример {i+1}:")
    print(f"Текст:         {item['text']}")
    print(f"MT5 предсказал:  {item['mt5_prediction']}")
    print(f"ByT5 предсказал: {item['byt5_prediction']}")

# Итоги
print("\nРезультаты сравнения MT5 vs ByT5:")
print(f"Всего примеров: {num_examples}")
print(f"Совпало стран: {country_match}")
print(f"Совпало городов: {city_match}")
print(f"Полных совпадений: {full_match}")
print(f"Общее время выполнения: {total_time:.4f} сек.")
print(f"Время на одно сравнение: {time_per_example:.6f} сек.")
