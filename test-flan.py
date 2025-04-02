import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset

# Путь к уже сохранённой модели
model_path = "./flan-t5-large-ru-autobatch"

# Загружаем модель и токенизатор
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

# Загружаем только валидационную часть датасета
data_files = {
    "validation": "mt5_ru_gen_eval.jsonl"
}
eval_dataset = load_dataset("json", data_files=data_files)["validation"]

# Переводим модель в режим оценки
model.eval()

# Отправляем на GPU, если доступен
if torch.cuda.is_available():
    model.to("cuda")

# Параметры генерации
generation_kwargs = {
    "max_length": 256,
    "num_beams": 4,
    "early_stopping": True,
    "no_repeat_ngram_size": 3,
    "repetition_penalty": 1.2,
    "pad_token_id": tokenizer.pad_token_id
}

# Путь к файлу вывода
output_file = "generated_outputs.txt"

# Открываем файл для записи
with open(output_file, "w", encoding="utf-8") as f_out:
    num_examples = min(30, len(eval_dataset))
    for i in range(num_examples):
        input_text = eval_dataset[i]["text"].strip().replace("\n", " ")
        inputs = tokenizer(input_text, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)

        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Пишем в файл
        f_out.write(f"=== Пример {i+1} ===\n")
        f_out.write("Вход:\n")
        f_out.write(input_text + "\n")
        f_out.write("Выход:\n")
        f_out.write(prediction + "\n")
        f_out.write("-" * 40 + "\n")

        # Также печатаем в консоль
        print(f"=== Пример {i+1} ===")
        print("Вход:")
        print(input_text)
        print("Выход:")
        print(prediction)
        print("-" * 40)

print(f"\n✅ Генерация завершена. Результаты сохранены в: {output_file}")
