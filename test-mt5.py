import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from datasets import load_dataset

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

# Переводим модель в режим оценки (обязательно для генерации)
model.eval()

# Если доступен GPU, отправляем модель на видеокарту
if torch.cuda.is_available():
    model.to("cuda")

# Генерируем и выводим 100 примеров
num_examples = min(100, len(eval_dataset))  # Если в eval меньше 100, берем сколько есть
for i in range(num_examples):
    input_text = eval_dataset[i]["text"]

    # Токенизируем
    inputs = tokenizer(input_text, return_tensors="pt")

    # При наличии GPU переносим тензоры
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Генерация (без подсчета градиента)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,      # Максимальная длина генерируемого ответа
            num_beams=4,         # beam search для более осмысленного результата
            early_stopping=True
        )

    # Декодируем результат
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"=== Пример {i+1} ===")
    print("Вход:")
    print(input_text)
    print("Выход:")
    print(prediction)
    print("-" * 40)
