from transformers import MT5ForConditionalGeneration, MT5Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import os
import wandb
import torch

# 🔧 Название модели и путь
model_name = "google/mt5-large"
run_id = "mt5-large-big_rain_1"
output_dir = f"./{run_id}"
start_batch_size = 50   # ⚠️ Начинаем с небольшого batch, чтобы избежать OOM
step_batch_size = 1

# 📦 Загружаем модель и токенизатор
model = MT5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = MT5Tokenizer.from_pretrained(model_name)

# 📂 Загружаем датасет
data_files = {
    "train": "train.jsonl",
    "validation": "eval.jsonl"
}
dataset = load_dataset("json", data_files=data_files)

# 🔠 Токенизация
def tokenize_function(examples):
    model_inputs = tokenizer(
        examples["text"], max_length=256, truncation=True, padding="max_length"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target"], max_length=256, truncation=True, padding="max_length"
        )
    # Заменяем PAD-токены на -100, чтобы не учитывать их в подсчёте loss
    labels["input_ids"] = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 🔑 Авторизация W&B
wandb.login(key="5f028bc0142fb7fa45bdacdde3c00dbbaf8bf98e")

# 🚀 Функция автоподбора batch size
def try_training_with_batch_size(batch_size_start):
    batch_size = batch_size_start
    while batch_size > 0:
        try:
            print(f"\n🚀 Пробуем batch_size = {batch_size}")
            training_args = TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="steps",
                eval_steps=100,
                learning_rate=3e-5,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                #fp16=True,  # Включайте при наличии подходящего GPU (A100 / V100 / T4)
                num_train_epochs=5,
                logging_steps=100,
                warmup_ratio=0.06,
                logging_first_step=True,
                weight_decay=0.01,
                logging_dir="./logs",
                save_total_limit=3,
                save_strategy="epoch",
                report_to="wandb",
                run_name=run_id,
                disable_tqdm=False,
                max_grad_norm=1.0
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"]
            )

            trainer.train(resume_from_checkpoint=True)
            return batch_size
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"❌ OOM на batch_size = {batch_size}, уменьшаем...")
                torch.cuda.empty_cache()
                batch_size -= step_batch_size
            else:
                raise e
    raise RuntimeError("Не удалось подобрать подходящий batch size 😢")

# 🏁 Запуск с автоподбором
optimal_batch_size = try_training_with_batch_size(start_batch_size)
print(f"\n✅ Успешно обучено с batch_size = {optimal_batch_size}")

# 💾 Сохраняем модель
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ Модель сохранена в {output_dir}")
