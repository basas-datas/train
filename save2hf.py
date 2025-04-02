from transformers import AutoTokenizer, T5ForConditionalGeneration
from huggingface_hub import HfApi, HfFolder
import os

# 🔐 Токен из переменной окружения
hf_token = os.environ["huggingface"]

# 📁 Локальная директория с моделью
local_model_dir = "./flan-t5-autobatch"

# 🏷️ Репозиторий (ВАЖНО: должен быть в формате 'username/repo_name')
repo_id = "ajkndfjsdfasdf/flan-5-small-bigdataset"

# 🔐 Авторизация
api = HfApi()
HfFolder.save_token(hf_token)

# 🔍 Проверяем наличие репозитория
try:
    api.repo_info(repo_id, token=hf_token)
    print(f"📦 Репозиторий {repo_id} уже существует.")
except:
    print(f"📦 Репозиторий {repo_id} не найден. Создаём...")
    api.create_repo(repo_id=repo_id, token=hf_token, repo_type="model", exist_ok=True)

# 📦 Загружаем модель и токенизатор
model = T5ForConditionalGeneration.from_pretrained(local_model_dir)
tokenizer = AutoTokenizer.from_pretrained(local_model_dir)

# 🚀 Пушим в корень репозитория
model.push_to_hub(repo_id, token=hf_token, commit_message="🚀 Push latest model to root")
tokenizer.push_to_hub(repo_id, token=hf_token, commit_message="🚀 Push latest tokenizer to root")

print(f"✅ Модель загружена в: https://huggingface.co/{repo_id}")
