from transformers import AutoTokenizer, MT5ForConditionalGeneration
from huggingface_hub import HfApi, HfFolder
import os

# 🔐 Токен из переменной окружения
hf_token = os.environ["huggingface"]

# 📁 Путь к чекпоинту
local_checkpoint_dir = "./mt5-large-big_rain_1/checkpoint-14880"

# 🏷️ Название репозитория
repo_id = "ajkndfjsdfasdf/mt5-large-big_rain_1-checkpoint-14880"

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

# ✅ Загружаем модель из чекпоинта
model = MT5ForConditionalGeneration.from_pretrained(local_checkpoint_dir)

# ✅ Загружаем токенизатор из базовой модели (если в чекпоинте его нет)
tokenizer = AutoTokenizer.from_pretrained("google/mt5-large")

# 🚀 Публикуем в Hugging Face
model.push_to_hub(repo_id, token=hf_token, commit_message="🚀 Push checkpoint 14880 model")
tokenizer.push_to_hub(repo_id, token=hf_token, commit_message="🚀 Push tokenizer from base model")

print(f"✅ Чекпоинт загружен в: https://huggingface.co/{repo_id}")
