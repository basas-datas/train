# Установите необходимые пакеты
pip install --no-cache-dir transformers sentencepiece datasets safetensors torch accelerate scipy wandb

# Установите Git LFS
apt update && apt install -y git-lfs screen

# Инициализируйте Git LFS
git lfs install

# Клонируйте репозиторий прямо в /workspace
git clone https://github.com/basas-datas/train /workspace

# Создайте screen-сессию для обучения
screen -S train

# Перейдите в рабочую директорию
cd /workspace

# вводим команду, останется только нажать вроде
python train.py