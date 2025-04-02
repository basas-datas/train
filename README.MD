pip install --no-cache-dir transformers sentencepiece datasets safetensors torch accelerate scipy wandb
apt update && apt install -y git-lfs screen
git lfs install
git clone https://github.com/basas-datas/train /workspace
screen -S train
cd /workspace
python train.py