mkdir data
mkdir experiments
mkdir pretrained
python3 -m venv diffusionEnv
source diffusionEnv/bin/activate
pip install -r requirements.txt
gdown https://drive.google.com/uc?id=1ZIfi3Tqv_KQBBJruoQ_VA90mesWoZQrI
gdown https://drive.google.com/uc?id=1qc2Gq6vdTTXhtu1qG629jbKbPaduHf6d
sudo apt update
sudo apt install unrar-free
unrar x imagenet100_128x128.rar DiffusionHW5/data -y
nvidia-smi