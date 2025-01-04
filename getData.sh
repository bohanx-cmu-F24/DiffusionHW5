mkdir data
mkdir experiments
mkdir pretrained
python3 -m venv diffusionEnv
source diffusionEnv/bin/activate
pip install -r requirements.txt
gdown https://drive.google.com/uc?id=1ZIfi3Tqv_KQBBJruoQ_VA90mesWoZQrI
gdown https://drive.google.com/uc?id=1gjWrgfcw7UBquKuR7icQpCmZHZ6BEquH

sudo apt update
sudo apt install unrar-free
unzip imagenet100_128x128.zip -d data
nvidia-smi