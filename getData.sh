mkdir data
mkdir experiments
python3 -m venv diffusionEnv
source diffusionEnv/bin/activate
pip install -r requirements.txt
gdown https://drive.google.com/uc?id=1qc2Gq6vdTTXhtu1qG629jbKbPaduHf6d
sudo apt update
sudo apt install p7zip-full
7z x imagenet100_128x128.rar  -o/data
nvidia-smi