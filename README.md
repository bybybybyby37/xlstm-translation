## Training the translation dataset using the xLSTM architecture

## 1. Environment setup

```
conda create -n xlstm python=3.10 -y
conda activate xlstm

#Can be adjusted based on your cuda version
pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 --index-url https://download.pytorch.org/whl/cu118 

pip install -r requirements.txt
```