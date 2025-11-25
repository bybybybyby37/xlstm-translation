## Training the translation dataset using the xLSTM architecture

## 1. Environment setup

```
conda create -n xlstm python=3.10 -y
conda activate xlstm

#Can be adjusted based on your cuda version
pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 --index-url https://download.pytorch.org/whl/cu118 

pip install -r requirements.txt
```

## 2. Test Run
Firstly activate the conda env
```
conda activate xlstm
```
Then run the training script (you can modify the hyperparameters)
```
python -m scripts.train_iwslt17_xlstm
```

## Citation
@inproceedings{beck:24xlstm,
  title = {xLSTM: Extended Long Short-Term Memory}, 
  author = {Maximilian Beck and Korbinian Pöppel and Markus Spanring and Andreas Auer and Oleksandra Prudnikova and Michael Kopp and Günter Klambauer and Johannes Brandstetter and Sepp Hochreiter},
  booktitle = {Thirty-eighth Conference on Neural Information Processing Systems},
  year = {2024},
  url = {https://arxiv.org/abs/2405.04517}, 
}

@article{beck:25xlstm7b,
  title = {{xLSTM 7B}: A Recurrent LLM for Fast and Efficient Inference},
  author = {Maximilian Beck and Korbinian Pöppel and Phillip Lippe and Richard Kurle and Patrick M. Blies and Günter Klambauer and Sebastian Böck and Sepp Hochreiter},
  booktitle = {Forty-second International Conference on Machine Learning},
  year = {2025},
  url = {https://arxiv.org/abs/2503.13427}
}
