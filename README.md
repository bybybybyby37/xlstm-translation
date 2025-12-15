## Training the translation dataset using the xLSTM architecture

## 1. Environment setup

```
conda env create -n xlstm-translation -f environment_pt240cu124.yaml
conda activate xlstm-translation

pip install -r requirements.txt
```

## 2. Test Run
Firstly activate the conda environment
```
conda activate xlstm-translation
```
Then run the training script (you can modify the hyperparameters)

```
# 1:0 Variant（mLSTM-only）
python -m scripts.train_iwslt17_xlstm --config config/iwslt17_xlstm10.yaml --variant 10 --eval_split test

# 0:1 Variant（sLSTM-only）
python -m scripts.train_iwslt17_xlstm --config config/iwslt17_xlstm01.yaml --variant 01 --eval_split test

# 1:1 Varivant（mLSTM + sLSTM）
python -m scripts.train_iwslt17_xlstm --config config/iwslt17_xlstm11.yaml --variant 11 --eval_split test
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
