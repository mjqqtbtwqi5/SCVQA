# SCVQA

Screen Content Video Quality Assessment

# Description

This project will be separated into 2 parts:

1. CNN features extraction using Anaconda python enviroment
2. Deep learning model training with Google colab.

# CNN features extraction

```bash
conda remove -n SCVQA-env --all

conda create -n SCVQA-env python=3.9.18

conda activate SCVQA-env

conda config --env --add channels conda-forge

conda install -c conda-forge ffmpeg

pip install -r requirements.txt

python cnn_feature_extraction.py

conda deactivate
```

# Training

python train.py --model={LSTM,Transformer} --database={CSCVQ,SCVD} --cnn_extraction={ResNet50}

Optional args
--batch_size, default=32
--num_workers, default=0
--num_epochs, default=100
--learning_rate, default=0.00001
--seed, type=int, default=22035001

```bash
python train.py --model=LSTM --database=CSCVQ --cnn_extraction=ResNet50 --batch_size=8 --num_epochs=200
python train.py --model=LSTM --database=SCVD --cnn_extraction=ResNet50 --batch_size=32 --num_epochs=200

python train.py --model=Transformer --database=CSCVQ --cnn_extraction=ResNet50 --batch_size=8 --num_epochs=200
python train.py --model=Transformer --database=SCVD --cnn_extraction=ResNet50 --batch_size=32 --num_epochs=200

python train.py --model=VSFA_GRU --database=CSCVQ --cnn_extraction=ResNet50 --batch_size=8 --num_epochs=200
python train.py --model=VSFA_GRU --database=SCVD --cnn_extraction=ResNet50 --batch_size=32 --num_epochs=200
```
