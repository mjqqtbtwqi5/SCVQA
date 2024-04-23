# SCVQA

Screen Content Video Quality Assessment

Video data and feature data will not be included since file sizes of them are too large.

# Description

This project will be separated into 2 parts:

1. CNN Features Extraction
   - Local machine: cnn_feature_extraction.py
   - Google Colab: cnn_feature_extraction.ipynb
2. Temporal Memory Effects (deep learning model training)
   - Local machine: train.py
   - Google Colab: train.ipynb

# Setup for Anaconda env

```bash
conda remove -n SCVQA-env --all

conda create -n SCVQA-env python=3.9.18

conda activate SCVQA-env

conda config --env --add channels conda-forge

conda install -c conda-forge ffmpeg

pip install -r requirements.txt


# conda deactivate
```

# CNN Features Extraction/Training using Anaconda env

python cnn_feature_extraction.py

python train.py --model={LSTM,Transformer,VSFA_GRU} --database={CSCVQ,SCVD} --cnn_extraction={\_ResNet18,\_ResNet34,\_ResNet50,\_ResNet101,\_ResNet34_ResNet50}

Optional args for train.py

1. --batch_size, default=32
2. --num_workers, default=0
3. --num_epochs, default=100
4. --learning_rate, default=0.00001
5. --seed, type=int, default=22035001

```bash
python train.py --model=Transformer --database=CSCVQ --cnn_extraction=ResNet50 --batch_size=8 --num_epochs=1000
python train.py --model=Transformer --database=SCVD --cnn_extraction=ResNet50 --batch_size=32 --num_epochs=1000

python train.py --model=LSTM --database=CSCVQ --cnn_extraction=ResNet50 --batch_size=8 --num_epochs=1000
python train.py --model=LSTM --database=SCVD --cnn_extraction=ResNet50 --batch_size=32 --num_epochs=1000

python train.py --model=VSFA_GRU --database=CSCVQ --cnn_extraction=ResNet50 --batch_size=8 --num_epochs=1000
python train.py --model=VSFA_GRU --database=SCVD --cnn_extraction=ResNet50 --batch_size=32 --num_epochs=1000
```
