<div align="center">

# Minerals Classification

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.8+-blue?style=flat&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.8+-ee4c2c?style=flat&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.5+-792ee5?style=flat&logo=pytorchlightning&logoColor=white"></a>

<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra 1.1-89b8cd?style=flat&labelColor=gray"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=flat&labelColor=gray"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

MLOps approach to task image classification. Deep learning framework with PyTorch Lightning, Parameter/argument configuration with Hydra, Hyperparameter tuning with Optuna, and Logging with Weight&Bias.

## How to Inference
Weights dari model terbaik (ResNet18) sudah disedikan pada `weights/resnet18.pt`.
```bash
python inference.py \
    weights_path=./weights/resnet18.pt \
    source_dir=./data/test
```
Hasil inference berupa `.json` yang terdiri dari:
```json
{"test_image_name": {"class": "prediction_class", "confidence": "confidence"}}
{"bornite2": {"class": "bornite", "confidence": 0.34}}
```

## How to Train Model
Anda dapat dengan mudah melakukan training pada model menggunakan `configs/train.yaml` yang telah disediakan. Anda dapat mengubah isi `configs/train.yaml` sesuai kebutuhan.
```bash
# normal training
python src/train.py 

# change configuration
python src/train.py \
    model.net._target_=torchvision.models.resnet18 \
    datamodule.batch_size=32 \
    trainer.max_epochs=20
```
Anda juga dapet menggunakan lebih dari satu GPU untuk training (multi-gpu) dengan mengubah konfigurasi `configs/train.yaml` atau dengan:
```bash
python src/train.py \
    trainer=gpu \
    device=1,2,3,4
```

## Hyperparameter Tuning
Anda dapat dengan mudah melakukan hyperparameter tuning (mengganti model, menggubah learning rate, etc) dengan terlebih dahulu mengedit `configs/hparams_search/minerals_optuna.yaml` sesuai dengan kebutuhan. Selanjutnya, script train dapat dijalankan:
```bash
python src/train.py -m hparams_search=minerals_optuna
```

## Convert Model from Checkpoint
Semua hasil training disimpan dalam bentuk checkpoint (`.ckpt`) anda dapat mengubahnya menjadi bentuk `.pt` atau `.onnx`, dengan mengedit `configs/convert.yaml` terlebih dahulu:
```bash
python convert.py

# atau langsung di terminal
python convert.py \
    weights_path=logs/train/runs/2022-09-23_09-45-30/checkpoints/epoch_008.ckpt \
    convert_to=pytorch \
    save_path=weights/resnet18_xx.pt
```

## Evaluation Model
Evaluasi dapat dengan mudah dilakukan dengan pertama-tama mengedit `configs/eval.yaml` sesuai dengan kebutuhan. Setelahnya dapat menjalankan script evaluasi:
```bash
python src/eval.py

# atau langsung di terminal
python src/eval.py \
    ckpt_path=logs/train/multiruns/2022-09-23_14-23-34/3/checkpoints/epoch_014.ckpt
```
Hasil dari model evaluation adalah `confmat.png`, `pr_curve.png` dan `roc_curve.png` yang simpan pada root. 

## Reference
- [Lightning Hydra Template](https://github.com/ashleve/lightning-hydra-template)