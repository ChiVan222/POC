# POC

## Installation Guide

### 1️⃣ Clone the repository

```bash
git clone https://github.com/ChiVan222/POC.git
cd POC
```

### 2️⃣ Create a Python environment

```bash
conda create -n poc python=3.10
conda activate poc
```

### 3️⃣ Install PyTorch

```bash
# Adjust the command based on your GPU / CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4️⃣ Install GroundingDINO

```bash
cd GroundingDINO && python setup.py install
cd ..
```

### 5️⃣ Install the remaining dependencies

```bash
pip install -r requirements.txt
```

### 6️⃣ Install pretrained weights
The weight can downloaded from  [here](https://drive.google.com/drive/folders/1broKlAkmrGxxFllXA96NCW7VsXt2O_1D?usp=drive_link) in the ckpt.
Place them as the following: 

1. **VG Pretrained Weights**

```text
POC/
└─ weights/
    └─ vg-pretrain-coco-swinb.pth
```

2. **GroundingDINO Weights**

```text
POC/GroundingDINO/
└─ weights/
    └─ groundingdino_swint_ogc.pth
```
3. **Model ckpt** 
Create a checkpoints folder and place the epoch_50.pth 
```text
POC/
└─ checkpoints/
    └─ epoch_50.pth
```
### 7️⃣ Install dataset

Follow the guide from [Dataset](https://github.com/gpt4vision/OvSGTR/blob/master/datasets/data.md).
The preprocessed features can downloaded from  [here](https://drive.google.com/drive/folders/1broKlAkmrGxxFllXA96NCW7VsXt2O_1D?usp=drive_link) in the Preprocessed Data.
The `vg_data` directory structure should look like this:

```
POC/
└─ vg_data/
    ├─ VG_100K/
    ├─ vg_stats.pt          # frequency bias
    ├─ zeroshot_triplet.pytorch
    ├─ stanford_filtered/
    │   ├─ image_data.json
    │   ├─ VG-SGG-dicts.json  # add split_GLIPunseen
    │   └─ VG-SGG.h5
    └─ features/
        ├─ test_features_with_ids.h5
        └─ train_features_negatives.h5
```

> Make sure the `.h5` feature files are placed inside `vg_data/features/`.

---

## Evaluation

To evaluate the model, simply run:

```bash
python evaluate.py
```
