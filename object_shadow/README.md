# Object Shadow Cues
<p align="center">
<img height="400" alt="Architecture for Object Shadow Classifier" src="../assets/object_shadow_cues.jpg">
</p>

## Getting Started

**The main packages are listed below**
```bash
#Conda
python=3.11.4
torchaudio=2.0.2=py311_cu117
torchvision=0.15.2=py311_cu117
tqdm=4.65.0
pillow=10.2.0
#pip
pandas==2.1.1
scikit-learn==1.3.2
matplotlib==3.8.0
```

**Download the Trained Model [here](https://drive.google.com/drive/folders/1pg6pW1A7n-UGb0HXkm0a8p0HDkc79sDS?usp=sharing) and place them in the `checkpoints` folder**

**To extracting Shadow and Object masks, use [SSISv2](https://github.com/stevewongv/SSIS)**

## Usage

**Training**

Run one of the following:
```bash
python train.py --category indoor
python train.py --category outdoor
python train.py --category combined
```

**Testing**

Run one of the following:
```bash
python test.py --category indoor
python test.py --category outdoor
python test.py --category combined
```