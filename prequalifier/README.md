# Prequalifier

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

Download the Trained Model [here](https://drive.google.com/drive/folders/1QGQm9SjJ2SyB0FJObOsS2tpVULyHXMCZ?usp=share_link) and place them in the `checkpoints` folder

## Dataset Download

The dataset can be found [here](https://huggingface.co/datasets/amitabh3/Projective-Geometry/tree/main). Download them and place them in the `../dataset` directory.

To get started, download the Kandinsky_Indoor.zip and Kandinsky_Outdoor.zip

## Usage

**Testing**

Run one of the following:
```bash
python prequalifier_test.py --category indoor
python prequalifier_test.py --category outdoor
python prequalifier_test.py --category combined
```

After testing, results will be printed and plots will be generated in the `plots` directory