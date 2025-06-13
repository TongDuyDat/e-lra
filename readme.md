
# E-LRA: Efficient Edge-Aware GAN for Lightweight and Accurate Polyp Segmentation

This repository contains the official implementation of **E-LRA**, a lightweight GAN-based framework for polyp segmentation in colonoscopy images. E-LRA achieves state-of-the-art performance with only **1.07 million parameters**, making it highly suitable for real-time clinical applications.


## Table of Contents
1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Datasets](#datasets)
6. [Results](#results)
7. [Citation](#citation)
8. [License](#license)
9. [Contact](#contact)


## Introduction
E-LRA is designed to address the challenges of polyp segmentation in colonoscopy images, offering a balance between accuracy

## Key Features
- **Lightweight Design:** Only **1.07 million parameters**, making it **17x smaller** than the smallest existing model.
- **State-of-the-Art Performance:** Achieves a **Dice coefficient of 0.959** and an **IoU of 0.925** on the challenging Kvasir_SEG dataset.
- **Robust Generalization:** Validated on five benchmark datasets, including Kvasir-SEG, CVC-ClinicDB, ETIS, CVC-300, and PolypGen.
- **Real-Time Applicability:** 

---
## Installation
### Prerequisites
### steps
1. Clone this repository:
   ```bash
   git clone https://github.com/TongDuyDat/lgps_pytorch.git
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
## Datasets
- Kvasir-SEG: [Download](https://datasets.simula.no/kvasir-seg/)
- CVC-ClinicDB: [Download](https://polyp.grand-challenge.org/CVCClinicDB/)
- ETIS: Download [Download](https://polyp.grand-challenge.org/ETISLarib/)
- CVC-300: [Download](http://pages.cvc.uab.es/CVC-Colon/)
- PolypGen: [Download](https://drive.google.com/drive/folders/16uL9n84SrMt7IiQFzTUQNaJ9TbHJ8DhW)

Place the datasets in the data/ directory.
1. Train the model:
   ```bash
   python train.py
2. Evaluate the model:download the pretrained model from [here](https://)
   ```bash
   python benchmark.py --data_path "data/CVC-ClinicDB" --model_path "XXX.h5"
## Results


## Citation

## License
