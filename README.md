# PAF-Net: Progressive Alignment and Feature Fusion Network for Detecting Microfractures in X-Ray Images

## 01. Overview of the Proposed Model

PAF-Net (Progressive Alignment and Feature Fusion Network) is a lightweight detection architecture designed to enhance accuracy in skeletal X-ray microfracture detection.  
The model integrates three major components:  

- **PDC (Pinwheel-shaped Dual-Split Attention Convolution):** Suppresses background interference and enhances skeletal region perception.  
- **CAGS (Content-Aware Guided Sampling):** Dynamically refines upsampling to restore fine fracture details.  
- **LR (Low-Rank Asymmetric Reconstruction):** Combines low-rank decomposition and asymmetric convolution for efficient directional feature fusion.  

---

## 02. Environment

Install the dependencies before training or validation:

```bash
pip install -r requirements.txt
```

---

## 03. Model Training Example

### Command Line

```bash
python train.py --model cfg/models/PAF-Net.yaml --dataset_config cfg/datasets/BoneFracture.yaml --epochs 200 --output_dir ./output --run_name PAFNet_BoneFracture_Exp01
```

---

## 04. Model Validation Example

```bash
python val.py
```

---
