# PAF-Net: Progressive Alignment and Feature Fusion Network for Detecting Microfractures in X-Ray Images

## 01. Overview of the Proposed Model

PAF-Net (Progressive Alignment and Feature Fusion Network) is a lightweight detection architecture designed to enhance accuracy in skeletal X-ray microfracture detection.  
The model integrates three major components:  

**PDC (Pinwheel-shaped Dual-Split Attention Convolution):** Suppresses background interference and enhances skeletal region perception.  
**CAGS (Content-Aware Guided Sampling):** Dynamically refines upsampling to restore fine fracture details.  
**LR (Low-Rank Asymmetric Reconstruction):** Combines low-rank decomposition and asymmetric convolution for efficient directional feature fusion.  
<img width="915" height="687" alt="image" src="https://github.com/user-attachments/assets/d39568db-0ba6-4fa0-aff3-54aa66ce62c6" />

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

## 05. Dataset

### (a) Bone Fracture Detection Dataset (RoboFlow)

This dataset contains **2,521 annotated skeletal X-ray images** for bone fracture detection, divided into training, validation, and test sets.  
It is widely used for evaluating lightweight detection architectures.

**Download Link:** [https://universe.roboflow.com/yukseklisans/fracnonfrac](https://universe.roboflow.com/yukseklisans/fracnonfrac)

---

### (b) GRAZPEDWRI-DX Dataset (Medical University of Graz)

The **GRAZPEDWRI-DX** dataset is a large-scale pediatric wrist trauma X-ray dataset collected between 2008 and 2018, containing **over 20,000 labeled images from more than 6,000 patients**.  
It serves as a benchmark for evaluating the robustness of medical image detection models.

**Download Link:** [https://www.nature.com/articles/s41597-022-01328-z](https://www.nature.com/articles/s41597-022-01328-z)
