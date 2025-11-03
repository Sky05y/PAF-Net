# 🧠 PAF-Net: Pyramid Attention Fusion Network for Enhanced Object Detection

## 01. Overview of the Proposed Model

We propose **PAF-Net (Pyramid Attention Fusion Network)**, a novel architecture designed to improve the representation of multi-scale and small-object features in dense object detection tasks.  
In PAF-Net, we integrate a **Pyramid Attention Fusion (PAF) module** that hierarchically fuses features from multiple levels with adaptive attention weighting. This design enhances context awareness and maintains spatial detail, addressing common weaknesses in existing one-stage detectors.  

In addition, the **ReLU6-based activation** and **lightweight fusion blocks** are applied to reduce computational complexity while maintaining accuracy.  
*(학위논문 그림 1)*

---

## 02. Environment

Before running training or validation, please install all required dependencies using the following command:

```bash
pip install -r requirements.txt
```

---

## 03. Model Training Example

### 💻 Command Line
To start training, run the following command:

```bash
python start_train.py --model cfg/models/pafnet.yaml                       --dataset_config cfg/datasets/COCO.yaml                       --epochs 150                       --output_dir E:/PAFNet_train/                       --run_name PAFNet_COCO_Exp01
```

### 📓 Notebook Example (.ipynb)

```python
from ultralytics import YOLO
from multiprocessing import freeze_support

# Set the model path
model_path = 'E:/Degree_project/PAF-Net/cfg/models/pafnet.yaml'

# Set the dataset config path
data_path = 'E:/Degree_project/PAF-Net/cfg/datasets/COCO.yaml'

# Set the project output path
project_path = 'E:/PAFNet_train/'

# Load the model
model = YOLO(model_path)

if __name__ == '__main__':
    freeze_support()

    # Train the model
    model.train(data=data_path, epochs=150, project=project_path, name='PAFNet_COCO_Exp01')
```

---

## 04. Model Validation Example

### 📓 Notebook Example (.ipynb)

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO("E:/PAFNet_train/PAFNet_COCO_Exp01/weights/best.pt")  # Custom trained model

# Validate the model
metrics = model.val()  # Retains dataset and settings automatically

# Display key metrics
metrics.box.map     # mAP50-95
metrics.box.map50   # mAP50
metrics.box.map75   # mAP75
metrics.box.maps    # List of mAP50-95 values for each category
```

---

## 05. Key Features

- **Pyramid Attention Fusion (PAF)** module for hierarchical feature integration  
- **Lightweight multi-scale fusion** preserving spatial detail  
- **Adaptive attention weighting** to highlight small-object features  
- Compatible with YOLOv8 training framework  
- **ReLU6 activation** for better numerical stability  

---

## 06. Citation

If you use **PAF-Net** in your research, please cite:

```
@article{YourName2025PAFNet,
  title={PAF-Net: Pyramid Attention Fusion Network for Enhanced Object Detection},
  author={Your Name and Others},
  year={2025},
  journal={Thesis / Conference Name},
}
```

---

## 07. Acknowledgments

This work builds upon the **Ultralytics YOLOv8** framework and integrates techniques inspired by **CBAM**, **FPN**, and **EfficientDet**.  
Special thanks to the open-source community for providing invaluable tools for vision research.
