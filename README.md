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
python start_train.py --model cfg/models/pafnet.yaml                       --dataset_config cfg/datasets/BoneFracture.yaml                       --epochs 150                       --output_dir E:/PAFNet_train/                       --run_name PAFNet_BoneFracture_Exp01
```

### Notebook (.ipynb)

```python
from ultralytics import YOLO
from multiprocessing import freeze_support

# Set paths
model_path = 'E:/Degree_project/PAF-Net/cfg/models/pafnet.yaml'
data_path = 'E:/Degree_project/PAF-Net/cfg/datasets/BoneFracture.yaml'
project_path = 'E:/PAFNet_train/'

# Load model
model = YOLO(model_path)

if __name__ == '__main__':
    freeze_support()
    model.train(data=data_path, epochs=150, project=project_path, name='PAFNet_BoneFracture_Exp01')
```

---

## 04. Model Validation Example

```python
from ultralytics import YOLO

# Load trained model
model = YOLO("E:/PAFNet_train/PAFNet_BoneFracture_Exp01/weights/best.pt")

# Validate
metrics = model.val()
metrics.box.map     # mAP50-95
metrics.box.map50   # mAP50
metrics.box.map75   # mAP75
metrics.box.maps    # List of mAP values per class
```

---

## 05. Key Features

- Progressive feature alignment with PDC and CAGS modules  
- Low-rank asymmetric feature fusion for lightweight performance  
- Enhanced robustness against background interference  
- Improved restoration of small-scale fracture structures  
- Compatible with YOLOv8 training framework  

---

## 06. Citation

```
@article{Zhang2025PAFNet,
  title={Progressive Alignment and Feature Fusion Network for Detecting Microfractures in X-Ray Images},
  author={Dan Zhang and Yitao Mai and Xiaohuan Zhang and Penghao Jiang},
  journal={Medical Imaging Research},
  year={2025}
}
```

---

## 07. Acknowledgments

This work was supported by the Education Department of Guangdong Province, project “Research on an Accurate Segmentation Model for PET/CT Multimodal Lung Cancer Lesion Regions Based on Feature-Adaptive U-Net” (Grant No. 2024KTSCX088).
