# 🔭 AstroVision — Deep Space Object Detection


![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6F00)
![CBAM](https://img.shields.io/badge/CBAM-Attention%20Module-purple)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-Model%20Inference-005CED)
![Git](https://img.shields.io/badge/Git-Version%20Control-F05032?logo=git)
![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)
![License](https://img.shields.io/badge/License-MIT-brightgreen)
![Status](https://img.shields.io/badge/Project-Completed-success)



An end-to-end Deep Learning project for detecting and classifying astronomical objects in deep-space images using **YOLOv8 enhanced with CBAM (Convolutional Block Attention Module)**.

AstroVision provides an interactive **Streamlit web application** that allows users to upload astronomical images, detect celestial objects, visualize bounding boxes and confidence scores, filter object classes, inspect detected objects, and export prediction results.

---

# 📌 Project Overview

Astronomical images contain a large variety of celestial structures such as galaxies, nebulae, comets, and globular clusters. Detecting these objects manually can be time-consuming and difficult, especially in large astronomical surveys.

**AstroVision** uses a YOLOv8-based object detection model enhanced with **CBAM attention mechanisms** to automatically identify important regions and detect astronomical objects.

The system recognizes four major astronomical object categories:

* ☄️ Comet
* 🌌 Galaxy
* 🌟 Globular Cluster
* 🌫️ Nebula

The project includes:

* Deep Learning Object Detection
* YOLOv8 Architecture
* CBAM Attention Mechanism
* Astronomical Image Processing
* Model Training Pipeline
* ONNX Model Inference
* PyTorch Model Fallback
* Interactive Streamlit Dashboard
* Confidence and IoU Controls
* Object Filtering
* Bounding Box Visualization
* Prediction Export

---

# 🚀 Demo

### 🌐 Live Demo

Add your deployed Streamlit URL here:

```text
https://your-astrovision-app.streamlit.app/
```

> If you have not deployed the application yet, remove this section until a live URL is available.

---

# 📂 Project Structure

```text
AstroVision-Deep-Space-Object-Detection
│
├── app.py
├── best_fixed.onnx
├── best_fixed.pt
├── cosmica.yaml
├── train.py
├── yolov8_cbam.yaml
├── requirements.txt
├── packages.txt
├── .python-version
├── .gitignore
├── LICENSE
└── README.md
```

### Training Dataset Structure

The training configuration expects the Cosmica dataset in the following structure:

```text
dataset/
└── cosmica/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    │
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

The dataset itself is **not included in this repository**.

---

# 🔄 Workflow

```text
Astronomical Image
        │
        ▼
Image Upload
        │
        ▼
Image Preprocessing
        │
        ▼
YOLOv8 + CBAM Model
        │
        ▼
Feature Extraction
        │
        ▼
Channel Attention
        │
        ▼
Spatial Attention
        │
        ▼
Object Detection
        │
        ▼
Non-Maximum Suppression
        │
        ▼
Confidence Filtering
        │
        ▼
Celestial Object Classification
        │
        ▼
Bounding Box Visualization
        │
        ▼
Detection Dashboard
        │
        ▼
Export Results
```

---

# 🛠 Tech Stack

### Programming Language

* Python 3.11

### Deep Learning

* YOLOv8
* PyTorch
* Ultralytics
* CBAM Attention Mechanism

### Data Processing

* NumPy
* Pandas
* Pillow
* OpenCV

### Model Deployment

* ONNX
* ONNX Runtime
* Streamlit

### Visualization

* Streamlit
* Custom CSS
* Matplotlib/visualization utilities

### Version Control

* Git
* GitHub

---

# 📚 Libraries Used

The main dependencies include:

* `streamlit`
* `ultralytics`
* `torch`
* `torchvision`
* `numpy`
* `pillow`
* `opencv-python-headless`
* `onnxruntime`
* `pandas`

Install all required Python packages using:

```bash
pip install -r requirements.txt
```

---

# ✨ Features

### 🔭 Astronomical Object Detection

Detects four types of celestial objects:

| Class               | Description                                                   |
| ------------------- | ------------------------------------------------------------- |
| ☄️ Comet            | Icy Solar System objects that can develop a coma and tail     |
| 🌌 Galaxy           | Large gravitationally bound systems of stars and other matter |
| 🌟 Globular Cluster | Dense spherical groups of old stars                           |
| 🌫️ Nebula          | Large clouds of interstellar gas and dust                     |

### 🎯 Detection Controls

Users can interactively control:

* Confidence Threshold
* IoU / NMS Threshold
* Target Object Classes
* Bounding Box Thickness
* Detection Labels

### 🖼️ Image Upload

Supported image formats include:

```text
JPG
JPEG
PNG
WEBP
TIF
TIFF
```

### ✨ Sample Image

AstroVision also includes a synthetic cosmic field generator that can be used to test the application without uploading an image.

### 🤖 Multiple Model Backends

The application attempts to use:

```text
best_fixed.onnx
```

for optimized inference.

If ONNX loading fails, it falls back to:

```text
best_fixed.pt
```

---

# 🤖 Machine Learning Model

## Model Used

**YOLOv8 Object Detection**

The model architecture is enhanced using:

**CBAM — Convolutional Block Attention Module**

CBAM applies two attention mechanisms:

```text
Feature Map
     │
     ▼
Channel Attention
     │
     ▼
Refined Feature Map
     │
     ▼
Spatial Attention
     │
     ▼
Final Attention Feature Map
```

### Channel Attention

Channel attention helps the network determine **which feature channels are important**.

### Spatial Attention

Spatial attention helps the network determine **where important astronomical features are located**.

This can help the detector focus on relevant celestial structures while reducing the influence of background noise.

---

# 🧠 Detection Classes

The model is configured for four classes:

```yaml
names:
  0: comet
  1: galaxy
  2: globular_cluster
  3: nebula
```

---

# ⚙️ Model Configuration

The custom architecture is defined in:

```text
yolov8_cbam.yaml
```

The configuration contains:

* YOLOv8 backbone
* C2f feature extraction blocks
* SPPF
* CBAM attention
* Feature Pyramid Network
* PAN feature fusion
* Multi-scale detection heads

The detector uses three detection scales:

```text
P3 → Small Objects
P4 → Medium Objects
P5 → Large Objects
```

---

# 🏋️ Model Training

The training script is provided in:

```text
train.py
```

The training configuration uses:

```text
Epochs: 50
Image Size: 640
Batch Size: 4
```

The script automatically selects:

```text
GPU → CUDA
CPU → fallback
```

Run training with:

```bash
python train.py
```

### ⚠️ Training Requirements

The Cosmica dataset is not included in the repository.

Before training, the dataset must be available according to:

```text
dataset/cosmica/
```

and must contain the required YOLO-format images and labels.

---

# 🚀 Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/AstroVision-Deep-Space-Object-Detection.git
```

### 2. Enter the Project

```bash
cd AstroVision-Deep-Space-Object-Detection
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on Linux/macOS:

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the Streamlit Application

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

# 📊 Application Workflow

```text
1. Open AstroVision
        ↓
2. Upload an astronomical image
        ↓
3. Select detection settings
        ↓
4. Adjust confidence threshold
        ↓
5. Run YOLOv8 + CBAM inference
        ↓
6. View detected objects
        ↓
7. Inspect confidence scores
        ↓
8. Filter object classes
        ↓
9. Analyze bounding boxes
        ↓
10. Export prediction results
```

---

# 📈 Model Output

For every detected object, the application can determine:

* Object Class
* Confidence Score
* Bounding Box
* Object Location
* Number of Detected Objects

Example:

```text
Detected Objects
------------------------------
Comet              0.91
Galaxy             0.87
Nebula             0.83
Globular Cluster   0.78
```

---

# 💾 Model Files

The repository contains two trained model formats:

### ONNX

```text
best_fixed.onnx
```

Used as the preferred optimized inference backend.

### PyTorch

```text
best_fixed.pt
```

Used as a fallback model when ONNX inference cannot be initialized.

---

# 🔐 License

This project is licensed under the **MIT License**.

See the `LICENSE` file for complete license information.

---

# ⚠️ Project Status

### Current Status: Deployment / Inference Ready

The current repository contains the main components required to run the trained AstroVision detection application:

* ✅ Streamlit application
* ✅ Trained ONNX model
* ✅ PyTorch model
* ✅ YOLOv8 configuration
* ✅ Training script
* ✅ Dataset configuration
* ✅ Dependency files
* ✅ MIT License
* ✅ Git configuration

However, the repository does **not** currently contain the original Cosmica dataset or a standalone Python implementation of the custom CBAM module.

Therefore, the repository is best described as:

> **A trained-model astronomical object detection and deployment project, with the training configuration included.**

For complete reproducible training from scratch, the missing dataset and CBAM implementation/integration should also be added.

---

# 🚀 Future Improvements

* Improve model accuracy with additional astronomical datasets
* Hyperparameter tuning
* Add precision, recall and mAP evaluation
* Add confusion matrix visualization
* Add model performance dashboard
* Add batch image inference
* Add video/astronomical survey inference
* Add Docker support
* Add automated model training pipeline
* Add cloud deployment
* Add experiment tracking
* Add model versioning
* Add authentication
* Add database integration
* Add astronomy-specific metadata extraction
* Add FITS image support
* Add advanced astronomical image preprocessing

