# 🔭 AstroVision: AI Deep Space Object Detection

**AstroVision** is a state-of-the-art Deep Space Object Detection web application powered by **YOLOv8** enhanced with **CBAM (Convolutional Block Attention Module)**. It automatically detects, classifies, and analyzes celestial objects in optical astronomical survey images.

---

## 🌟 Key Features

- **Deep Space UI Theme**: Sleek glassmorphism aesthetic built for astronomers, astrophotographers, and space enthusiasts.
- **CBAM Attention Neural Network**: Enhanced feature extraction using Channel Attention and Spatial Attention mechanisms for precision detection of faint cosmic objects.
- **4 Astronomical Classes Recognized**:
  - ☄️ **Comet** (Class 0)
  - 🌌 **Galaxy** (Class 1)
  - 🌟 **Globular Cluster** (Class 2)
  - 🌫️ **Nebula** (Class 3)
- **Interactive Controls Sidebar**:
  - Real-time Confidence Threshold & IoU NMS Threshold Sliders.
  - Multi-select Target Object Filtering.
  - Custom Bounding Box Styling.
- **Multi-Tab Workspace**:
  - **🔭 Detection Dashboard**: Side-by-side original vs annotated views, KPI cards, and detailed detection table.
  - **🔍 Object Crop Gallery**: Auto-cropped, zoomed-in cards of each detected celestial object.
  - **📘 Astronomical Field Guide**: Comprehensive scientific descriptions for each object type.
  - **💾 Export & Reports**: Download annotated PNG images, structured CSV tables, and full JSON reports.
- **Multi-Backend Runtime Support**: Automatically loads optimized `ONNX` runtime weights (`best_fixed.onnx`) with fallback to PyTorch (`best_fixed.pt`).

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation

Ensure Python 3.8+ is installed on your system. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Running the Web Application

Launch the Streamlit interface:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 📁 Repository Structure

```
AstroVision-main/
├── app.py              # Main Streamlit Web Application
├── best_fixed.onnx     # Optimized ONNX model weights
├── best_fixed.pt       # PyTorch model weights fallback
├── cosmica.yaml        # Dataset configuration & class definitions
├── train.py            # Training script for YOLOv8 + CBAM
├── yolov8_cbam.yaml    # Neural network architecture definition (YOLOv8 + CBAM)
├── requirements.txt    # Python package dependencies
├── packages.txt        # System level dependencies
└── README.md           # Documentation
```

---

## 🔬 Model Architecture

AstroVision integrates **CBAM (Convolutional Block Attention Module)** into YOLOv8's feature extraction backbone and FPN/PAN neck:
1. **Channel Attention Module (CAM)**: Focuses on *what* celestial spectral features are important across channels.
2. **Spatial Attention Module (SAM)**: Focuses on *where* astronomical features are located in space, suppressing background star noise.

---

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.
