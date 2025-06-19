<div align="center">

# 🎥 DeepDetective

**AI-Powered Deepfake Video Detection**

[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-orange?logo=streamlit)](https://streamlit.io/) 
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red?logo=pytorch)](https://pytorch.org/) 
[![License: Custom](https://img.shields.io/badge/license-Custom-lightgrey.svg)](#license)

</div>

---

## 🚀 Overview

**DeepDetective** is an AI-powered tool for detecting deepfake videos. Leveraging a hybrid deep learning model (ResNet50 + BiLSTM), it analyzes video frames to determine authenticity. The app features a user-friendly web interface for uploading videos and instantly receiving predictions and confidence scores.

---

## ✨ Features

- 🎬 **Upload & Analyze**: Supports MP4, AVI, MOV video files
- 🧑‍💻 **Face Detection**: Uses MTCNN for robust face extraction
- 🧠 **Hybrid Model**: ResNet50 for spatial, BiLSTM for temporal analysis
- 📊 **Real/Fake Prediction**: With confidence score
- 🌐 **Streamlit UI**: Interactive and easy to use

---

## 🏗️ Model Architecture

- **Backbone:** ResNet50 (pretrained on ImageNet)
- **Temporal Modeling:** Bidirectional LSTM (BiLSTM)
- **Classifier Head:** Fully connected layers with dropout & sigmoid
- **Input:** Sequence of frames (default: 6 per video)
- **Output:** Probability score (0 = real, 1 = fake)

---

## 📁 File Structure

```
├── app.py                # Main Streamlit app
├── requirements.txt      # Python dependencies
├── model_ep10.pth        # Trained model weights (PyTorch)
```

---

## ⚡ Quickstart

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Starbuster2004/DeepDetective.git
   cd DeepDetective
   ```
2. **(Optional) Create a virtual environment:**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Add the model weights:**
   - Place `model_ep10.pth` in the project root directory.

5. **Run the app:**
   ```bash
   streamlit run app.py
   ```
   - Open the provided local URL in your browser.
   - Upload a video and click **Analyze Video**.

---

## 🧬 Model Details

- **Training:**
  - Trained on labeled real/fake video datasets
  - 6 frames sampled per video
  - Faces detected/cropped, fallback to center crop if needed
  - Features extracted (ResNet50) → Temporal modeling (BiLSTM) → Classification
- **Preprocessing:**
  - Face detection (MTCNN)
  - Center crop fallback
  - Resize to 224x224, normalization

---

## 📝 Notes

- Expects videos with at least 6 frames (duplicates last frame if fewer)
- Uses GPU if available, otherwise CPU
- `model_ep10.pth` is required for predictions

---

## 🚚 Push to GitHub

To push your code to [Starbuster2004/DeepDetective](https://github.com/Starbuster2004/DeepDetective.git):

```bash
git init  # If not already initialized
git remote add origin https://github.com/Starbuster2004/DeepDetective.git
git add .
git commit -m "Initial commit: DeepDetective app, model, and requirements"
git branch -M main
git push -u origin main
```
> **Note:** For large files like `model_ep10.pth`, use [Git LFS](https://git-lfs.github.com/).

---

## 📜 License

This project currently does not include a license file. Please add a LICENSE file to specify usage rights.

---

## 🙏 Acknowledgements

- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch)
- [Streamlit](https://streamlit.io/)

---

<div align="center">
  <b>For questions or issues, please open an issue on the <a href="https://github.com/Starbuster2004/DeepDetective.git">GitHub repository</a>.</b>
</div> 