# Palm Fruit Quality Detection and Counting using YOLO

This project is a simple video inference system that detects, classifies, tracks, and counts fruits on a conveyor belt using a YOLO model.

---

## 🖥️ Environment
- OS: **Windows 11**
- Python: **3.8.18**
- CUDA: **11.2** (GPU Memory: 4GB)
- RAM: **16GB**

---

## 📦 Installation

### 1. Using `venv` (Python virtual environment)
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

### 2. Using `conda` (Anaconda or Miniconda)
```bash
conda create -n yolovideo python=3.8.18
conda activate yolovideo
pip install -r requirements.txt
```

---

## 🚀 How to Run

Example command to run the inference:

```bash
python .\main.py --model YOUR_MODEL_PATH --input YOUR_VIDEO_INPUT_PATH --xyxy YOUR_LINE_COORDINATES
```

### Arguments Explanation:
- `--model`: **(str)** Path to the YOLO model file.  
  Example: `model/22K-5-M.pt`
  
- `--input`: **(str)** Path to the input video file.  
  Example: `video/conveyor.mp4`
  
- `--xyxy`: **(str)** Coordinates for the counting line in the format `[x1,y1,x2,y2]`.  
  Example: `"[117,650,1771,650]"`

---

## 🎯 Task Requirements
- Perform inference every **30 frames** (~1 FPS if video is 30 FPS).
- Relabel detected objects based on fruit quality:
  - `0 = Ripe`
  - `1 = Unripe`
  - `2 = OverRipe`
  - `3 = Rotten`
  - `4 = EmptyBunch`
- Track each detected object **until** it crosses the **counting line**.

---

## 🖼️ Example Result
Result sample (animated GIF):

![Inference Video](result/output.gif)

> _**Note:** You can find the result GIF or video output in the `result/` folder._

---

## 🧠 Code Explanation

This code is a program to run a YOLO model detection inference process with bytetrack tracking method to track movement of a palm fruit on a conveyor and also count how many palm fruit detected that passed through line counter for each class. For the detection inference requirement part, it will become a future improvement if possible. The reason is because it's imposible to maintaining track id of an object when doing some frame skipping without using more complex solution such as ReID model or using high end track method (DeepSort etc.)

---
