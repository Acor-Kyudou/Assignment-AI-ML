# Assignment-AI-ML
Scouts Applied ML Engineer position Assignment
This repository contains the deployment-ready code, optimized YOLOv8 weights, raw training artifacts, and technical documentation for a real-time kinetic threat detection system. 

Unlike standard out-of-the-box object detectors, this pipeline implements a custom mathematical heuristic engine that combines algorithmic confidence with geometric spatial awareness to calculate a dynamic Threat Score.

## Repository Contents

### Primary Deliverables
* **`Test.py`**: The core execution script for real-time webcam inference and heuristic scoring.
* **`onnx_exp.py`**: The script to convert .pt model to .onnx format.
* **`Assignment Report-M.R.S.C.Karunarathne.pdf`**: The comprehensive technical diagnostic, architectural write-up, and methodology breakdown.
* **`best.pt` and `best.onnx`**: The custom-trained YOLOv8 weights (optimized via AdamW and Cosine Annealing).
* **`requirements.txt`**: The strict dependency map.

### Training & Diagnostic Artifacts (Supplementary)
* **`Weapon Detection with YOLO.ipynb`**: The raw Kaggle Jupyter notebook containing the end-to-end data pipeline, hyperparameter tuning, and offline offline evaluation logic.
* **`Results`**: The raw Ultralytics training directory containing all epoch logs, precision-recall curves, loss graphs, and confusion matrices referenced in the report along with Auto-generated screenshots during live webcam execution.
---

## Systematic Setup & Execution Guide

Follow these steps precisely to replicate the environment and run the live inference engine.

### Step 1: Environment Isolation
**What to do:** Open your terminal in this project folder and create a virtual environment.
* **Command:** `python -m venv venv`
* **Activate (Windows):** `.\venv\Scripts\activate`
* **Activate (Mac/Linux):** `source venv/bin/activate`

### Step 2: Install Dependencies
**What to do:** With the virtual environment active, install the required packages.
* **Command:** `pip install -r requirements.txt`

### Step 3: Launch the Threat Engine
**What to do:** Ensure `best.pt` is in the same folder as the script, then execute the main Python file.
* **Command:** `python Test.py`

### Step 4: Interact and Evaluate
**What to do:** Hold a test object (ex: a toy weapon, scissors, or a phone) up to the webcam. Watch the terminal output and the live UI. Press `q` on your keyboard to terminate the script. 

---

## Known Issues & Architectural Limitations

As detailed in Section 7 of the Project Report, reviewers should be aware of the following system constraints:

1. **Hardware/Latency Blocking:** The current `Test.py` script utilizes synchronous blocking—meaning `cap.read()` runs in the exact same thread as the YOLO inference. On consumer hardware, this CPU/GPU bottleneck may result in dropped frames. In a true production environment, frame ingestion, model inference, and UI rendering would be decoupled into asynchronous daemon threads to ensure uninterrupted camera streaming.

2. **Behavioral Blind Spots:** The current heuristic signals (Confidence, Scale, Proximity) do not encode the spatial relationship between a weapon and a human entity. For example, a knife resting unattended on a desk close to the camera can trigger a CRITICAL alert solely based on proximity and scale, while a handgun held further back might only score ELEVATED. Future iterations require a secondary behavioral-mapping signal to distinguish between "resting" and "brandished."
