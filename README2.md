
# Grounding DINO Fine-Tuning 🦖

This repository builds upon the original work by  
[**IDEA-Research/GroundingDINO**](https://github.com/IDEA-Research/GroundingDINO)  
and the open-source implementation by  
[**Techwolf (Gitee)**](https://gitee.com/techwolf/Grounding-Dino-FineTuning).  

---

### 🧩 Additional Info
This implementation introduces the capability to **train the model with image-to-text grounding** — a crucial feature in applications where textual descriptions must align with image regions.  
For instance, when the model is given a caption *"a cat on the sofa"*, it should be able to localize both the *"cat"* and the *"sofa"* in the image.

---

### 🧠 Author’s Note
I have **extended and customized** this implementation for my own experiments on **fine-tuning, evaluation, and visualization** of the Grounding DINO model on a custom dataset.  
All my scripts are saved in the **`myscripts/`** folder.

In addition, since many developers face environment setup issues, I have included my working **Conda environment file** (`conda_env_file.yml`) to make replication and setup easier.

---

## ✨ Features

- **Fine-tuning DINO** – allows you to fine-tune DINO on your own dataset.  
- **Bounding Box Regression** – uses Generalized IoU and Smooth L1 loss for improved bounding box prediction.  
- **Position-Aware Logit Losses** – enables the model to learn both object categories and positional context.  
- **Phrase-Based NMS** – removes redundant boxes of the same object.

---

## ⚙️ Installation
Follow the installation steps from the [original GroundingDINO repository](https://github.com/IDEA-Research/GroundingDINO).  
Ensure all prerequisites are installed before running training or testing.

---

## 🧩 Training

1. Prepare your dataset with images and associated textual captions.  
   A small demo dataset (`multimodal-data/`) demonstrates the expected format.  
2. Run the training script:
   ```bash
   python train.py
