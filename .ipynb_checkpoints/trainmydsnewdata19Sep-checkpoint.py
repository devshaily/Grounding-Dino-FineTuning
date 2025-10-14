from datetime import datetime
from groundingdino.util.train import load_model, load_image, train_image
import cv2
import os
import torch
from collections import defaultdict
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# =========================
# CONFIG
# =========================
CONFIG_PATH = "D:/Deep_Learning/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
PRETRAINED_PATH = "D:/Deep_Learning/GroundingDINO/weights/groundingdino_swint_ogc.pth"   # official base weights
OLD_MODEL_PATH = "D:/Deep_Learning/Grounding-Dino-FineTuning/weights_myds/epoch_100.pth" # your fine-tuned weights
NEW_SAVE_PATH = "D:/Deep_Learning/GroundingDino_Finetune_NewDS"  # new folder for saving

train_images_dir = r"D:\Deep_Learning\Dataset\NewData19Sep\images"
train_labels_dir = r"D:\Deep_Learning\Dataset\NewData19Sep\labels"

os.makedirs(NEW_SAVE_PATH, exist_ok=True)

# =========================
# LOAD MODEL
# =========================
print("🔄 Loading base architecture...")
model = load_model(CONFIG_PATH, PRETRAINED_PATH)

print("🔄 Loading fine-tuned weights...")
state_dict = torch.load(OLD_MODEL_PATH, map_location="cpu")
if "model_state_dict" in state_dict:  # if checkpoint dict
    model.load_state_dict(state_dict["model_state_dict"], strict=False)
else:  # plain state dict
    model.load_state_dict(state_dict, strict=False)
print(f"✅ Loaded fine-tuned model from {OLD_MODEL_PATH}")

# =========================
# YOLO Dataset Reader
# =========================
# Class mapping from your YAML
CLASS_MAP = {
    0: "Delamination",
    1: "Crack",
    2: "NoDefect"
}

def read_yolo_dataset(images_dir, labels_dir):
    ann_Dict = defaultdict(lambda: defaultdict(list))
    img_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for img_file in img_files:
        img_path = os.path.join(images_dir, img_file)
        label_file = os.path.join(labels_dir, os.path.splitext(img_file)[0] + ".txt")

        if not os.path.exists(label_file):
            continue

        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            with open(label_file, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"⚠️ Skipping malformed line in {label_file}: {line.strip()}")
                        continue
                    try:
                        cls, x, y, bw, bh = map(float, parts)
                    except ValueError:
                        print(f"⚠️ Non-numeric entry in {label_file}: {line.strip()}")
                        continue

                    x1 = int((x - bw / 2) * w)
                    y1 = int((y - bh / 2) * h)
                    x2 = int((x + bw / 2) * w)
                    y2 = int((y + bh / 2) * h)

                    # Store bbox + human-readable label
                    ann_Dict[img_path]['boxes'].append([x1, y1, x2, y2])
                    ann_Dict[img_path]['captions'].append(CLASS_MAP.get(int(cls), str(int(cls))))

        except Exception as e:
            print(f"⚠️ Skipping {img_file}: {e}")
            continue

    print(f"✅ Loaded {len(ann_Dict)} images with annotations from {images_dir}")
    return ann_Dict


# =========================
# TRAINING FUNCTION
# =========================
def train(model, train_img_dir, train_lbl_dir,
          epochs=100, save_epoch=5, resume_from=None):

    log_file = os.path.join(NEW_SAVE_PATH, "training_log.csv")
    if not os.path.exists(log_file) or resume_from is None:
        with open(log_file, 'w') as f:
            f.write("epoch,train_loss,lr,timestamp\n")
    else:
        print(f"📌 Resuming — appending logs")

    train_ann_Dict = read_yolo_dataset(train_img_dir, train_lbl_dir)

    optimizer = optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    start_epoch = 0

    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            print(f"🔄 Resumed from checkpoint at epoch {start_epoch}")
        else:
            model.load_state_dict(checkpoint)
            print("⚠️ Resumed from raw weights only")

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss, processed_samples = 0, 0

        with tqdm(train_ann_Dict.items(), desc=f"Train Epoch {epoch+1}") as pbar:
            for img_path, vals in pbar:
                try:
                    image_source, image = load_image(img_path)
                    image = image.to(device)
                    boxes = torch.tensor(vals['boxes'], dtype=torch.float32).to(device)

                    optimizer.zero_grad()
                    loss = train_image(model=model,
                                       image_source=image_source,
                                       image=image,
                                       caption_objects=vals['captions'],
                                       box_target=boxes)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    processed_samples += 1
                    pbar.set_postfix({'loss': f"{loss.item():.4f}"})

                except Exception as e:
                    print(f"⚠️ Skipped {img_path}: {e}")
                    continue

        avg_train_loss = epoch_loss / processed_samples if processed_samples > 0 else float('inf')
        scheduler.step()

        # Logging
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{avg_train_loss:.6f},{optimizer.param_groups[0]['lr']:.2e},{datetime.now()}\n")

        print(f"\n📊 Epoch {epoch+1}/{epochs} | Train: {avg_train_loss:.4f}")

        # Save periodically
        if (epoch+1) % save_epoch == 0 or (epoch+1) == 1:
            torch.save(model.state_dict(), os.path.join(NEW_SAVE_PATH, f"epoch_{epoch+1}.pth"))

    # Final save
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_train_loss,
    }, os.path.join(NEW_SAVE_PATH, "final_model.pth"))

    plot_training_history(log_file, NEW_SAVE_PATH)

# =========================
# PLOT TRAINING HISTORY
# =========================
def plot_training_history(log_file, save_path):
    df = pd.read_csv(log_file)
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "training_history.png"))
    plt.close()

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("\n=== Device Info ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")

    train(model,
          train_images_dir, train_labels_dir,
          epochs=100,
          save_epoch=5,
          resume_from=None)  # set checkpoint path if resuming
