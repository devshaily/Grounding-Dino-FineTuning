from datetime import datetime 
from groundingdino.util.train import load_model, load_image, train_image, annotate
import cv2
import os
import json
import csv
import torch
from collections import defaultdict
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd 

# Model
model = load_model("D:/Deep_Learning/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", 
                  "D:/Deep_Learning/GroundingDINO/weights/groundingdino_swint_ogc.pth")

# Dataset paths
train_images_dir = "D:/Deep_Learning/Dataset/MRCNN/train/images"
train_ann_file = "D:/Deep_Learning/Dataset/MRCNN/train/annotations_final1.csv"
val_images_dir = "D:/Deep_Learning/Dataset/MRCNN/val/images"  # Add validation path
val_ann_file = "D:/Deep_Learning/Dataset/MRCNN/val/annotations_final1.csv"  # Add validation path

def draw_box_with_label(image, output_path, coordinates, label, color=(0, 0, 255), thickness=2, font_scale=0.5):
    """
    Draw a box and a label on an image using OpenCV.

    Parameters:
    - image (numpyarray): input image.
    - output_path (str): Path to save the image with the box and label.
    - coordinates (tuple): A tuple (x1, y1, x2, y2) indicating the top-left and bottom-right corners of the box.
    - label (str): The label text to be drawn next to the box.
    - color (tuple, optional): Color of the box and label in BGR format. Default is red (0, 0, 255).
    - thickness (int, optional): Thickness of the box's border. Default is 2 pixels.
    - font_scale (float, optional): Font scale for the label. Default is 0.5.
    """
    
    # Draw the rectangle
    cv2.rectangle(image, (coordinates[0], coordinates[1]), (coordinates[2], coordinates[3]), color, thickness)
    
    # Define a position for the label (just above the top-left corner of the rectangle)
    label_position = (coordinates[0], coordinates[1]-10)
    
    # Draw the label
    cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    
    # Save the modified image
    cv2.imwrite(output_path, image)

def read_dataset(ann_file):
    ann_Dict = defaultdict(lambda: defaultdict(list))
    zero_bbox_count = 0
    valid_bbox_count = 0
    
    # Determine if we're processing train or val data
    if 'train' in ann_file.lower():
        base_dir = "D:/Deep_Learning/Dataset/MRCNN/train/images"
    else:
        base_dir = "D:/Deep_Learning/Dataset/MRCNN/val/images"
    
    with open(ann_file) as file_obj:
        rows = [row for row in csv.DictReader(file_obj) 
               if not any(x in str(row.values()).lower() 
                         for x in ['sum', 'categories'])]
        
        for row in rows:
            try:
                img_name = row['image_name'].strip()
                img_path = os.path.join(base_dir, img_name)
                
                # Clean and convert bbox coordinates
                x1 = int(float(row['bbox_x1'].replace(',', '')))
                y1 = int(float(row['bbox_y1'].replace(',', '')))
                x2 = int(float(row['bbox_x2'].replace(',', '')))
                y2 = int(float(row['bbox_y2'].replace(',', '')))
                label = row['label_name'].strip()
                
                # Skip zero-area boxes
                if x1 == x2 or y1 == y2:
                    zero_bbox_count += 1
                    continue
                
                # Validate coordinates
                img_w = int(float(row.get('image_width', '2048').replace(',', '')))
                img_h = int(float(row.get('image_height', '2046').replace(',', '')))
                
                if (x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h):
                    print(f"Out-of-bounds bbox in {img_name}: ({x1},{y1},{x2},{y2})")
                    continue
                
                valid_bbox_count += 1
                ann_Dict[img_path]['boxes'].append([x1, y1, x2, y2])
                ann_Dict[img_path]['captions'].append(label)
                
            except (KeyError, ValueError) as e:
                print(f"Skipping {row.get('image_name','unknown')}: {str(e)}")
                continue
    
    print(f"\nDataset Summary:")
    print(f"- Valid images: {len(ann_Dict)}")
    print(f"- Valid bounding boxes: {valid_bbox_count}")
    print(f"- Skipped zero-area boxes: {zero_bbox_count}")
    
    return ann_Dict
    
def visualize_annotations(ann_Dict, n_per_class=2):
    """Visualize samples from each class with bounding boxes"""
    try:
        class_samples = defaultdict(list)
        for img_path, anns in ann_Dict.items():
            if anns['captions']:  # Only if captions exist
                class_name = anns['captions'][0]
                class_samples[class_name].append(img_path)
        
        for class_name, img_paths in class_samples.items():
            print(f"\nVisualizing {class_name} samples:")
            for img_path in img_paths[:n_per_class]:
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Couldn't read {img_path}")
                        continue
                        
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
                    
                    # Draw all boxes
                    for box in ann_Dict[img_path]['boxes']:
                        cv2.rectangle(img, 
                                     (box[0], box[1]), 
                                     (box[2], box[3]), 
                                     (0, 255, 0), 2)
                    
                    plt.figure(figsize=(12, 8))
                    plt.imshow(img)
                    plt.title(f"{class_name}\n{os.path.basename(img_path)}")
                    plt.axis('off')
                    plt.show()
                    
                except Exception as e:
                    print(f"Error visualizing {img_path}: {str(e)}")
                    
    except Exception as e:
        print(f"Visualization error: {str(e)}")

def train(model, train_ann_file, val_ann_file=None, epochs=1000, 
          save_path='./weights_myds', save_epoch=5, early_stop_patience=15,
          resume_from=None):
    
    os.makedirs(save_path, exist_ok=True)
    log_file = os.path.join(save_path, "training_log.csv")
    
    # Initialize logging
    # with open(log_file, 'w') as f:
    #     f.write("epoch,train_loss,val_loss,lr,early_stop_counter,timestamp\n")
    # Initialize logging safely (append if file exists, write header only if new)
    # Logging Setup — append if resuming, else fresh
    log_header = "epoch,train_loss,val_loss,lr,early_stop_counter,timestamp\n"
    
    # If resuming, don't erase the log
    if not os.path.exists(log_file) or resume_from is None:
        with open(log_file, 'w') as f:
            f.write(log_header)
    else:
        print(f"✅ Resuming from checkpoint — will append to existing log.")


    
    # Data loading
    train_ann_Dict = read_dataset(train_ann_file)
    val_ann_Dict = read_dataset(val_ann_file) if val_ann_file else None
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Tracking
    start_epoch = 0
    best_train_loss = float('inf')  # Track training loss instead of val
    early_stop_counter = 0

    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            print(f"Resumed from checkpoint at epoch {start_epoch}")
        else:
            model.load_state_dict(checkpoint)
            start_epoch = int(resume_from.split("_")[-1].split(".")[0])
            print(f"Resumed from raw weights at epoch {start_epoch}")
    
    for epoch in range(start_epoch, epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        processed_samples = 0
        
        with tqdm(train_ann_Dict.items(), desc=f"Train Epoch {epoch+1}") as pbar:
            for IMAGE_PATH, vals in pbar:
                try:
                    image_source, image = load_image(IMAGE_PATH)
                    image = image.to(device)
                    boxes = torch.tensor(vals['boxes'], dtype=torch.float32).to(device)
                    
                    optimizer.zero_grad()
                    loss = train_image(
                        model=model,
                        image_source=image_source,
                        image=image,
                        caption_objects=vals['captions'],
                        box_target=boxes
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    processed_samples += 1
                    pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                    
                except Exception as e:
                    print(f"\nSkipped {IMAGE_PATH}: {str(e)}")
                    continue
        
        # Calculate epoch metrics
        avg_train_loss = epoch_loss / processed_samples if processed_samples > 0 else float('inf')
        scheduler.step()
        
        # Early stopping based on TRAIN LOSS
        if avg_train_loss < best_train_loss - 0.001:  # Threshold to avoid noise
            best_train_loss = avg_train_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
        else:
            early_stop_counter += 1
        
        # Optional validation (for logging only)
        avg_val_loss = float('inf')
        if val_ann_Dict:
            avg_val_loss = validate_model(model, val_ann_Dict, device)
        
        # Log all metrics
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{avg_train_loss:.6f},{avg_val_loss:.6f},{optimizer.param_groups[0]['lr']:.2e},{early_stop_counter},{current_time}\n")
        
        # Epoch summary
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"- Train Loss: {avg_train_loss:.4f} (Best: {best_train_loss:.4f})")
        if val_ann_Dict:
            print(f"- Val Loss: {avg_val_loss:.4f} (Monitoring Only)")
        print(f"- Early Stop Counter: {early_stop_counter}/{early_stop_patience}")
        print(f"- Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save weights periodically
        if (epoch+1) % save_epoch == 0 or (epoch+1) == 1:
            torch.save(model.state_dict(), os.path.join(save_path, f"epoch_{epoch+1}.pth"))
        
        # Early stopping
        if early_stop_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1} (no train loss improvement)!")
            break
    
    # Final save
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_train_loss,
    }, os.path.join(save_path, "final_model.pth"))
    
    plot_training_history(log_file, save_path)

def plot_training_history(log_file, save_path):
    """Generate plots from log file"""
    df = pd.read_csv(log_file)
    df = df.drop_duplicates(subset="epoch", keep="last").sort_values("epoch")

    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    if 'val_loss' in df.columns:
        plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(df['epoch'], df['lr'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    
    plt.subplot(1, 3, 3)
    if 'val_loss' in df.columns:
        plt.plot(df['epoch'], df['early_stop_counter'])
        plt.xlabel('Epoch')
        plt.ylabel('Early Stop Counter')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "training_history.png"))
    plt.close()

def validate_model(model, val_ann_Dict, device):
    """Simplified validation"""
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for val_path, vals in val_ann_Dict.items():
            try:
                val_source, val_img = load_image(val_path)
                val_loss += train_image(
                    model=model,
                    image_source=val_source,
                    image=val_img.to(device),
                    caption_objects=vals['captions'],
                    box_target=torch.tensor(vals['boxes'], dtype=torch.float32).to(device)
                ).item()
            except Exception as e:
                print(f"Skipped validation {val_path}: {str(e)}")
    return val_loss / len(val_ann_Dict)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    resume_checkpoint = 'D:/Deep_Learning/Grounding-Dino-FineTuning/weights_myds/epoch_54.pth'

    
    # Add these diagnostic prints
    print("\n=== Device Information ===")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    else:
        print("No GPU available, using CPU")
    print("=========================\n")
    train(model=model, 
      train_ann_file=train_ann_file,
      val_ann_file=val_ann_file,
      epochs=1000, 
      save_path='D:/Deep_Learning/Grounding-Dino-FineTuning/weights_myds',
      save_epoch=5,
      early_stop_patience=15,
      resume_from=resume_checkpoint)
