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
    
    with open(ann_file) as file_obj:
        # Skip summary rows (those containing 'sum' or 'categories')
        rows = [row for row in csv.DictReader(file_obj) 
               if not any(x in str(row.values()).lower() 
                         for x in ['sum', 'categories'])]
        
        for row in rows:
            try:
                img_name = row['image_name'].strip()
                img_path = os.path.join("D:/Deep_Learning/Dataset/MRCNN/train/images", img_name)
                
                # Clean and convert bbox coordinates
                x1 = int(float(row['bbox_x1'].replace(',', '')))
                y1 = int(float(row['bbox_y1'].replace(',', '')))
                x2 = int(float(row['bbox_x2'].replace(',', '')))
                y2 = int(float(row['bbox_y2'].replace(',', '')))
                label = row['label_name'].strip()
                
                # Skip zero-area boxes but count them
                if x1 == x2 or y1 == y2:
                    zero_bbox_count += 1
                    continue
                
                # Validate coordinates are within image bounds
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
        
def validate(model, val_ann_Dict, device):
    """Validation loop with confidence tracking"""
    model.eval()
    val_losses = []
    confidences = []
    
    with torch.no_grad():
        for IMAGE_PATH, vals in tqdm(val_ann_Dict.items(), desc="Validating"):
            try:
                image_source, image = load_image(IMAGE_PATH)
                image = image.to(device)
                
                bxs = torch.tensor(vals['boxes'], dtype=torch.float32).to(device)
                captions = vals['captions']

                # Get both loss and predictions
                loss, outputs = train_image(
                    model=model,
                    image_source=image_source,
                    image=image,
                    caption_objects=captions,
                    box_target=bxs,
                    return_outputs=True
                )
                
                if outputs is not None:
                    confidences.extend(outputs['logits'].cpu().tolist())
                
                val_losses.append(loss.item())
                
            except Exception as e:
                print(f"Validation error on {IMAGE_PATH}: {str(e)}")
                continue
    
    avg_loss = np.mean(val_losses) if val_losses else float('inf')
    avg_confidence = np.mean(confidences) if confidences else 0
    model.train()
    return avg_loss, avg_confidence

def train(model, train_ann_file, val_ann_file, epochs=1, 
          save_path='D:/Deep_Learning/Grounding-Dino-FineTuning/weights_myds',
          save_epoch=5, early_stop_patience=15):
    """Stable training implementation for GroundingDINO"""
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    
    # Read datasets
    train_ann_Dict = read_dataset(train_ann_file)
    val_ann_Dict = read_dataset(val_ann_file) if val_ann_file else None
    
    # Setup optimizer with lower initial LR
    optimizer = optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-4)
    
    # Simple LR scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Training tracking
    best_val_loss = float('inf')
    early_stop_counter = 0
    train_history = {'loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_ann_Dict.items(), desc=f"Epoch {epoch+1}/{epochs}")
        
        for idx, (IMAGE_PATH, vals) in enumerate(progress_bar):
            try:
                # Load data
                image_source, image = load_image(IMAGE_PATH)
                image = image.to(device)
                bxs = torch.tensor(vals['boxes'], dtype=torch.float32).to(device)
                captions = vals['captions']
                
                # Forward pass
                optimizer.zero_grad()
                loss = train_image(
                    model=model,
                    image_source=image_source,
                    image=image,
                    caption_objects=captions,
                    box_target=bxs
                )
                
                if torch.isnan(loss):
                    print(f"NaN loss on {IMAGE_PATH}")
                    continue
                
                # Backward pass with gradient clipping
                loss.backward()
                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=1.0
                )
                optimizer.step()
                
                # Track loss
                epoch_loss += loss.item()
                
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'grad': f"{total_grad_norm:.1f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.1e}"
                })
                
            except Exception as e:
                print(f"Error on {IMAGE_PATH}: {str(e)}")
                continue
        
        # Update LR scheduler
        scheduler.step()
        
        # Calculate epoch loss
        avg_loss = epoch_loss / len(train_ann_Dict)
        train_history['loss'].append(avg_loss)
        
        # Validation phase
        if val_ann_Dict:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_path, val_anns in val_ann_Dict.items():
                    try:
                        val_source, val_img = load_image(val_path)
                        val_img = val_img.to(device)
                        val_boxes = torch.tensor(val_anns['boxes'], dtype=torch.float32).to(device)
                        
                        val_loss += train_image(
                            model=model,
                            image_source=val_source,
                            image=val_img,
                            caption_objects=val_anns['captions'],
                            box_target=val_boxes
                        ).item()
                    except Exception as e:
                        print(f"Validation error on {val_path}: {str(e)}")
            
            avg_val_loss = val_loss / len(val_ann_Dict)
            train_history['val_loss'].append(avg_val_loss)
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"- Train Loss: {avg_loss:.4f}")
            print(f"- Val Loss: {avg_val_loss:.4f}")
            print(f"- LR: {optimizer.param_groups[0]['lr']:.1e}")
            
            # Early stopping check
            if avg_val_loss < best_val_loss - 0.001:
                best_val_loss = avg_val_loss
                early_stop_counter = 0
                torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
                print("New best model saved!")
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}!")
                    break
        else:
            print(f"\nEpoch {epoch+1} Train Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch+1) % save_epoch == 0:
            save_file = os.path.join(save_path, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_file)
    
    # Final save
    torch.save(model.state_dict(), os.path.join(save_path, "final_model.pth"))
    plot_training_curves(train_history, save_path)

def plot_training_curves(history, save_path):
    """Plot training metrics"""
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title("Loss Curves")
    
    plt.subplot(1, 2, 2)
    if 'grad_norm' in history:
        plt.plot(history['grad_norm'])
        plt.title("Gradient Norm")
    
    plt.savefig(os.path.join(save_path, "training_curves.png"))
    plt.close()
if __name__ == "__main__":
    train(model=model, 
          train_ann_file=train_ann_file,
          val_ann_file=val_ann_file,
          epochs=1000, 
          save_path='D:/Deep_Learning/Grounding-Dino-FineTuning/weights_myds',
          save_epoch=5,
          early_stop_patience=15)