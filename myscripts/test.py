# Grounding DINO Bulk Inference Script (Refactored)
# 🔄 Fully synchronized logging ↔ filtering ↔ visualization

from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import torch
import torchvision.ops as ops
from torchvision.ops import box_convert
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import numpy as np
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.models import build_model
from datetime import datetime
from tqdm import tqdm

def filter_detections(boxes, logits, phrases, iou_threshold=0.5):
    if boxes.shape[0] == 0:
        return boxes, logits, phrases

    boxes_xyxy = box_convert(boxes, 'cxcywh', 'xyxy')
    
    # Class-specific thresholds
    class_thresholds = {
        'crack': 0.4,      # Higher threshold for cracks
        'delamination': 0.4,
        'no defect': 0.5    # Highest threshold to ignore "no defect"
    }

    # Filter by class thresholds
    valid_indices = [
        i for i, (phrase, logit) in enumerate(zip(phrases, logits))
        if logit > class_thresholds.get(phrase.lower().strip(), 0.3)
    ]
    
    boxes_xyxy = boxes_xyxy[valid_indices]
    logits = logits[valid_indices]
    phrases = [phrases[i] for i in valid_indices]

    # Apply NMS per class
    unique_classes = set(p.lower().strip() for p in phrases)
    final_boxes, final_logits, final_phrases = [], [], []

    for cls in unique_classes:
        cls_indices = [i for i, p in enumerate(phrases) if p.lower().strip() == cls]
        cls_boxes = boxes_xyxy[cls_indices]
        cls_logits = logits[cls_indices]

        keep = ops.nms(cls_boxes, cls_logits, iou_threshold)
        final_boxes.append(cls_boxes[keep])
        final_logits.append(cls_logits[keep])
        final_phrases.extend([phrases[cls_indices[k]] for k in keep])

    if final_boxes:
        return torch.cat(final_boxes), torch.cat(final_logits), final_phrases
    return boxes_xyxy, logits, phrases

def remove_contained_boxes(boxes, phrases):
    """Remove boxes completely contained within others of same class"""
    keep = []
    for i in range(len(boxes)):
        box_i = boxes[i]
        contained = False
        for j in range(len(boxes)):
            if i != j and phrases[i] == phrases[j]:
                box_j = boxes[j]
                if (box_i[0] >= box_j[0] and box_i[1] >= box_j[1] and
                    box_i[2] <= box_j[2] and box_i[3] <= box_j[3]):
                    contained = True
                    break
        if not contained:
            keep.append(i)
    return keep

def print_detection_summary(phrases, logits, boxes, stage_name, top_k=5):
    """Print formatted detection summary"""
    print(f"\n=== {stage_name} ===")
    print(f"Total detections: {len(phrases)}")
    
    class_stats = defaultdict(lambda: {'count': 0, 'conf_sum': 0})
    for phrase, logit in zip(phrases, logits):
        cls = phrase.lower().strip()
        class_stats[cls]['count'] += 1
        class_stats[cls]['conf_sum'] += logit
        
    for cls, stats in class_stats.items():
        avg_conf = stats['conf_sum'] / stats['count'] if stats['count'] > 0 else 0
        print(f"  {cls}: {stats['count']} boxes | Avg confidence: {avg_conf:.4f}")
        
    if boxes.shape[0] > 0:
        print("📊 Top boxes:")
        sorted_indices = torch.argsort(logits, descending=True)
        for i in sorted_indices[:min(top_k, len(phrases))]:
            box = boxes[i].tolist()
            print(f"{phrases[i].upper():<12} | Box: {np.round(box, 2)} | Confidence: {logits[i]:.4f}")

def visualize_results(image_source, boxes, phrases, logits, save_path=None, top_n=None):
    """Visualize detection results with matplotlib"""
    plt.figure(figsize=(12, 8))
    plt.imshow(image_source)
    
    if boxes.shape[0] > 0:
        h, w = image_source.shape[:2]
        
        # Convert boxes if normalized
        if boxes.max() <= 1.5:  # probably normalized
            boxes_xyxy = box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
            boxes = torch.clamp(boxes_xyxy, 0, 1) * torch.tensor([w, h, w, h], device=boxes.device)

        # Color mapping
        color_map = {
            'crack': 'red', 
            'delamination': 'blue', 
            'no defect': 'green',
            'default': 'yellow'
        }

        # Highlight top confidence box
        if len(logits) > 0:
            top_conf_idx = torch.argmax(logits)
            
        for i in range(len(phrases)):
            if top_n and i >= top_n:
                break
                
            box = boxes[i].int().cpu().numpy()
            color = color_map.get(phrases[i].lower(), color_map['default'])
            lw = 4 if i == top_conf_idx else 2  # Thicker border for top confidence
        
            plt.gca().add_patch(plt.Rectangle(
                (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                fill=False, edgecolor=color, linewidth=lw))
            plt.text(box[0], box[1], f"{phrases[i]}: {logits[i]:.2f}",
                    fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
                      
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

def process_single_image(model, image_path, text_prompt, thresholds):
    """Process single image through model"""
    img_rgb, img_tensor = load_image(image_path)
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        boxes, logits, phrases = predict(
            model, img_tensor, text_prompt,
            box_threshold=thresholds['box'], 
            text_threshold=thresholds['text']
        )
    phrases = [p.lower().strip() for p in phrases]
    return img_rgb, boxes, logits, phrases
    
def build_model_from_config(config_path, weights_path):
    """Load model from config and weights"""
    try:
        args = SLConfig.fromfile(config_path)
        model = build_model(args)
        
        checkpoint = torch.load(weights_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
            
        # Handle key mismatches
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def evaluate_epoch_models(
    model_config,
    weights_folder,
    input_folders,
    output_base_dir,
    text_prompt="Crack . Delamination .",
    thresholds={'box': 0.35, 'text': 0.25},
    top_n=3
):
    """Evaluate all epoch models in weights_folder against input_folders"""
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 Starting evaluation on {device.upper()}")
    
    # Find all epoch weights files
    epoch_files = sorted([
        f for f in os.listdir(weights_folder) 
        if f.startswith('epoch_') and f.endswith('.pth')
    ])
    
    if not epoch_files:
        raise ValueError(f"No epoch weights found in {weights_folder}")
    
    print(f"Found {len(epoch_files)} epoch models to evaluate")
    
    # Prepare results dataframe
    results = []
    folder_to_gt = {
        'C': 'crack',
        'D': 'delamination',
        'ND': 'no defect'
    }
    
    # Process each epoch model
    for epoch_file in tqdm(epoch_files, desc="Evaluating epochs"):
        epoch_num = epoch_file.split('_')[1].split('.')[0]
        model_path = os.path.join(weights_folder, epoch_file)
        
        try:
            model = build_model_from_config(model_config, model_path).to(device).eval()
        except Exception as e:
            print(f"⚠️ Error loading {epoch_file}: {e}")
            continue
            
        # Process each folder
        for folder in input_folders:
            folder_name = os.path.basename(folder)
            gt_class = folder_to_gt.get(folder_name, 'unknown')
            
            # Process each image
            for fname in os.listdir(folder):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                    
                fpath = os.path.join(folder, fname)
                
                try:
                    # Run inference
                    _, boxes, logits, phrases = process_single_image(
                        model, fpath, text_prompt, thresholds
                    )
                    
                    # Filter detections
                    boxes, logits, phrases = filter_detections(boxes, logits, phrases)
                    
                    # Determine predicted class (highest confidence detection)
                    if len(phrases) > 0:
                        pred_class = phrases[torch.argmax(logits)].lower()
                    else:
                        pred_class = 'no detection'
                    
                    # Record result
                    results.append({
                        'epoch': epoch_num,
                        'image': fname,
                        'source_folder': folder_name,
                        'ground_truth': gt_class,
                        'predicted': pred_class,
                        'is_correct': pred_class == gt_class,
                        'num_detections': len(phrases),
                        'max_confidence': float(logits.max()) if len(logits) > 0 else 0,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                except Exception as e:
                    print(f"Error processing {fname} with {epoch_file}: {e}")
                    continue
    
    # Create dataframe and calculate metrics
    df = pd.DataFrame(results)
    
    # Calculate epoch-wise metrics
    epoch_metrics = df.groupby('epoch').agg({
        'is_correct': ['mean', 'sum', 'count'],
        'max_confidence': 'mean'
    }).reset_index()
    
    epoch_metrics.columns = ['epoch', 'accuracy', 'correct_count', 'total_images', 'mean_confidence']
    
    # Save results
    os.makedirs(output_base_dir, exist_ok=True)
    results_csv = os.path.join(output_base_dir, "epoch_evaluation_results.csv")
    metrics_csv = os.path.join(output_base_dir, "epoch_metrics_summary.csv")
    
    df.to_csv(results_csv, index=False)
    epoch_metrics.to_csv(metrics_csv, index=False)
    
    print(f"\n🎉 Evaluation complete!")
    print(f"Detailed results saved to: {results_csv}")
    print(f"Epoch metrics saved to: {metrics_csv}")
    
    # Find best epoch
    best_epoch = epoch_metrics.loc[epoch_metrics['accuracy'].idxmax()]
    print(f"\n🏆 Best epoch: {best_epoch['epoch']}")
    print(f"  Accuracy: {best_epoch['accuracy']:.2%}")
    print(f"  Correct predictions: {best_epoch['correct_count']}/{best_epoch['total_images']}")
    print(f"  Mean confidence: {best_epoch['mean_confidence']:.4f}")
    
    return df, epoch_metrics



if __name__ == "__main__":
    # Configuration
    CONFIG = "D:/Deep_Learning/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    WEIGHTS_FOLDER = "D:/Deep_Learning/Grounding-Dino-FineTuning/weights_myds"
    INPUT_FOLDERS = [
        "D:/NF/C",
        "D:/NF/D", 
        "D:/NF/ND"
    ]
    OUTPUT_DIR = "D:/Deep_Learning/Grounding-Dino-FineTuning/epoch_evaluation"
    
    # Run evaluation
    evaluate_epoch_models(
        model_config=CONFIG,
        weights_folder=WEIGHTS_FOLDER,
        input_folders=INPUT_FOLDERS,
        output_base_dir=OUTPUT_DIR,
        text_prompt="Crack . Delamination .",
        thresholds={'box': 0.35, 'text': 0.25},
        top_n=3
    )