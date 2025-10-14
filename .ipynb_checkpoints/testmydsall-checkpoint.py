# Grounding DINO Bulk Inference Script (Refactored)
# Fully synchronized logging ↔ filtering ↔ visualization

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

def filter_detections(boxes, logits, phrases, iou_threshold=0.5):
    nms_indices = apply_nms(boxes, logits, iou_threshold)
    boxes = boxes[nms_indices]
    logits = logits[nms_indices]
    phrases = [phrases[i] for i in nms_indices]

    keep_indices = remove_contained_boxes(boxes, phrases)
    boxes = boxes[keep_indices]
    logits = logits[keep_indices]
    phrases = [phrases[i] for i in keep_indices]

    return boxes, logits, phrases


def remove_contained_boxes(boxes, phrases):
    keep = []
    for i in range(len(boxes)):
        box_i = boxes[i]
        contained = False
        for j in range(len(boxes)):
            if i != j and phrases[i] == phrases[j]:
                box_j = boxes[j]
                if (
                    box_i[0] >= box_j[0] and box_i[1] >= box_j[1] and
                    box_i[2] <= box_j[2] and box_i[3] <= box_j[3]
                ):
                    contained = True
                    break
        if not contained:
            keep.append(i)
    return keep

def apply_nms(boxes, logits, iou_threshold=0.5):
    keep = ops.nms(boxes, logits, iou_threshold)
    return keep

def print_detection_summary(phrases, logits, boxes, stage_name, top_k=5):
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
    plt.figure(figsize=(12, 8))
    plt.imshow(image_source)
    if boxes.shape[0] > 0:
        h, w = image_source.shape[:2]
        # if boxes already in pixel space, do not rescale
        print("Before scale:", boxes.min().item(), boxes.max().item())

        if boxes.max() <= 1.5:  # probably normalized
            boxes_xyxy = box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
            boxes = torch.clamp(boxes_xyxy, 0, 1) * torch.tensor([w, h, w, h], device=boxes.device)

        # Apply NMS to remove overlapping boxes
        nms_indices = apply_nms(boxes, logits, iou_threshold=0.5)
        boxes = boxes[nms_indices]
        logits = logits[nms_indices]
        phrases = [phrases[i] for i in nms_indices]

        # Remove contained boxes before drawing
        keep_indices = remove_contained_boxes(boxes, phrases)
        boxes = boxes[keep_indices]
        logits = logits[keep_indices]
        phrases = [phrases[i] for i in keep_indices]

        if top_n:
            indices = torch.argsort(logits, descending=True)[:top_n]
        else:
            indices = range(len(phrases))
        color_map = {'crack': 'red', 'delamination': 'blue', 'no defect': 'green'}

        top_conf_idx = torch.argmax(logits)

        for i in indices:
            box = boxes[i].int().cpu().numpy()
            color = color_map.get(phrases[i].lower(), 'yellow')
            lw = 2
        
            # ✅ Highlight the most confident box
            if i == top_conf_idx:
                color = 'magenta'  # or 'cyan', 'black', etc.
                lw = 2  # thicker border
        
            plt.gca().add_patch(plt.Rectangle(
                (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                fill=False, edgecolor=color, linewidth=lw))
            plt.text(box[0], box[1], f"{phrases[i]}: {logits[i]:.2f}",
                     fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
                      
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    #plt.show()

def process_single_image(model, image_path, text_prompt, thresholds):
    img_rgb, img_tensor = load_image(image_path)
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        boxes, logits, phrases = predict(
            model, img_tensor, text_prompt,
            box_threshold=thresholds['box'], text_threshold=thresholds['text']
        )
    phrases = [p.lower().strip() for p in phrases]
    return img_rgb, boxes, logits, phrases

def process_bulk_folders(
    model_config,
    model_weights,
    folders,
    output_base_dir,
    text_prompt="Crack . Delamination .",
    thresholds={'box': 0.35, 'text': 0.25},
    top_n=3,
    auto_save=True
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_config, model_weights).to(device).eval()
    os.makedirs(output_base_dir, exist_ok=True)
    metrics = defaultdict(list)

    for folder in folders:
        print(f"\n📂 Folder: {folder}")
        out_dir = os.path.join(output_base_dir, os.path.basename(folder))
        os.makedirs(out_dir, exist_ok=True)

        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")): continue
            fpath = os.path.join(folder, fname)
            print(f"  • {fname}")
            try:
                img_rgb, boxes, logits, phrases = process_single_image(
                    model, fpath, text_prompt, thresholds
                )
                boxes, logits, phrases = filter_detections(boxes, logits, phrases)
                print_detection_summary(phrases, logits, boxes, "Detection Summary", top_k=top_n)

                h, w = img_rgb.shape[:2]
                boxes_xyxy = box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
                boxes_pixel = torch.clamp(boxes_xyxy, 0, 1) * torch.tensor([w, h, w, h], device=boxes.device)


                # Save annotated image and visualization
                annotated = annotate(img_rgb, boxes_pixel, logits, phrases)
                # Show visualization first
                #visualize_results(img_rgb, boxes_pixel, phrases, logits, top_n=1)
                
               # user_input = input(f"Save result for {fname}? (y/n): ").strip().lower()
                # if user_input == 'y':
                #     result_path = os.path.join(out_dir, fname.replace(".png", "_result.jpg"))
                #     viz_path = os.path.join(out_dir, fname.replace(".png", "_viz.png"))
                #     cv2.imwrite(result_path, annotated)
                #     visualize_results(img_rgb, boxes, phrases, logits, save_path=viz_path, top_n=top_n)
                #     print(f"✅ Saved result to: {result_path}")
                # else:
                #     print("❌ Skipped saving.")

                folder_tag = os.path.basename(folder)
                base_name = os.path.splitext(fname)[0]  # removes .png/.jpg
                result_path = os.path.join(out_dir, f"{folder_tag}_{base_name}_result.jpg")
                viz_path = os.path.join(out_dir, f"{folder_tag}_{base_name}_viz.png")

                cv2.imwrite(result_path, annotated)
                visualize_results(img_rgb, boxes_pixel, phrases, logits, save_path=viz_path, top_n=1)
                print(f"✅ Saved result to: {result_path}")


                metrics["image"].append(fname)
                metrics["folder"].append(os.path.basename(folder))
                metrics["detections"].append(len(phrases))
                metrics["mean_conf"].append(float(logits.mean()) if logits.numel() > 0 else 0)

            except Exception as e:
                print(f"    ⚠️ Error in {fname}: {e}")

    pd.DataFrame(metrics).to_csv(os.path.join(output_base_dir, "metrics.csv"), index=False)
    print("\n✅ Done. Summary CSV saved.")

if __name__ == "__main__":
    process_bulk_folders(
        model_config="D:/Deep_Learning/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        model_weights="D:/Deep_Learning/Grounding-Dino-FineTuning/weights_myds_1/epoch_55.pth",
        folders=[
            "D:/NF/C",
            "D:/NF/D",
            "D:/NF/ND"
        ],
        output_base_dir="D:/Deep_Learning/Grounding-Dino-FineTuning/bulk_results",
        text_prompt="Crack . Delamination ."
    )
