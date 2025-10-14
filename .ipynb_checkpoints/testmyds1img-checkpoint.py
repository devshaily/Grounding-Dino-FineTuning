from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import torch
import torchvision.ops as ops
from torchvision.ops import box_convert
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import csv
from datetime import datetime
import pandas as pd
import numpy as np

def apply_nms_per_class(boxes, logits, phrases, nms_thresholds, max_boxes=5):
    """
    Apply NMS per class and return top max_boxes detections
    Returns boxes in xyxy format
    """
    if boxes.shape[0] == 0:
        return boxes, logits, phrases

    # Convert single threshold to dict if needed
    if isinstance(nms_thresholds, (float, int)):
        classes = set(p.lower().strip() for p in phrases)
        nms_thresholds = {cls: float(nms_thresholds) for cls in classes}

    cleaned_phrases = [p.lower().strip() for p in phrases]
    all_boxes = []
    all_logits = []
    all_phrases = []

    for cls, threshold in nms_thresholds.items():
        cls_indices = [i for i, p in enumerate(cleaned_phrases) if p == cls]
        if not cls_indices:
            continue

        # Convert to xyxy and filter by confidence
        cls_boxes = box_convert(boxes[cls_indices], 'cxcywh', 'xyxy')
        cls_logits = logits[cls_indices]
        
        # Sort by confidence and keep top candidates
        sorted_indices = torch.argsort(cls_logits, descending=True)
        top_indices = sorted_indices[:max_boxes*3]  # Keep more initially for NMS
        
        # Apply NMS on top candidates
        keep = ops.nms(cls_boxes[top_indices], cls_logits[top_indices], threshold)
        keep = keep[:max_boxes]  # Final limit
        
        all_boxes.append(cls_boxes[top_indices][keep])
        all_logits.append(cls_logits[top_indices][keep])
        all_phrases.extend([phrases[cls_indices[top_indices[i]]] for i in keep])

    if all_boxes:
        return torch.cat(all_boxes), torch.cat(all_logits), all_phrases
    return torch.empty((0, 4)), torch.empty((0,)), []

def visualize_results(image_source, boxes, phrases, logits, save_path=None):
    """Visualization with proper box scaling"""
    plt.figure(figsize=(12, 8))
    plt.imshow(image_source)
    
    if boxes.shape[0] > 0:
        h, w = image_source.shape[:2]
        
        # Clip boxes to [0,1] range before scaling
        boxes = torch.clamp(boxes, 0, 1)
        boxes = boxes * torch.tensor([w, h, w, h], device=boxes.device)
        
        # Color coding
        color_map = {'crack': 'red', 'delamination': 'blue', 'no defect': 'green'}
        
        for box, phrase, logit in zip(boxes, phrases, logits):
            box = box.int().cpu().numpy()
            plt.gca().add_patch(plt.Rectangle(
                (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                fill=False, edgecolor=color_map.get(phrase.lower(), 'yellow'), linewidth=2))
            plt.text(box[0], box[1], f"{phrase}: {logit:.2f}",
                   fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def debug_iou_heatmap(boxes, title=""):
    """Helper to visualize box overlaps"""
    if boxes.shape[0] == 0:
        return
        
    iou_matrix = ops.box_iou(boxes, boxes)
    plt.figure(figsize=(10, 8))
    plt.imshow(iou_matrix.cpu().numpy(), cmap='hot', vmin=0, vmax=1)
    plt.title(f"IoU Heatmap {title}")
    plt.colorbar()
    plt.show()

def process_image(
        model_config="D:/Deep_Learning/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        model_weights="D:/Deep_Learning/GroundingDINO/weights/groundingdino_swint_ogc.pth",
        image_path=None,
        text_prompt=None,
        box_threshold=0.5,
        text_threshold=0.25,
        nms_threshold=0.4,
        output_dir="results",
        visualize=True
    ):
    """
    Process an image with GroundingDINO model
    
    Args:
        model_config: Path to model config file
        model_weights: Path to model weights
        image_path: Path to input image
        text_prompt: Text prompt for detection
        box_threshold: Box confidence threshold
        text_threshold: Text similarity threshold
        nms_threshold: NMS IoU threshold
        output_dir: Directory to save results
        visualize: Whether to show visualization
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model (use CUDA if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_config, model_weights)
    model.to(device)
    model.eval()
    
    # Load image
    image_source, image = load_image(image_path)
    image = image.to(device)
    
    # Run prediction
    with torch.no_grad():
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
    
    print(f"Initial detections: {boxes.shape[0]}")
    
    # Apply NMS
    if boxes.shape[0] > 0:
        boxes, logits, phrases = apply_nms_per_phrase(
            image_source, boxes, logits, phrases, nms_threshold)
    
    print(f"After NMS: {boxes.shape[0]} detections")
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    result_path = os.path.join(output_dir, f"{base_name}_result.jpg")
    
    # Annotate and save image
    annotated_frame = annotate(
        image_source=image_source,
        boxes=boxes,
        logits=logits,
        phrases=phrases
    )
    cv2.imwrite(result_path, annotated_frame)
    
    # Visualize results if requested
    if visualize:
        visualize_results(
            image_source,
            boxes.cpu() if boxes.device.type == 'cuda' else boxes,
            phrases,
            logits.cpu() if logits.device.type == 'cuda' else logits,
            save_path=os.path.join(output_dir, f"{base_name}_plot.png")
        )
    
    return boxes, logits, phrases


# def process_folders(
#         model_weights,
#         folders=["D:/NF/C", "D:/NF/D", "D:/NF/ND"],
#         output_base_dir="D:/Deep_Learning/Grounding-Dino-FineTuning/mydsresult",
#         #text_prompt = ["Crack", "Delamination", "No_defect"],  # Better prompt format
#         text_prompt = "Crack . Delamination . No_defect .",
#         box_threshold=0.35,
#         text_threshold=0.25,
#         nms_threshold=0.7  # More aggressive NMS
#     ):
#     """
#     Updated with metrics collection
#     """
#     os.makedirs(output_base_dir, exist_ok=True)
#     metrics = defaultdict(list)
    
#     for folder in folders:
#         folder_name = os.path.basename(folder.rstrip('/\\'))
#         output_dir = os.path.join(output_base_dir, folder_name)
#         os.makedirs(output_dir, exist_ok=True)
        
#         print(f"\nProcessing folder: {folder}")
        
#         for img_name in os.listdir(folder):
#             if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 continue
                
#             img_path = os.path.join(folder, img_name)
#             print(f"Processing image: {img_name}")
            
#             try:
#                 boxes, logits, phrases = process_image(
#                     model_weights=model_weights,
#                     image_path=img_path,
#                     text_prompt=text_prompt,
#                     box_threshold=box_threshold,
#                     text_threshold=text_threshold,
#                     nms_threshold=nms_threshold,
#                     output_dir=output_dir,
#                     visualize=False
#                 )
                
#                 # Record metrics
#                 metrics['image_name'].append(img_name)
#                 metrics['folder'].append(folder_name)
#                 initial_dets = boxes.shape[0] if boxes is not None else 0
#                 final_dets = len(boxes) if boxes is not None else 0
#                 metrics['initial_detections'].append(initial_dets)
#                 metrics['final_detections'].append(final_dets)
#                 metrics['nms_reduction'].append(1 - (final_dets / initial_dets) if initial_dets > 0 else 0)
#                 metrics['mean_confidence'].append(logits.mean().item() if logits.numel() > 0 else 0)
#                 unique_phrases = set(phrases)
#                 metrics['unique_phrases'].append(len(unique_phrases))
#                 metrics['phrases'].append(", ".join(unique_phrases))
                
#             except Exception as e:
#                 print(f"Error processing {img_name}: {str(e)}")
#                 continue
    
#     # Save metrics
#     import pandas as pd
#     metrics_df = pd.DataFrame(metrics)
#     metrics_path = os.path.join(output_base_dir, 'detection_metrics.csv')
#     metrics_df.to_csv(metrics_path, index=False)
#     print(f"\nSaved detailed metrics to {metrics_path}")
    
#     # Print summary
#     print("\nSummary Statistics:")
#     print(f"Total images processed: {len(metrics['image_name'])}")
#     print(f"Average initial detections: {pd.Series(metrics['initial_detections']).mean():.1f}")
#     print(f"Average final detections: {pd.Series(metrics['final_detections']).mean():.1f}")
#     print(f"Average NMS reduction: {pd.Series(metrics['nms_reduction']).mean():.1%}")
#     print(f"Average confidence: {pd.Series(metrics['mean_confidence']).mean():.3f}")

def smart_matching_selection(
    boxes, logits, phrases, max_per_image=5,
    iou_threshold=0.5, x_distance_thresh=0.05, height_overlap_thresh=0.7
):
    """
    Enhanced 1-to-1 box selection to reduce redundant crack boxes.
    Filters using:
    - IoU
    - x-center proximity
    - vertical overlap
    """
    if boxes.shape[0] == 0:
        return boxes, logits, phrases

    selected_boxes = []
    selected_logits = []
    selected_phrases = []

    indices = torch.argsort(logits, descending=True)

    def vertical_overlap(b1, b2):
        y1_top, y1_bot = b1[1], b1[3]
        y2_top, y2_bot = b2[1], b2[3]
        inter = max(0, min(y1_bot, y2_bot) - max(y1_top, y2_top))
        height1 = y1_bot - y1_top
        height2 = y2_bot - y2_top
        return inter / min(height1, height2)

    for i in indices:
        box = boxes[i]
        logit = logits[i]
        phrase = phrases[i]

        too_close = False
        for sel_box in selected_boxes:
            iou = ops.box_iou(box.unsqueeze(0), sel_box.unsqueeze(0))[0][0].item()

            cx1 = (box[0] + box[2]) / 2
            cx2 = (sel_box[0] + sel_box[2]) / 2
            x_dist = abs(cx1 - cx2)

            h_overlap = vertical_overlap(box, sel_box)

            if iou > iou_threshold or (x_dist < x_distance_thresh and h_overlap > height_overlap_thresh):
                too_close = True
                break

        if not too_close:
            selected_boxes.append(box)
            selected_logits.append(logit)
            selected_phrases.append(phrase)

        if len(selected_boxes) >= max_per_image:
            break

    return (
        torch.stack(selected_boxes),
        torch.tensor(selected_logits),
        selected_phrases
    )

def merge_overlapping_boxes(boxes, logits, phrases, iou_threshold=0.5):
    """
    Merge overlapping boxes of the same class using weighted average of coordinates.
    """
    if boxes.shape[0] == 0:
        return boxes, logits, phrases

    merged_boxes = []
    merged_logits = []
    merged_phrases = []

    used = set()

    for i in range(boxes.shape[0]):
        if i in used:
            continue

        box_i = boxes[i]
        logit_i = logits[i]
        phrase_i = phrases[i]

        group = [(box_i, logit_i)]
        used.add(i)

        for j in range(i + 1, boxes.shape[0]):
            if j in used or phrases[j] != phrase_i:
                continue

            box_j = boxes[j]
            iou = ops.box_iou(box_i.unsqueeze(0), box_j.unsqueeze(0))[0][0].item()

            if iou > iou_threshold:
                group.append((box_j, logits[j]))
                used.add(j)

        # Merge group using weighted average
        if len(group) == 1:
            merged_boxes.append(group[0][0])
            merged_logits.append(group[0][1])
            merged_phrases.append(phrase_i)
        else:
            weights = torch.tensor([g[1] for g in group])
            boxes_tensor = torch.stack([g[0] for g in group])
            weighted_box = (boxes_tensor.T @ weights) / weights.sum()
            merged_boxes.append(weighted_box)
            merged_logits.append(weights.mean())
            merged_phrases.append(phrase_i)

    return (
        torch.stack(merged_boxes),
        torch.tensor(merged_logits),
        merged_phrases
    )


def process_single_image_for_debug(
        model_config="D:/Deep_Learning/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        model_weights="D:/Deep_Learning/GroundingDINO/weights/groundingdino_swint_ogc.pth",
        image_path=None,
        text_prompt=None,
        box_threshold=0.35,
        text_threshold=0.25,
        nms_threshold=0.5,  # Can be float or dict
        visualize_intermediate=True,
        confidence_boost={'delamination': 1},
        min_counts={'crack': 10, 'delamination': 5}
    ):
    """
    Final working version with:
    - Fixed NMS threshold handling
    - Confidence boosting
    - Minimum detection enforcement
    """
    # Load model and image
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_config, model_weights).to(device)
    image_source, image = load_image(image_path)
    image = image.to(device)
    
    # Run prediction
    with torch.no_grad():
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
    
    # Standardize phrases
    phrases = [p.lower().strip() for p in phrases]
    
    # Apply confidence boost
    for cls, boost_factor in confidence_boost.items():
        cls_indices = [i for i,p in enumerate(phrases) if cls in p]
        if cls_indices:
            logits[cls_indices] *= boost_factor
            print(f"Boosted confidence for {len(cls_indices)} {cls} detections (x{boost_factor})")
    
    # Debug output before NMS
    print(f"\n=== Initial Detections ===")
    print(f"Total detections: {len(phrases)}")
    for cls in set(phrases):
        cls_count = phrases.count(cls)
        cls_conf = logits[[i for i,p in enumerate(phrases) if p==cls]].mean().item()
        print(f"  {cls}: {cls_count} boxes | Avg confidence: {cls_conf:.4f}")
    
    # if visualize_intermediate:
    #     visualize_results(image_source, boxes.cpu(), phrases, logits.cpu(), "before_nms.png")
    
    # Apply NMS
    if boxes.shape[0] > 0:
        boxes_after_nms, logits_after_nms, phrases_after_nms = apply_nms_per_class(
            boxes, logits, phrases, nms_threshold)
        debug_iou_heatmap(boxes_after_nms, title="Post-NMS")
        # Smart matching to reduce redundancy
        boxes_smart, logits_smart, phrases_smart = smart_matching_selection(
            boxes_after_nms, logits_after_nms, phrases_after_nms, max_per_image=5, iou_threshold=0.5)
        
        # Merge overlapping boxes
        boxes_final, logits_final, phrases_final = merge_overlapping_boxes(
            boxes_smart, logits_smart, phrases_smart, iou_threshold=0.2)

    else:
        boxes_after_nms, logits_after_nms, phrases_after_nms = boxes, logits, phrases
        
    
    # print(f"\n=== After NMS ===")
    # print(f"Detections remaining: {len(phrases_after_nms)}")
    # print("Boxes after NMS:")
    # for box, conf, label in zip(boxes_after_nms, logits_after_nms, phrases_after_nms):
    #     box = box.tolist() if isinstance(box, torch.Tensor) else box
    #     print(f"{label.upper():<12} | Box: {np.round(box, 2)} | Confidence: {conf:.4f}")
    print(f"\n=== After Smart Matching ===")
    print(f"Detections remaining: {len(phrases_final)}")
    print("Boxes selected:")
    for box, conf, label in zip(boxes_final, logits_final, phrases_final):
        box = box.tolist() if isinstance(box, torch.Tensor) else box
        print(f"{label.upper():<12} | Box: {np.round(box, 2)} | Confidence: {conf:.4f}")


    # Enforce minimum detections
    final_boxes = boxes_final.tolist() if isinstance(boxes_final, torch.Tensor) else boxes_final
    final_logits = logits_final.tolist() if isinstance(logits_final, torch.Tensor) else logits_final
    final_phrases = phrases_final.copy()


    for cls, min_count in min_counts.items():
        cls_indices = [i for i,p in enumerate(final_phrases) if cls in p]
        if len(cls_indices) < min_count:
            all_cls_indices = [i for i,p in enumerate(phrases) if cls in p]
            if all_cls_indices:
                sorted_indices = sorted(all_cls_indices, 
                                      key=lambda i: logits[i]/confidence_boost.get(cls, 1.0), 
                                      reverse=True)[:min_count]
                
                if isinstance(boxes, torch.Tensor):
                    final_boxes.extend(boxes[sorted_indices].tolist())
                    final_logits.extend(logits[sorted_indices].tolist())
                else:
                    final_boxes.extend([boxes[i] for i in sorted_indices])
                    final_logits.extend([logits[i] for i in sorted_indices])
                final_phrases.extend([phrases[i] for i in sorted_indices])
                
                print(f"Added {min_count-len(cls_indices)} {cls} detections to meet minimum")

    # Convert back to tensors if needed
    if isinstance(boxes, torch.Tensor):
        final_boxes = torch.tensor(final_boxes)
        final_logits = torch.tensor(final_logits)
    # ✅ Add this block here
    print(f"\nBefore final merging: {len(final_phrases)} boxes")
    final_boxes, final_logits, final_phrases = merge_overlapping_boxes(
        final_boxes, final_logits, final_phrases, iou_threshold=0.2)
    print(f"After final merging: {len(final_phrases)} boxes")
    # ✅ Remove contained boxes after merging
    filtered_boxes = []
    filtered_logits = []
    filtered_phrases = []
    
    for i in range(len(final_boxes)):
        box_i = final_boxes[i]
        contained = False
        for j in range(len(final_boxes)):
            if i != j and final_phrases[i] == final_phrases[j]:
                box_j = final_boxes[j]
                if (
                    box_i[0] >= box_j[0] and box_i[1] >= box_j[1] and
                    box_i[2] <= box_j[2] and box_i[3] <= box_j[3]
                ):
                    contained = True
                    break
        if not contained:
            filtered_boxes.append(box_i)
            filtered_logits.append(final_logits[i])
            filtered_phrases.append(final_phrases[i])
    
    final_boxes = torch.stack(filtered_boxes)
    final_logits = torch.tensor(filtered_logits)
    final_phrases = filtered_phrases

    # === Smart Filtering for Visualization ===
    conf_thresh = 0.7
    #priority_class = 'crack'
    top_idx = torch.argmax(final_logits)
    priority_class = final_phrases[top_idx]
    spread_thresh = 0.05
    
    # Step 1: Filter by confidence
    high_conf_indices = [i for i, c in enumerate(final_logits) if c >= conf_thresh]
    
    # Step 2: Prioritize crack boxes
    #priority_indices = [i for i in high_conf_indices if final_phrases[i] == priority_class]
    priority_indices = [i for i in high_conf_indices if final_phrases[i] == priority_class]
    
    # Step 3: Filter by confidence spread
    if priority_indices:
        top_conf = max([final_logits[i] for i in priority_indices])
        filtered_indices = [i for i in priority_indices if abs(final_logits[i] - top_conf) <= spread_thresh]
    else:
        filtered_indices = []
    
    # Step 4: Fallback to top 3 if nothing passes
    if not filtered_indices:
        top_indices = torch.argsort(final_logits, descending=True)[:3]
        filtered_indices = top_indices.tolist()
    
    # Step 5: Prepare filtered outputs
    filtered_boxes = final_boxes[filtered_indices]
    filtered_logits = final_logits[filtered_indices]
    filtered_phrases = [final_phrases[i] for i in filtered_indices]
    
    # Step 6: Visualize
    visualize_results(
        image_source,
        filtered_boxes,
        filtered_phrases,
        filtered_logits,
        save_path="filtered_high_conf_boxes.png"
    )
    
    # Optional: Print summary
    print(f"\nSmart filtered boxes for visualization:")
    for box, conf, label in zip(filtered_boxes, filtered_logits, filtered_phrases):
        print(f"{label.upper():<12} | Box: {np.round(box.tolist(), 3)} | Confidence: {conf:.4f}")

    # Final output
    print(f"\n=== Final Results ===")
    print(f"Total detections: {len(final_phrases)}")
    for cls in set(final_phrases):
        cls_indices = [i for i,p in enumerate(final_phrases) if p==cls]
        avg_conf = final_logits[cls_indices].mean().item()
        print(f"  {cls}: {len(cls_indices)} boxes | Avg confidence: {avg_conf:.4f}")
    
    # ✅ Add this block here
    if visualize_intermediate:
        visualize_results(
            image_source,
            final_boxes,
            final_phrases,
            final_logits,
            save_path="all_merged_boxes.png"
        )
   
    if visualize_intermediate:
        if not isinstance(final_boxes, torch.Tensor):
            final_boxes = torch.tensor(final_boxes)
        if not isinstance(final_logits, torch.Tensor):
            final_logits = torch.tensor(final_logits)
    
        # Step 1: Sort all by confidence
        top_indices = torch.argsort(final_logits, descending=True)
        sorted_boxes = final_boxes[top_indices]
        sorted_logits = final_logits[top_indices]
        sorted_phrases = [final_phrases[i] for i in top_indices]
    
        # Step 2: Smart selection - avoid contained boxes, and always collect top 3
        selected_boxes = []
        selected_logits = []
        selected_phrases = []
    
        def is_inside(inner, outer):
            return (
                inner[0] >= outer[0] and inner[1] >= outer[1] and
                inner[2] <= outer[2] and inner[3] <= outer[3]
            )
    
        for i in range(len(sorted_boxes)):
            box_i = sorted_boxes[i]
            skip = False
            for sel_box in selected_boxes:
                if is_inside(box_i, sel_box):
                    skip = True
                    break
            if not skip:
                selected_boxes.append(box_i)
                selected_logits.append(sorted_logits[i])
                selected_phrases.append(sorted_phrases[i])
            if len(selected_boxes) >= 3:
                break  # stop once we have 3 non-contained boxes
    
        selected_boxes = torch.stack(selected_boxes)
        selected_logits = torch.stack(selected_logits)
    
        print(f"\nTop {len(selected_boxes)} smart boxes drawn (non-contained):")
        for box, conf, label in zip(selected_boxes, selected_logits, selected_phrases):
            print(f"{label.upper():<12} | Box: {np.round(box.tolist(), 3)} | Confidence: {conf:.4f}")
    
        visualize_results(
            image_source,
            selected_boxes,
            selected_phrases,
            selected_logits,
            save_path="smart_top_boxes.png"
        )

    return final_boxes, final_logits, final_phrases

if __name__ == "__main__":
    # Test with a single image first
    test_image_path = "D:/NF/C/Picture11.png"  # Replace with your test image path
    print(f"\nTesting single image: {test_image_path}")
    
    boxes, logits, phrases = process_single_image_for_debug(
        model_weights="D:/Deep_Learning/Grounding-Dino-FineTuning/weights_myds_1/epoch_55.pth",
        image_path=test_image_path,
        text_prompt = "Crack . Delamination . No_defect .",
        nms_threshold={'crack': 0.4, 'delamination': 0.7}
    )
    
    # After debugging, you can run the full processing if everything looks good
    # process_folders(...)
    
# if __name__ == "__main__":
#     process_folders(
#         model_weights="D:/Deep_Learning/Grounding-Dino-FineTuning/weights_myds/epoch_10.pth",
#         text_prompt="Crack . Delamination . No defect .",
#         nms_threshold=0.7
#     )