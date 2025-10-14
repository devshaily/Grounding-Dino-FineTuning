from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import torch
import torchvision.ops as ops
from torchvision.ops import box_convert
import os
import matplotlib.pyplot as plt
from collections import defaultdict

def apply_nms_per_phrase(image_source, boxes, logits, phrases, threshold=0.5):  # Increased default threshold
    """
    Improved NMS that handles phrase variations better
    """
    h, w, _ = image_source.shape
    scaled_boxes = boxes * torch.Tensor([w, h, w, h])
    scaled_boxes = box_convert(boxes=scaled_boxes, in_fmt="cxcywh", out_fmt="xyxy")
    
    # Clean phrases for better grouping
    cleaned_phrases = [p.strip().lower().replace('_', ' ').replace('.', ' ') for p in phrases]
    unique_phrases = set(cleaned_phrases)
    
    print(f"Unique detected phrases: {unique_phrases}")
    
    nms_boxes_list, nms_logits_list, nms_phrases_list = [], [], []
    
    for unique_phrase in unique_phrases:
        indices = [i for i, phrase in enumerate(cleaned_phrases) if phrase == unique_phrase]
        if not indices:
            continue
            
        phrase_scaled_boxes = scaled_boxes[indices]
        phrase_boxes = boxes[indices]
        phrase_logits = logits[indices]
        
        keep_indices = ops.nms(phrase_scaled_boxes, phrase_logits, threshold)
        
        nms_boxes_list.extend(phrase_boxes[keep_indices])
        nms_logits_list.extend(phrase_logits[keep_indices])
        nms_phrases_list.extend([phrases[indices[i]] for i in keep_indices])  # Keep original phrase
    
    return (torch.stack(nms_boxes_list) if nms_boxes_list else torch.empty(0, 4),
            torch.stack(nms_logits_list) if nms_logits_list else torch.empty(0),
            nms_phrases_list)

def visualize_results(image_source, boxes, phrases, logits, save_path=None):
    """
    Visualize detection results with matplotlib
    
    Args:
        image_source: Original image
        boxes: Detected boxes
        phrases: Text phrases
        logits: Confidence scores
        save_path: Optional path to save visualization
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(image_source)
    
    if boxes.shape[0] > 0:
        h, w = image_source.shape[:2]
        boxes = boxes * torch.Tensor([w, h, w, h])
        boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
        
        for box, phrase, logit in zip(boxes, phrases, logits):
            box = box.int().numpy()
            plt.gca().add_patch(plt.Rectangle(
                (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                fill=False, edgecolor='red', linewidth=2))
            plt.text(
                box[0], box[1], f"{phrase}: {logit:.2f}",
                fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5))
    
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()

def process_image(
        model_config="D:/Deep_Learning/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        model_weights="D:/Deep_Learning/GroundingDINO/weights/groundingdino_swint_ogc.pth",
        image_path=None,
        text_prompt=None,
        box_threshold=0.35,
        text_threshold=0.25,
        nms_threshold=0.7,
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

def process_folders(
        model_weights,
        folders=["D:/NF/C", "D:/NF/D", "D:/NF/ND"],
        output_base_dir="D:/Deep_Learning/Grounding-Dino-FineTuning/mydsresult",
        text_prompt="Crack . Delamination . No defect .",  # Better prompt format
        box_threshold=0.35,
        text_threshold=0.25,
        nms_threshold=0.7  # More aggressive NMS
    ):
    """
    Updated with metrics collection
    """
    os.makedirs(output_base_dir, exist_ok=True)
    metrics = defaultdict(list)
    
    for folder in folders:
        folder_name = os.path.basename(folder.rstrip('/\\'))
        output_dir = os.path.join(output_base_dir, folder_name)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nProcessing folder: {folder}")
        
        for img_name in os.listdir(folder):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(folder, img_name)
            print(f"Processing image: {img_name}")
            
            try:
                boxes, logits, phrases = process_image(
                    model_weights=model_weights,
                    image_path=img_path,
                    text_prompt=text_prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    nms_threshold=nms_threshold,
                    output_dir=output_dir,
                    visualize=False
                )
                
                # Record metrics
                metrics['image_name'].append(img_name)
                metrics['folder'].append(folder_name)
                initial_dets = boxes.shape[0] if boxes is not None else 0
                final_dets = len(boxes) if boxes is not None else 0
                metrics['initial_detections'].append(initial_dets)
                metrics['final_detections'].append(final_dets)
                metrics['nms_reduction'].append(1 - (final_dets / initial_dets) if initial_dets > 0 else 0)
                metrics['mean_confidence'].append(logits.mean().item() if logits.numel() > 0 else 0)
                unique_phrases = set(phrases)
                metrics['unique_phrases'].append(len(unique_phrases))
                metrics['phrases'].append(", ".join(unique_phrases))
                
            except Exception as e:
                print(f"Error processing {img_name}: {str(e)}")
                continue
    
    # Save metrics
    import pandas as pd
    metrics_df = pd.DataFrame(metrics)
    metrics_path = os.path.join(output_base_dir, 'detection_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved detailed metrics to {metrics_path}")
    
    # Print summary
    print("\nSummary Statistics:")
    print(f"Total images processed: {len(metrics['image_name'])}")
    print(f"Average initial detections: {pd.Series(metrics['initial_detections']).mean():.1f}")
    print(f"Average final detections: {pd.Series(metrics['final_detections']).mean():.1f}")
    print(f"Average NMS reduction: {pd.Series(metrics['nms_reduction']).mean():.1%}")
    print(f"Average confidence: {pd.Series(metrics['mean_confidence']).mean():.3f}")

if __name__ == "__main__":
    process_folders(
        model_weights="D:/Deep_Learning/Grounding-Dino-FineTuning/weights_myds/epoch_10.pth",
        text_prompt="Crack . Delamination . No defect .",
        nms_threshold=0.7
    )