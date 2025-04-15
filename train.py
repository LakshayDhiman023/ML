import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from model import FasterRCNN
from dataset import DAVISDataset, get_transform
import numpy as np
from tqdm import tqdm
import os
import cv2
import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import random

def collate_fn(batch):
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    return images, targets  # torchvision FasterRCNN expects list of tensors, not batched tensor

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = box1_area + box2_area - intersection
    return intersection / union if union > 0 else 0.0

def calculate_f_measure(predictions, ground_truth, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for pred, gt in zip(predictions, ground_truth):
        # Skip if no boxes
        if len(pred['boxes']) == 0 or len(gt['boxes']) == 0:
            false_negatives += len(gt['boxes'])
            false_positives += len(pred['boxes'])
            continue
        
        # Calculate IoU for each prediction
        for i, p_box in enumerate(pred['boxes']):
            # Find best matching ground truth box
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt_box in enumerate(gt['boxes']):
                iou = calculate_iou(p_box.cpu().numpy(), gt_box.cpu().numpy())
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            # Check if prediction matches ground truth
            if best_iou >= iou_threshold:
                true_positives += 1
            else:
                false_positives += 1
        
        # Count ground truth boxes that were not matched
        matched_gt = set()
        for i, p_box in enumerate(pred['boxes']):
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt_box in enumerate(gt['boxes']):
                if j in matched_gt:
                    continue  # Skip already matched ground truth
                    
                iou = calculate_iou(p_box.cpu().numpy(), gt_box.cpu().numpy())
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_threshold and best_gt_idx != -1:
                matched_gt.add(best_gt_idx)
        
        # Count unmatched ground truth as false negatives
        false_negatives += len(gt['boxes']) - len(matched_gt)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f_measure, precision, recall

def calculate_s_measure(predictions, ground_truth):
    """Calculate structural similarity measure"""
    s_scores = []
    
    for pred, gt in zip(predictions, ground_truth):
        # Skip if no boxes
        if len(pred['boxes']) == 0 or len(gt['boxes']) == 0:
            continue
            
        # Convert predictions to binary masks
        pred_mask = np.zeros((480, 854), dtype=np.uint8)  # DAVIS 480p resolution
        gt_mask = np.zeros((480, 854), dtype=np.uint8)
        
        for box in pred['boxes']:
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(854, x2), min(480, y2)
            if x2 > x1 and y2 > y1:  # Valid box
                pred_mask[y1:y2, x1:x2] = 1
            
        for box in gt['boxes']:
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(854, x2), min(480, y2)
            if x2 > x1 and y2 > y1:  # Valid box
                gt_mask[y1:y2, x1:x2] = 1
        
        # Calculate structural similarity
        try:
            s_score = cv2.matchTemplate(pred_mask, gt_mask, cv2.TM_CCOEFF_NORMED)[0][0]
            s_scores.append(s_score)
        except:
            # Skip if error in matching
            pass
    
    return np.mean(s_scores) if s_scores else 0

def calculate_mae(predictions, ground_truth):
    """Calculate Mean Absolute Error"""
    if not predictions or not ground_truth:
        return torch.tensor(0.0)
        
    errors = []
    for pred, gt in zip(predictions, ground_truth):
        if len(pred['boxes']) == 0 or len(gt['boxes']) == 0:
            continue
            
        # Find matching boxes based on IoU
        for p_box in pred['boxes']:
            best_iou = 0
            best_gt_box = None
            
            for gt_box in gt['boxes']:
                iou = calculate_iou(p_box.cpu().numpy(), gt_box.cpu().numpy())
                if iou > best_iou:
                    best_iou = iou
                    best_gt_box = gt_box
            
            if best_iou > 0.5 and best_gt_box is not None:
                error = torch.abs(p_box - best_gt_box).mean()
                errors.append(error)
    
    return torch.stack(errors).mean() if errors else torch.tensor(0.0)

def fine_tune_model(model, train_loader, val_loader, device, num_epochs=2, output_dir='output'):
    """Fine-tune a pre-trained model with minimal computation"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Use a smaller learning rate for fine-tuning
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0005
    )
    
    # Use a learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=3, 
        gamma=0.1
    )
    
    # Track metrics
    best_loss = float('inf')
    metrics = {
        'train_loss': [],
        'val_loss': []
    }
    
    start_time = time.time()
    print(f"Starting fine-tuning for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train for one epoch
        model.train()
        train_loss = 0
        num_batches = 0
        
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Skip empty batches or batches with invalid boxes
            valid_batch = True
            for t in targets:
                if len(t['boxes']) == 0 or t['labels'].max() > model.model.roi_heads.box_predictor.cls_score.out_features - 1:
                    valid_batch = False
                    print(f"Skipping batch with invalid targets: max label = {t['labels'].max() if len(t['labels']) > 0 else 'empty'}")
                    break
                    
            if not valid_batch:
                continue
                
            # Move data to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            try:
                loss_dict = model(images, targets)
                
                # Extract losses
                losses = sum(loss for loss in loss_dict.values())
                
                # Backward pass
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                train_loss += losses.item()
                num_batches += 1
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
        
        # Calculate average training loss
        if num_batches > 0:
            train_loss /= num_batches
        
        # Update learning rate
        lr_scheduler.step()
        
        # Evaluate model
        model.eval()
        val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validating"):
                # Skip empty batches or batches with invalid boxes
                valid_batch = True
                for t in targets:
                    if len(t['boxes']) == 0 or t['labels'].max() > model.model.roi_heads.box_predictor.cls_score.out_features - 1:
                        valid_batch = False
                        print(f"Skipping batch with invalid targets: max label = {t['labels'].max() if len(t['labels']) > 0 else 'empty'}")
                        break
                        
                if not valid_batch:
                    continue
                    
                # Move data to device
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # Forward pass in validation mode
                try:
                    # During training, we need to pass targets to get loss dict
                    # During validation, we can either:
                    # 1. Pass targets to get loss dict
                    # 2. Pass only images to get predictions
                    # Here we still pass targets to get the loss dict for consistency
                    loss_dict = model(images, targets)
                    
                    # Extract losses - ensure this works with the return type
                    if isinstance(loss_dict, dict):
                        losses = sum(loss for loss in loss_dict.values())
                    else:
                        # If not a dict (e.g., it's a tensor), use it directly
                        losses = loss_dict if isinstance(loss_dict, torch.Tensor) else torch.tensor(0.0).to(device)
                    
                    val_loss += losses.item()
                    num_val_batches += 1
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        # Calculate average validation loss
        if num_val_batches > 0:
            val_loss /= num_val_batches
        
        # Track metrics
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            print(f"Saved best model with val_loss: {val_loss:.4f}")
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s")
    
    return metrics

def visualize_predictions(model, val_dataset, device, output_dir, num_samples=5, score_threshold=0.5):
    """
    Visualize model predictions on random samples from the validation dataset
    
    Args:
        model: The trained model
        val_dataset: Validation dataset
        device: Device to run inference on
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
        score_threshold: Threshold for showing detections
    """
    # Create output directory for visualizations
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Get random indices from dataset
    indices = random.sample(range(len(val_dataset)), min(num_samples, len(val_dataset)))
    
    # Process each sample
    for i, idx in enumerate(indices):
        # Get image and target
        image, target = val_dataset[idx]
        
        # Get original image path for reference
        original_image_path = val_dataset.frame_paths[idx]
        sequence_name = original_image_path.split(os.sep)[-3]  # Get sequence name
        frame_name = os.path.basename(original_image_path)
        
        # Move image to device and add batch dimension
        image = image.unsqueeze(0).to(device)
        
        # Get predictions
        with torch.no_grad():
            predictions = model(image)
        
        # Extract first (and only) prediction
        prediction = predictions[0]
        
        # Convert image back to PIL for visualization
        # First denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image.squeeze(0).cpu()
        image = image * std + mean
        image = image.permute(1, 2, 0).numpy()
        image = (image * 255).astype('uint8')
        pil_image = Image.fromarray(image)
        
        # Create figure and axis
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        
        # Get boxes, scores, and labels
        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        
        # Create a title with sequence and frame info
        ax.set_title(f"Sequence: {sequence_name}, Frame: {frame_name}, Detections: {len(boxes[scores > score_threshold])}")
        
        # Ground truth boxes (green)
        gt_boxes = target['boxes'].cpu().numpy()
        gt_labels = target['labels'].cpu().numpy()
        
        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1-10, f"GT: {label}", color='g', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        
        # Prediction boxes (red for low confidence, yellow for high confidence)
        for box, score, label in zip(boxes, scores, labels):
            if score > score_threshold:
                x1, y1, x2, y2 = box
                color = 'y' if score > 0.7 else 'r'
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1, f"{label}: {score:.2f}", color=color, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"detection_{i}_{sequence_name}_{frame_name}"))
        plt.close()
        
        # Also create a simple image with detections for easy viewing
        draw = ImageDraw.Draw(pil_image)
        
        # Draw ground truth
        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
            draw.text((x1, y1-15), f"GT: {label}", fill="green")
        
        # Draw predictions
        for box, score, label in zip(boxes, scores, labels):
            if score > score_threshold:
                x1, y1, x2, y2 = box
                color = "yellow" if score > 0.7 else "red"
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                draw.text((x1, y1), f"{label}: {score:.2f}", fill=color)
        
        # Save the image
        pil_image.save(os.path.join(vis_dir, f"simple_{i}_{sequence_name}_{frame_name}"))
    
    print(f"Visualizations saved to {vis_dir}")
    return vis_dir

def main():
    # Create output directory
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    root_dir = 'DAVIS-2017-trainval-480p/DAVIS'
    train_dataset = DAVISDataset(root_dir, split='train', transform=get_transform(train=True), max_samples=50)
    val_dataset = DAVISDataset(root_dir, split='val', transform=get_transform(train=False), max_samples=25)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    num_classes = 129  # 128 unique class IDs + 1 for background
    model = FasterRCNN(num_classes=num_classes, pretrained=True)
    model.to(device)
    
    # Fine-tune model
    metrics = fine_tune_model(model, train_loader, val_loader, device, num_epochs=2, output_dir=output_dir)
    
    # Evaluate model with metrics
    print("\nEvaluating model with metrics...")
    model.eval()
    
    predictions_list = []
    targets_list = []
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Collecting predictions"):
            # Skip invalid batches
            valid_batch = True
            for t in targets:
                if len(t['boxes']) == 0 or t['labels'].max() > model.model.roi_heads.box_predictor.cls_score.out_features - 1:
                    valid_batch = False
                    break
            
            if not valid_batch:
                continue
                
            # Move data to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass - get predictions only (no targets passed)
            try:
                # For evaluation, pass only images to get predictions
                predictions = model(images)
                
                # Store predictions and targets for metric calculation
                predictions_list.extend(predictions)
                targets_list.extend(targets)
            except Exception as e:
                print(f"Error during evaluation: {e}")
                continue
    
    # Calculate metrics if we have predictions
    if predictions_list and targets_list:
        f_measure, precision, recall = calculate_f_measure(predictions_list, targets_list)
        s_measure = calculate_s_measure(predictions_list, targets_list)
        mae = calculate_mae(predictions_list, targets_list)
        
        print(f"Evaluation Results:")
        print(f"F-measure: {f_measure:.3f} (Precision: {precision:.3f}, Recall: {recall:.3f})")
        print(f"S-measure: {s_measure:.3f}")
        print(f"MAE: {mae:.3f}")
    else:
        print("No valid predictions collected for evaluation.")

    print(f"Unique class IDs in the dataset: {sorted(train_dataset.unique_class_ids)}")

    # Visualize predictions
    visualize_predictions(model, val_dataset, device, output_dir)

if __name__ == '__main__':
    main() 