import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
from PIL import Image
import torchvision.transforms as T

class DAVISDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, frame_interval=1, max_samples=None):
        """
        Args:
            root_dir (string): Directory with DAVIS-2017 dataset (should point to DAVIS-2017-trainval-480p/DAVIS)
            split (string): 'train' or 'val'
            transform (callable, optional): Optional transform to be applied on a frame
            frame_interval (int): Interval between frames to process
            max_samples (int, optional): Maximum number of samples to load (for testing)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.frame_interval = frame_interval
        
        # Validate root directory structure
        required_dirs = ['JPEGImages', 'Annotations', 'ImageSets']
        for dir_name in required_dirs:
            if not os.path.exists(os.path.join(root_dir, dir_name)):
                raise ValueError(f"Required directory {dir_name} not found in {root_dir}")
        
        # Load video sequences for the specified split
        split_file = os.path.join(root_dir, 'ImageSets', '2017', f'{split}.txt')
        if not os.path.exists(split_file):
            raise ValueError(f"Split file {split_file} not found")
            
        with open(split_file, 'r') as f:
            self.sequences = [line.strip() for line in f.readlines() if line.strip()]
        
        # Initialize frame and annotation paths
        self.frame_paths = []
        self.annotation_paths = []
        
        # Track class IDs we've seen for reporting
        self.unique_class_ids = set()
        
        for seq in self.sequences:
            # Get paths for JPEG images
            jpeg_dir = os.path.join(root_dir, 'JPEGImages', '480p', seq)
            if not os.path.exists(jpeg_dir):
                print(f"Warning: Sequence {seq} not found in JPEGImages")
                continue
                
            frames = sorted([f for f in os.listdir(jpeg_dir) if f.endswith('.jpg')])
            
            # Get paths for annotations
            anno_dir = os.path.join(root_dir, 'Annotations', '480p', seq)
            if not os.path.exists(anno_dir):
                print(f"Warning: Sequence {seq} not found in Annotations")
                continue
                
            annotations = sorted([f for f in os.listdir(anno_dir) if f.endswith('.png')])
            
            # Verify frame and annotation counts match
            if len(frames) != len(annotations):
                print(f"Warning: Frame count mismatch for sequence {seq}")
                continue
            
            # Find class IDs in this sequence by examining first annotation
            if annotations:
                try:
                    first_anno_path = os.path.join(anno_dir, annotations[0])
                    first_anno = cv2.imread(first_anno_path, cv2.IMREAD_GRAYSCALE)
                    if first_anno is not None:
                        obj_ids = np.unique(first_anno)
                        obj_ids = obj_ids[obj_ids != 0]  # Remove background
                        self.unique_class_ids.update(obj_ids)
                except Exception as e:
                    print(f"Warning: Error examining class IDs for sequence {seq}: {e}")
            
            # Add paths with specified interval
            for i in range(0, len(frames), frame_interval):
                self.frame_paths.append(os.path.join(jpeg_dir, frames[i]))
                self.annotation_paths.append(os.path.join(anno_dir, annotations[i]))
                
                # Limit samples if requested
                if max_samples and len(self.frame_paths) >= max_samples:
                    break
            
            # Limit samples if requested
            if max_samples and len(self.frame_paths) >= max_samples:
                break
                
        if len(self.frame_paths) == 0:
            raise ValueError("No valid sequences found in the dataset")
            
        print(f"Loaded {len(self.frame_paths)} frames from {len(self.sequences)} sequences")
        print(f"Found class IDs: {sorted(self.unique_class_ids)}")
        if self.unique_class_ids:
            print(f"Max class ID: {max(self.unique_class_ids)}")
    
    def __len__(self):
        return len(self.frame_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.frame_paths[idx]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        image = Image.open(img_path).convert('RGB')
        
        # Load annotation mask
        anno_path = self.annotation_paths[idx]
        if not os.path.exists(anno_path):
            raise FileNotFoundError(f"Annotation file not found: {anno_path}")
        annotation = cv2.imread(anno_path, cv2.IMREAD_GRAYSCALE)
        
        if annotation is None:
            raise ValueError(f"Failed to load annotation: {anno_path}")
        
        # Convert annotation to bounding boxes
        boxes, labels = self._mask_to_boxes(annotation)
        
        # If no boxes found, create a dummy box to avoid errors
        if len(boxes) == 0:
            boxes = [[0, 0, 10, 10]]
            labels = [1]  # Use class ID 1 as a placeholder
            
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Create target dictionary
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        return image, target
    
    def _mask_to_boxes(self, mask):
        """Convert segmentation mask to bounding boxes and labels"""
        boxes = []
        labels = []
        
        # Get unique object IDs (excluding background)
        object_ids = np.unique(mask)
        object_ids = object_ids[object_ids != 0]  # Remove background
        
        for obj_id in object_ids:
            # Get binary mask for current object
            obj_mask = (mask == obj_id).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Skip very small contours
                if cv2.contourArea(contour) < 50:
                    continue
                    
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Skip tiny boxes
                if w < 10 or h < 10:
                    continue
                    
                boxes.append([x, y, x + w, y + h])
                labels.append(int(obj_id))  # Ensure label is int
        
        return boxes, labels

def get_transform(train):
    """Get transforms for training or validation"""
    transforms = []
    # Resize to fixed size
    transforms.append(T.Resize((480, 854)))  # Standard DAVIS 480p resolution
    # Convert PIL image to tensor
    transforms.append(T.ToTensor())
    if train:
        # Add data augmentation transforms
        transforms.extend([
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.RandomGrayscale(p=0.1)
        ])
    # Normalize
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms) 