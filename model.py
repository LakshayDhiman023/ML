import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.ops import nms, roi_align, box_iou
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class RPN(nn.Module):
    def __init__(self, in_channels=1024, mid_channels=512, num_anchors=9):
        super(RPN, self).__init__()
        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.cls_logits = nn.Conv2d(mid_channels, num_anchors, kernel_size=1)
        self.bbox_pred = nn.Conv2d(mid_channels, num_anchors * 4, kernel_size=1)
        
        # Generate anchor boxes
        self.anchors = self._generate_anchors()
        
    def _generate_anchors(self):
        """Generate a set of anchor boxes with different scales and aspect ratios"""
        scales = torch.tensor([8, 16, 32])
        aspect_ratios = torch.tensor([0.5, 1.0, 2.0])
        
        anchors = []
        for scale in scales:
            for ratio in aspect_ratios:
                w = scale * torch.sqrt(ratio)
                h = scale / torch.sqrt(ratio)
                anchors.append([-w/2, -h/2, w/2, h/2])
        
        return torch.tensor(anchors)
        
    def forward(self, x):
        x = F.relu(self.conv(x))
        logits = self.cls_logits(x)
        bbox_pred = self.bbox_pred(x)
        return logits, bbox_pred
    
    def generate_anchors_for_feature_map(self, feature_map_size):
        """Generate anchors for all positions in the feature map"""
        device = next(self.parameters()).device
        anchors = self.anchors.to(device)
        
        # Create a grid of anchor centers
        height, width = feature_map_size
        shifts_x = torch.arange(0, width) * 16  # 16 is the stride
        shifts_y = torch.arange(0, height) * 16
        shifts_x, shifts_y = torch.meshgrid(shifts_x, shifts_y, indexing="ij")
        shifts = torch.stack((shifts_x.reshape(-1), shifts_y.reshape(-1),
                             shifts_x.reshape(-1), shifts_y.reshape(-1)), dim=1).to(device)
        
        # Add shifts to anchors
        num_shifts = shifts.shape[0]
        num_anchors = anchors.shape[0]
        all_anchors = anchors.view(1, num_anchors, 4) + shifts.view(num_shifts, 1, 4)
        all_anchors = all_anchors.reshape(-1, 4)
        
        return all_anchors

class PretrainedFasterRCNN(nn.Module):
    def __init__(self, num_classes=80, pretrained=True):
        super(PretrainedFasterRCNN, self).__init__()
        
        # Load pre-trained model
        self.model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
        
        # Replace the classifier with a new one for our number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)  # +1 for background
        
        # Set NMS threshold for more detections
        self.model.roi_heads.nms_thresh = 0.5
        self.model.roi_heads.score_thresh = 0.05
        
    def forward(self, images, targets=None):
        return self.model(images, targets)

# For backward compatibility, keep the FasterRCNN class name
class FasterRCNN(PretrainedFasterRCNN):
    pass 