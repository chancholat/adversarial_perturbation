import cv2
import numpy as np
import torch
import torch.nn as nn
import os

from .base import BaseDetector
from ._models.yolov5.utils.augmentations import letterbox
from ._models.yolov5.utils.general import xyxy2xywhn
from ._models.yolov5.utils.loss import ComputeLoss

def loadDetectModel():
    # Get the absolute path of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate two levels up to reach the root directory
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    
    # Construct the model path relative to the discovered root
    model_path = os.path.join(root_dir, 'assets', 'pretrained', 'License-Plate-Recognition', 'model', 'LP_detector.pt')
    det_model = torch.hub.load(os.path.join(root_dir, '_models', 'detection', '_models', 'yolov5'), 
                                 'custom', path=model_path, 
                                 force_reload=True, source='local')
    
    for param in det_model.model.model.parameters():
        param.requires_grad = False
    
    return det_model


class YOLOv5Detector(BaseDetector):
    def __init__(self):
        super(YOLOv5Detector, self).__init__()

        self.model = loadDetectModel() # ?
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.compute_loss = ComputeLoss(self.model.model.model)

    def preprocess(self, images):
        preprocess_imgs = []

        for img in images:
            # Resizes and pads image to new_shape (640 for yolo) with stride-multiple constraints, returns resized image, ratio, padding.
            pad_img, ratio, pad = letterbox(img, 640, auto=False, scaleup=True)

            # pad_img = pad_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            # pad_img = np.ascontiguousarray(pad_img)
            preprocess_imgs.append(pad_img)

        preprocess_imgs = np.stack(preprocess_imgs, axis=0)
        # preprocess_imgs =  torch.from_numpy(preprocess_imgs)
        return preprocess_imgs
  
    def postprocess(self, adv_images):
        adv_images = [adv_image.detach().cpu().numpy().transpose(1,2,0) * 255.0 for adv_image in adv_images]
        return adv_images

    def forward(self, adv_images, targets):
        self.model.model.model.train()

        if len(adv_images.shape) == 3:
            adv_images = adv_images.unsqueeze(0)

        adv_images = adv_images.to(self.device)
        targets = targets.to(self.device)

        predictions = self.model.model.model(adv_images)
        loss, loss_items = self.compute_loss(predictions, targets)

        self.model.model.model.eval()
        return loss

    def detect(self, images):
        self.model.eval()
        predictions = []

        if len(images.shape) == 3:
            images = images.unsqueeze(0)

        for img in images:
            pred = self.model(img, size=640)
            pred = pred.pandas().xyxy[0].values.tolist()
            predictions.append(pred)

        return predictions
    
    def applied_targets(self, targets):
        return torch.cat(targets)
   
    def make_targets(self, predictions, images):
        targets = []
        for i, (pred, image) in  enumerate(zip(predictions, images)):
            h, w, _ = image.shape
            
            # extract class number, xmin, ymin, xmax, ymax
            pred = np.array([[item[5], item[0], item[1], item[2], item[3]] for item in pred])
            
            if len(pred) == 0:
                pred = np.zeros((0, 5))
                
            nl = len(pred)
            target = torch.zeros((nl, 6))
            # convert xyxy to xc, yc, wh
            pred[:, 1:5] = xyxy2xywhn(pred[:, 1:5], w=w, h=h, clip=True, eps=1e-3)
            target[:, 1:] = torch.from_numpy(pred)

            # add image index for build target
            target[:, 0] = i
            targets.append(target)

        return targets

    def get_bboxes(self, predictions):
        bboxes = []
        for pred in predictions:
            bbox_list = [[int(point) for point in box[0:4]] for box in pred]
            bboxes.append(bbox_list)
        
        return bboxes   

# det_model = YOLOv5Detector()