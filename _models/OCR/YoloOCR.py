import torch
import torch.nn as nn
import cv2
import numpy as np
import os

from ._models.yolov5.utils.augmentations import letterbox
from ._models.yolov5.utils.general import xyxy2xywhn
from ._models.yolov5.utils.loss import ComputeLoss
from .base import BaseOCR

def crop_image(image, bbox):
  xmin, ymin, xmax, ymax = bbox
  return image[ymin:ymax, xmin:xmax]

# Function to get bounding box for each detected character
import math

# license plate type classification helper function
def linear_equation(x1, y1, x2, y2):
    b = y1 - (y2 - y1) * x1 / (x2 - x1)
    a = (y1 - b) / x1
    return a, b

def check_point_linear(x, y, x1, y1, x2, y2):
    a, b = linear_equation(x1, y1, x2, y2)
    y_pred = a*x+b
    # print("distence: ", y_pred - y)
    return(math.isclose(y_pred, y, abs_tol = 3)), y_pred - y


class YoloLicensePlateOCR(BaseOCR):
  def __init__(self):
    super(YoloLicensePlateOCR, self).__init__()

    self.model = self.yoloLPOCR()
    self.model.eval()
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model = self.model.to(self.device)
    self.compute_loss = ComputeLoss(self.model.model.model)

  def yoloLPOCR(self):
    # Get the absolute path of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate two levels up to reach the root directory
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    
    # Construct the model path relative to the discovered root
    model_path = os.path.join(root_dir, 'assets', 'pretrained', 'License-Plate-Recognition', 'model', 'LP_ocr.pt')
    yolo_LP_OCR = torch.hub.load(os.path.join(root_dir, '_models', 'OCR', '_models', 'yolov5'), 
                                 'custom', path=model_path, 
                                 force_reload=True, source='local')
    
    for param in yolo_LP_OCR.model.model.parameters():
        param.requires_grad = False
    
    return yolo_LP_OCR


  def preprocess(self, images, bboxes):
    preprocess_imgs = []


    for img, bbox in zip(images, bboxes):
      if not len(bbox):
        preprocess_imgs.append(img) # if there wasn't any detected bbox, then use the whole image as input
        continue 
      bbox = bbox[0] # assume that there only one license plate per image, may be change later
      crop_img = crop_image(img, bbox)
      # Resizes and pads image to new_shape (640 for yolo) with stride-multiple constraints, returns resized image, ratio, padding.
      pad_img, ratio, pad = letterbox(crop_img, 640, auto=False, scaleup=True)

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
  
  def filter_targets(self, deid_images, targets):
    # Filter out the images whose targets can not be recogized in the deid images
    filtered_deid_images = []
    filtered_targets = {'OCR': [], 'detection': []}
    
    for i, (deid_image, ocr_target, det_target) in enumerate(zip(deid_images, targets['OCR'], targets['detection'])):
      if len(ocr_target) == 0:
        continue
      filtered_deid_images.append(deid_image)
      filtered_targets['OCR'].append(ocr_target)
      filtered_targets['detection'].append(det_target)

    return filtered_deid_images, filtered_targets
  
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
  
  def get_plates_and_bboxes(self, predictions):
    lps = []
    bboxes = []
    LP_type = "1"
    for pred in predictions:
      if len(pred) == 0:
          # return "unknown", pred
          lps.append("unknow")
          bboxes.append(pred)
          continue
      center_list = []
      y_mean = 0
      y_sum = 0
      for bb in pred:
          x_c = (bb[0]+bb[2])/2
          y_c = (bb[1]+bb[3])/2
          y_sum += y_c

          # x center, y center and the character
          center_list.append([x_c,y_c,bb[-1]])

      # find 2 point to draw line
      l_point = center_list[0]
      r_point = center_list[0]
      min_distance = 300
      for cp in center_list:
          if cp[0] < l_point[0]:
              l_point = cp
          if cp[0] > r_point[0]:
              r_point = cp
      for ct in center_list:
          if l_point[0] != r_point[0]:
            check, distance = check_point_linear(ct[0], ct[1], l_point[0], l_point[1], r_point[0], r_point[1])
            if abs(distance) < abs(min_distance):
              min_distance = abs(distance)

              # if (check_point_linear(ct[0], ct[1], l_point[0], l_point[1], r_point[0], r_point[1]) == False):
            if check:
                  LP_type = "2"

      y_mean = int(int(y_sum) / len(pred))

      # 1 line plates and 2 line plates
      line_1 = []
      line_2 = []
      license_plate = ""
      # print("Lp_type:", LP_type)
      if LP_type == "2":
          for c in center_list:
              if int(c[1]) > y_mean:
                  line_2.append(c)
              else:
                  line_1.append(c)
          for l1 in sorted(line_1, key = lambda x: x[0]):
              license_plate += str(l1[2])
          license_plate += "-"
          for l2 in sorted(line_2, key = lambda x: x[0]):
              license_plate += str(l2[2])
      else:
          for l in sorted(center_list, key = lambda x: x[0]):
              license_plate += str(l[2])

      lps.append(license_plate)
      bboxes.append(pred)
    return lps, bboxes
