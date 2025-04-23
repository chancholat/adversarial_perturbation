import torch
from torch import nn
import cv2
import numpy as np
import os
import math
import logging

from ._models.ultralytics.ultralytics import YOLO
from ._models.yolov5.utils.augmentations import letterbox
from ._models.ultralytics.ultralytics.utils import DEFAULT_CFG
from ._models.ultralytics.ultralytics.data.augment import Compose, LetterBox, Format, Instances
from ._models.ultralytics.ultralytics.data.dataset import YOLODataset
from ._models.ultralytics.ultralytics.cfg import get_cfg

from .base import BaseOCR

logging.basicConfig(level=logging.INFO)

def crop_image(image, bbox):
  xmin, ymin, xmax, ymax = bbox
  return image[ymin:ymax, xmin:xmax]

def linear_equation(x1, y1, x2, y2):
    b = y1 - (y2 - y1) * x1 / (x2 - x1)
    a = (y1 - b) / x1
    return a, b

def check_point_linear(p, lp, rb):
  x1, y1 = lp
  x2, y2 = rb
  x, y = p
  a, b = linear_equation(x1, y1, x2, y2)
  y_pred = a*x+b
  return(math.isclose(y_pred, y, abs_tol = 3)), y_pred - y


class Yolov8LPOCR(BaseOCR):
  def __init__(self):
    super(Yolov8LPOCR, self).__init__()

    self.model = self.yoloLPOCR()
    self.model.eval()
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model = self.model.to(self.device)
    self.names = self.model.names
    self.nc = self.model.nc
    self.transforms = self.build_transform()
    self.args = get_cfg(DEFAULT_CFG, overrides={})


  def yoloLPOCR(self):
    # Get the absolute path of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate two levels up to reach the root directory
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    
    # Construct the model path relative to the discovered root
    model_path = os.path.join(root_dir, 'assets', 'pretrained', 'yolov8', 'LP_OCR', 'best.pt')
    model = YOLO(model_path)
    return model
  
  def build_transform(self, imgsz=640):
    transforms = Compose([LetterBox(new_shape=(imgsz, imgsz), scaleup=False)])
    transforms.append(
        Format(
            bbox_format="xywh",
            normalize=True,
            return_mask=False,
            return_keypoint=False,
            return_obb=False,
            batch_idx=True,
            mask_ratio=DEFAULT_CFG.mask_ratio,
            mask_overlap=DEFAULT_CFG.overlap_mask,
            bgr=0.0,  # only affect training.
        )
    )
    return transforms

  def preprocess(self, images, bboxes):
    preprocess_imgs = []

    for img, bbox in zip(images, bboxes):
      if not len(bbox):
        preprocess_imgs.append(img) # if there wasn't any detected bbox, then use the whole image as input
        continue 
      bbox = bbox[0] # assume that there only one license plate per image, may be change later
      crop_img = crop_image(img, bbox)
      pad_img, ratio, pad = letterbox(crop_img, 640, auto=False, scaleup=True)

      # pad_img = pad_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
      # pad_img = np.ascontiguousarray(pad_img)
      preprocess_imgs.append(pad_img)
      
    # preprocess_imgs = np.stack(preprocess_imgs, axis=0)
    # preprocess_imgs =  torch.from_numpy(preprocess_imgs)
    return preprocess_imgs
  
  def get_label_from_prediction(self, prediction, image, im_file):
    label = {}
    label["im_file"] = im_file
    shape = image.shape
    label["shape"] = (shape[0], shape[1]) #hw
    if len(prediction) == 0:
      logging.warning(f"Empty prediction: no OCR results found on image {im_file}")
      bboxes = np.array([np.zeros((0, 4))], dtype=np.float32)
      cls = np.array([np.zeros((0, 1))], dtype=np.float32)
    else:
      boxes = prediction.boxes 
      cls = np.array([box.cls[0].cpu() for box in boxes], dtype=np.float32)  
      bboxes = np.array([box.xywhn[0].cpu() for box in boxes], dtype=np.float32)
    label["cls"] = cls
    label["bboxes"] = bboxes
    label["segments"] = []
    label["keypoints"] = None
    label["normalized"] = True
    label["bbox_format"] = 'xywh'
    return label

  def update_labels_info(self, label):
    bboxes = label.pop("bboxes")
    segments = label.pop("segments", [])
    keypoints = label.pop("keypoints", None)
    bbox_format = label.pop("bbox_format")
    normalized = label.pop("normalized")
    segment_resamples = 1000
    segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)

    label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
    return label
  
  def load_image(self, im, imgsz=640):
    h0, w0 = im.shape[:2]
    r = imgsz / max(h0, w0)  # ratio
    if r != 1:  # if sizes are not equal
      w, h = (min(math.ceil(w0 * r), imgsz), min(math.ceil(h0 * r), imgsz))
      im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)

    return im, (h0, w0), im.shape[:2]
  
  @staticmethod
  def collate_fn(batch):
    return YOLODataset.collate_fn(batch)
  
  def filter_targets(self, deid_images, targets):
    # Filter out the images whose targets can not be recogized in the deid images
    filtered_deid_images = []
    filtered_targets = []
    for deid_image, target in zip(deid_images, targets):
      if len(target['bboxes']) == 0:
        continue
     
      filtered_deid_images.append(deid_image)
      filtered_targets.append(target)

    return filtered_deid_images, filtered_targets
  
  def make_targets(self, predictions, images, img_files):
    targets = []
    for prediction, image, im_file in zip(predictions, images, img_files):
      label = self.get_label_from_prediction(prediction, image, im_file)
      label.pop("shape", None)  # shape is for rect, remove it
      label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(image)
      label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
      label = self.update_labels_info(label)
      label = self.transforms(label)
      targets.append(label)
    return targets

  def postprocess(self, adv_images):
    adv_images = [adv_image.detach().cpu().numpy().transpose(1,2,0) * 255.0 for adv_image in adv_images]
    return adv_images

  def preprocess_batch(self, adv_images, targets):
    # preprocess batch
    batch = self.collate_fn(targets)
    if len(adv_images.shape) == 3:
      adv_images = adv_images.unsqueeze(0)
    batch["img"] = adv_images
    batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
    return batch

  def forward(self, adv_images, targets):
    batch = self.preprocess_batch(adv_images, targets)

    self.model.model = self.model.model.to(self.device)
    # set up model attributes
    self.model.model.names = self.model.names
    self.model.model.nc = self.model.nc
    self.model.model.args = self.args
    self.model.model.train()

    freeze_layer_names = []
    # Freeze BN stat
    for n, m in self.model.model.named_modules():
      if any(filter(lambda f: f in n, freeze_layer_names)) and isinstance(m, nn.BatchNorm2d):
        m.eval()

    loss, loss_items = self.model.model(batch)

    self.model.model.eval()
    return loss

  def detect(self, images):
    self.model.eval()
    return self.model(images, verbose=False)

  def sort_lp_chars(self, preds):
    for pred in preds:
      x1, y1, x2, y2 = pred['bbox']
      center_x = (x1 + x2) / 2
      center_y = (y1 + y2) / 2
      pred['center_point'] = (center_x, center_y)


    # find 2 point to draw line
    preds = sorted(preds, key=lambda x: x['center_point'][0])
    lp = preds[0]['center_point']
    rp = preds[-1]['center_point']
    # print(lp, rp)
    min_distance = 300
    LP_type = "1"
    for pred in preds:
      c = pred['center_point']
      if lp[0] != rp[0]:
        check, distance = check_point_linear(c, lp, rp)
        if abs(distance) < abs(min_distance):
          min_distance = abs(distance)
          # if (check_point_linear(ct[0], ct[1], l_point[0], l_point[1], r_point[0], r_point[1]) == False):
        if check:
          LP_type = "2"
    y_sum = sum([item['center_point'][1] for item in preds])
    y_mean = int(int(y_sum) / len(preds))

    # 1 line plates and 2 line plates
    line_1 = []
    line_2 = []
    sorted_preds = []
    # print("Lp_type:", LP_type)
    if LP_type == "2":
        for item in preds:
            if int(item['center_point'][1]) > y_mean:
                line_2.append(item)
            else:
                line_1.append(item)

        for item in sorted(line_1, key = lambda x: x['bbox'][0]):
            sorted_preds.append(item)

        for item in sorted(line_2, key = lambda x: x['bbox'][0]):
            sorted_preds.append(item)
    else:
        for item in sorted(preds, key = lambda x: x['bbox'][0]):
            sorted_preds.append(item)

    return sorted_preds
  
  def get_plates_and_bboxes(self, predictions):
    lps = []
    bboxes = []
    
    for pred in predictions:
      if len(pred) == 0:
          # return "unknown", pred
          lps.append("unknow")
          bboxes.append([])
          continue
       
      preds = []
      for box in pred.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        bboxes.append(box.xywhn[0])
        # print(cls)
        label = self.model.names[cls]  # Assumes model.names maps class index to char

        preds.append({
            'bbox': (x1, y1, x2, y2),
            'char': label,
            'conf': conf
        })

      preds_sorted = self.sort_lp_chars(preds)
      text = ''.join([p['char'] for p in preds_sorted])
      lps.append(text)
      bboxes.append([item['bbox'] for item in preds_sorted])
    return lps, bboxes
