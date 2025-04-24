import torch.nn as nn

class BaseOCR(nn.Module):
    """
    Base Detector abstract class
    """
    def __init__(self):
        super().__init__()

    def preprocess(self, images, bboxes):
        """
        Preprocess the input images before being passed into model
        :params:
            images: images in cv2 format.
            bboxes: bounding boxes of license plate in the images, required in format [xmin, ymin, xmax, ymax ...]
        :return: processed image after being cropped after bboxes and padded
        """
        raise NotImplementedError("This is an interface method")

    def postprocess(self, adv_images):
        """
        Postprocess the adversarial image after being attacked.
        Convert the adversarial image into cv2 format
        :params:
            adv_images: attacked images.
        :return: cv2 image
        """
        raise NotImplementedError("This is an interface method")

    def forward(self, adv_images, targets):
        """
        Forward the attacking image and targets to compute gradients
        :params:
            adv_images: adversarial images, also stores gradients.
            targets: targets fit model and adversarial image.
        :return: 
            loss: loss of the model.
            loss value: loss value of the model.
        """
        raise NotImplementedError("This is an interface method")

    def detect(self, query_input):
        """
        Model inference on the processed input
        :params:
            query_input: processed input. The images have been cropped and padded
        :return: model predictions
        """
        raise NotImplementedError("This is an interface method")

    def make_targets(self, predictions, images):
        """
        Make the targets from the predictions of model
        :params:
            predictions: model prediction.
            images: list of cv2 image.
        :return: model targets
        """
        raise NotImplementedError("This is an interface method")
    
    def applied_targets(self, targets):
        """
        Apply the final modified to the targets
        :params:
            targets: targets fit model and adversarial image.
        :return: applied targets
        """
        return targets
    
    def filter_targets(self, deid_images, targets):
        """
        Filter out the images whose targets can not be recogized in the deid images
        :params:
            deid_images: list of deid images.
            targets: list of targets.
        :return: filtered deid images and targets
        """
        return deid_images, targets

    def get_plates_and_bboxes(self, predictions):
        """
        Extract the bounding box and the plate characters from model predictions
        :params:
            predictions: model prediction.
            
        :return: 
          lp: license plate characters and  
          bboxes: bounding box of characters in the license plate. In (x1,y1,x2,y2) format
        """
        raise NotImplementedError("This is an interface method")