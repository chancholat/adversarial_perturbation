import torch

from attack.algorithms import get_optim
from .base import Attacker


class FullAttacker(Attacker):
    """
    Adversarial attack on Face Detection / Landmark Estimation / Gaze Estimation models
    :params:
        optim: name of attack algorithm
        max_iter: maximum number of iterations
        eps: epsilon param
    """

    def __init__(self, optim, max_iter=250, eps=8/255, eps1=8 / 255.0, eps2= 8/255):
        super().__init__(optim, max_iter, eps, eps1, eps2)

    def _generate_targets(self, victims, images, **kwargs):
        """
        Generate target for image using victim models
        :params:
            images: list of cv2 image
            victims: dictionary of victim models.
        :return:
            targets_dict: dict of targets which required for _iterative_attack method
        """

        targets_dict = {}

        # Generate detection targets
        # Normalize image
        det_query = victims["detection"].preprocess(images)

        # To tensor, allow gradients to be saved
        # det_query_tensor = self._generate_tensors(det_query)

        # Detect on raw image
        predictions = victims["detection"].detect(det_query)

        # Make targets and face_box
        det_targets = victims["detection"].make_targets(predictions, det_query)

        # face_boxes = victims["detection"].get_face_boxes(predictions)
        bboxes = victims["detection"].get_bboxes(predictions)

        targets_dict["detection"] = det_targets

        # Generate OCR targets
        if "OCR" in victims.keys():
            # Normalize image
            rec_query = victims["OCR"].preprocess(det_query, bboxes)

            # Detect on raw image
            rec_predictions = victims["OCR"].detect(rec_query)

            # Make targets
            ocr_targets = victims["OCR"].make_targets(rec_predictions, rec_query, **kwargs)
            targets_dict["OCR"] = ocr_targets

        return targets_dict

    def _iterative_attack(self, att_imgs, targets, victims, optim, max_iter, mask=None):
        """
        Performs iterative adversarial attack on batch images
        :params:
            att_imgs: input attack image
            targets: dictionary of attack targets
            victims: dictionary of victim models.
            optim: optimizer
            max_iter: maximum number of attack iterations
            mask: gradient mask
        :return:
            results: tensor image with updated gradients
        """

        # Batch size for normalizing loss
        batch_size = att_imgs.shape[0]

        iter = 0
        # Start attack
        while True:
            optim.zero_grad()
            with torch.set_grad_enabled(True):

                # Forward face detection model
                det_loss = victims["detection"](att_imgs, targets["detection"])

                if "OCR" in victims.keys():
                    # Generate cropped tensors to prepare for OCR model
                    # Apply other kind of attack here
                    #===== TODO (OR NOT) ===== #

                    #===== ATTACK OCR ======= #
                    # Forward ocr model
                    ocr_loss = victims["OCR"](att_imgs, targets["OCR"])


                # Sum up loss
                if det_loss.item() / batch_size > self.eps1:
                    loss = det_loss
                elif "OCR" in victims.keys() and ocr_loss.item() / batch_size > self.eps2:
                    loss = ocr_loss + det_loss
                else:
                    break

                loss.backward()

            
            if mask is not None:
               with torch.no_grad():
                  att_imgs.grad *= mask

            optim.step()

            iter += 1
            if iter == max_iter:
                break

        print("Number of iter: ", iter)
        return att_imgs
    
    def filter_targets(self, deid_images, targets, victims):
        """
        Filter out the  deid images whose targets can not be recogized by the models
        :params:
            deid_images: list of deid images.
            victims: dictionary of victim models.
            targets: list of targets.
        :return: filtered deid images and targets
        """
        # Filter out the images whose targets can not be recogized in the deid images
        deid_images, targets = victims["detection"].filter_targets(deid_images, targets)
        if "OCR" in victims.keys():
            deid_images, targets  = victims["OCR"].filter_targets(deid_images, targets)
        
        return deid_images, targets

    def applied_targets(self, targets, victims):
        """
        Apply the final modified to the targets
        :params:
            victims: dictionary of victim models.
            targets: targets fit model and adversarial image.   
        :return: applied targets
        """
        # Apply the final modified to the targets
        targets['detection'] = victims["detection"].applied_targets(targets['detection'])
        if "OCR" in victims.keys():
            targets['OCR'] = victims["OCR"].applied_targets(targets['OCR'])

        return targets

    def attack(self, victims, images, deid_images, optim_params={}, **kwargs):
        """
        Performs attack flow on image
        :params:
            images: list of rgb cv2 images
            victims: dictionary of victim models.
            deid_images: list of De-identification cv2 images
            optim_params: keyword arguments that will be passed to optim
        :return:
            adv_res: adversarial cv2 images
        """

        # assert (
        #     "detection" in victims.keys() and "alignment" in victims.keys()
        # ), "Need both detection and alignment models to attack"

        targets = self._generate_targets(victims, images, **kwargs)

        # Filter out the images whose targets can not be recogized in the deid images
        deid_images, targets = self.filter_targets(deid_images, targets, victims)

        if len(deid_images) == 0:
            print("No images to attack")
            return None
        
        # Apply the final modified to the targets
        targets = self.applied_targets(targets, victims)

        # Process deid images for detection model
        deid_norm = victims["detection"].preprocess(deid_images)

        # To tensors and turn on gradients flow
        deid_tensor = self._generate_tensors(deid_norm)
        deid_tensor.requires_grad = True

        # Get attack algorithm
        optim = get_optim(
            self.optim, params=[deid_tensor], epsilon=self.eps, **optim_params
        )

        # Start iterative attack
        adv_res = self._iterative_attack(
            deid_tensor,
            targets=targets,
            victims=victims,
            optim=optim,
            max_iter=self.max_iter,
        )

        # Postprocess, return cv2 image
        adv_images = victims["detection"].postprocess(adv_res)
        return adv_images
