  
import albumentations as alb
import cv2
import os
import copy
import numpy as np
import json
from torch.utils.data import Dataset

######################################################################################
#
# CLASS DESCRIBING HOW TO LOAD AND ITERATE OVER THE DATASET 
# This custom class inherits from the abstract class torch.utils.data.Dataset
# An instance of LLNDataset has been created in the model.py file
# 
######################################################################################


class LLNDataset(Dataset):
    # --------------------------------------------------------------------------------
    # INITIALISATION OF THE LLNDATASET INSTANCE
    # INPUTS: 
    #     - imgDirectory (str): path to the folder containing the images 
    #     - maskDirectory (str): path to the folder containing the masks
    #     - setName (string): choose among "train", "val" or "test"
    #     - param (dic): dictionnary containing the parameters defined in the 
    #                    configuration (yaml) file
    # --------------------------------------------------------------------------------
    def __init__(self, imgDirectory, maskDirectory, setName, param):
        self.imgDirectory       = imgDirectory
        self.maskDirectory      = maskDirectory
        self.setName            = setName        
        self.transform_train    = getTransforms_train(param)
        self.transform_val_test = getTransforms_val_test(param)
        
        with open("./Dataset/" + self.setName + ".json", 'r') as f:
            self.files = json.load(f)
        
    # -----------------------------------
    # GET NUMBER OF IMAGES IN THE DATASET
    # -----------------------------------
    def __len__(self):
        return len(self.files)

    # ------------------------------------------------------
    # GET THE IDX'TH IMAGE OF THE DATASET
    # INPUTS: 
    #     - idx (int): index of the image you want to access
    # ------------------------------------------------------
    def __getitem__(self, idx):
        tileName = self.files[idx]
        filepathImg = os.path.join(self.imgDirectory, tileName)        
        originalImg  = cv2.imread(filepathImg)
        originalImg  = cv2.cvtColor(originalImg, cv2.COLOR_BGR2RGB)
        copyImg      = copy.deepcopy(originalImg)

        filepathMask = os.path.join(self.maskDirectory, tileName)
        mask          = cv2.imread(filepathMask, cv2.IMREAD_UNCHANGED)
        
        if self.setName == 'train':
            transform = self.transform_train
        else:
            transform = self.transform_val_test
            
        if transform is not None:
            transformed = transform(image=originalImg, mask=mask)
            image       = transformed["image"]
            mask        = transformed["mask"]
            # Used for ploting results (transform = Resize only!)
            resizedTransform = alb.Compose([t for t in  transform if isinstance(t, (alb.Resize, alb.pytorch.transforms.ToTensorV2))])
            resized          = resizedTransform(image=copyImg)
            resizedImg       = resized["image"]
            
        return image, mask, tileName, resizedImg  


# -----------------------------------------------------------------------
# GET A LIST OF ALBUMENTATION AUGMENTATION TRANSFORMS
# INPUTS: 
#     - param (dic): dictionnary containing the parameters defined in the 
#                    configuration (yaml) file
#
# Do not hesitate to consider other useful transforms!
# -----------------------------------------------------------------------
def getTransforms_train(param): 
    # Base transforms (always applied)
    imgTransformsList = [
        alb.Resize(
            height=int(param["DATASET"]["RESIZE_SHAPE"].split("x")[0]), 
            width=int(param["DATASET"]["RESIZE_SHAPE"].split("x")[1])
        )
    ]
    
    # Add augmentations if enabled
    if param["DATASET"].get("USE_AUGMENTATION", False):
        augmentations = []
        
        # Geometric augmentations
        if param["DATASET"].get("HORIZONTAL_FLIP", 0) > 0:
            augmentations.append(
                alb.HorizontalFlip(p=param["DATASET"]["HORIZONTAL_FLIP"])
            )
        
        if param["DATASET"].get("VERTICAL_FLIP", 0) > 0:
            augmentations.append(
                alb.VerticalFlip(p=param["DATASET"]["VERTICAL_FLIP"])
            )
        
        if param["DATASET"].get("ROTATE_LIMIT", 0) > 0:
            augmentations.append(
                alb.Rotate(
                    limit=param["DATASET"]["ROTATE_LIMIT"], 
                    p=0.5
                )
            )
        
        if param["DATASET"].get("SHIFT_SCALE_ROTATE", 0) > 0:
            augmentations.append(
                alb.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=param["DATASET"]["ROTATE_LIMIT"],
                    p=param["DATASET"]["SHIFT_SCALE_ROTATE"]
                )
            )
        
        # Color augmentations (only on images, not masks)
        if param["DATASET"].get("COLOR_JITTER", 0) > 0:
            augmentations.append(
                alb.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.1,
                    p=param["DATASET"]["COLOR_JITTER"]
                )
            )
        
        if param["DATASET"].get("HUE_SAT_VALUE", 0) > 0:
            augmentations.append(
                alb.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=param["DATASET"]["HUE_SAT_VALUE"]
                )
            )
        
        if param["DATASET"].get("BRIGHTNESS_CONTRAST", 0) > 0:
            augmentations.append(
                alb.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=param["DATASET"]["BRIGHTNESS_CONTRAST"]
                )
            )
        
        # Noise/blur augmentations
        if param["DATASET"].get("GAUSSIAN_BLUR", 0) > 0:
            augmentations.append(
                alb.GaussianBlur(
                    blur_limit=(3, 5),
                    p=param["DATASET"]["GAUSSIAN_BLUR"]
                )
            )
        
        if param["DATASET"].get("GAUSSIAN_NOISE", 0) > 0:
            augmentations.append(
                alb.GaussNoise(
                    var_limit=(10.0, 50.0),
                    p=param["DATASET"]["GAUSSIAN_NOISE"]
                )
            )
        
        # Add augmentations to the transform list
        imgTransformsList.extend(augmentations)
        
        if len(augmentations) > 0:
            print(f"✓ Data augmentation enabled with {len(augmentations)} transforms")
    
    # Final transforms (always applied)
    imgTransformsList.extend([
        alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
        alb.pytorch.transforms.ToTensorV2()
    ])
    
    return alb.Compose(imgTransformsList)

def getTransforms_val_test(param): 
    imgTransformsList = [alb.Resize(height = int(param["DATASET"]["RESIZE_SHAPE"].split("x")[0]), 
                                    width  = int(param["DATASET"]["RESIZE_SHAPE"].split("x")[1])), 
                         alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  
                         alb.pytorch.transforms.ToTensorV2(), 
                        ]
    return alb.Compose(imgTransformsList)
