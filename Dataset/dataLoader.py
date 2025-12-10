  
import albumentations as alb
import cv2
import os
import copy
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
# -----------------------------------------------------------------------
def getTransforms_train(param): 
    transforms_list = [
        alb.Resize(height=int(param["DATASET"]["RESIZE_SHAPE"].split("x")[0]), 
                   width=int(param["DATASET"]["RESIZE_SHAPE"].split("x")[1]))
    ]
    
    force_augmentation = param["DATASET"].get("FORCE_AUGMENTATION", False)
    prob = 1.0 if force_augmentation else 0.5
    
    # Add augmentations if enabled
    if param["DATASET"].get("USE_AUGMENTATION", False):
        if param["DATASET"].get("HORIZONTAL_FLIP", False):
            transforms_list.append(alb.HorizontalFlip(p=prob))
        
        if param["DATASET"].get("VERTICAL_FLIP", False):
            transforms_list.append(alb.VerticalFlip(p=prob))
        
        if param["DATASET"].get("ROTATE_LIMIT", 0) > 0:
            rotate_limit = param["DATASET"].get("ROTATE_LIMIT", 15)
            transforms_list.append(alb.Rotate(
                limit=rotate_limit, 
                p=prob, 
                border_mode=cv2.BORDER_REFLECT_101
            ))
        
        if param["DATASET"].get("BRIGHTNESS_CONTRAST", False):
            transforms_list.append(alb.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=prob
            ))
        
        if param["DATASET"].get("BLUR", False):
            blur_prob = 0.5 if force_augmentation else 0.3
            transforms_list.append(alb.OneOf([
                alb.MotionBlur(blur_limit=3, p=1),
                alb.GaussianBlur(blur_limit=3, p=1),
            ], p=blur_prob))
        
        if param["DATASET"].get("COLOR_JITTER", False):
            transforms_list.append(alb.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=prob
            ))
    
    # Always add normalization and tensor conversion
    transforms_list.extend([
        alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        alb.pytorch.transforms.ToTensorV2()
    ])
    
    return alb.Compose(transforms_list)

def getTransforms_val_test(param): 
    imgTransformsList = [alb.Resize(height = int(param["DATASET"]["RESIZE_SHAPE"].split("x")[0]), 
                                    width  = int(param["DATASET"]["RESIZE_SHAPE"].split("x")[1])), 
                         alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  
                         alb.pytorch.transforms.ToTensorV2(), 
                        ]
    return alb.Compose(imgTransformsList)
