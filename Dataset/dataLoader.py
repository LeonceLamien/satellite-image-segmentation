  
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
        
        self.use_aug = param["DATASET"].get("USE_AUGMENTATION", False)

        self.transform_train    = getTransforms_train(self.use_aug)
        self.transform_val_test = getTransforms_val_test()
        
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
        tileName     = self.files[idx]
        filepathImg  = os.path.join(self.imgDirectory, tileName)        
        originalImg  = cv2.imread(filepathImg)
        originalImg  = cv2.cvtColor(originalImg, cv2.COLOR_BGR2RGB)
        copyImg      = copy.deepcopy(originalImg)

        filepathMask = os.path.join(self.maskDirectory, tileName)
        mask         = cv2.imread(filepathMask, cv2.IMREAD_UNCHANGED)
        
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
def getTransforms_train(use_aug):
    resize_h, resize_w = 64, 64

    # << YOUR DATASET NORMALIZATION VALUES >>
    mean = [0.34311995, 0.35981545, 0.33854547]
    std  = [0.10209841, 0.09479538, 0.09755876]

    if not use_aug:
        # return only resize + normalize
        return alb.Compose([
            alb.Resize(resize_h, resize_w),
            alb.Normalize(mean=mean, std=std),
            alb.pytorch.transforms.ToTensorV2()
        ])

    # FULL AUGMENTATION HARDCODED HERE
    return alb.Compose([
        alb.Resize(resize_h, resize_w),

        # GEOMETRY
        alb.HorizontalFlip(p=0.5),
        alb.VerticalFlip(p=0.5),
        alb.RandomRotate90(p=0.5),
        alb.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.4),
        alb.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.1, rotate_limit=10,
            border_mode=cv2.BORDER_REFLECT_101, p=0.4
        ),

        # COLOR
        alb.ColorJitter(brightness=0.2, contrast=0.2,
                        saturation=0.2, hue=0.1, p=0.5),

        # SHADOW (DISCRETE)
        alb.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1),
            num_shadows_limit=(1, 2),
            shadow_half_radius=(0.05, 0.15),
            shadow_dimension=3,
            p=0.2
        ),

        # BLUR (VERY LIGHT)
        alb.GaussianBlur(blur_limit=(1, 2), p=0.10),

        # NORMALIZE & TENSOR
        alb.Normalize(mean=mean, std=std),
        alb.pytorch.transforms.ToTensorV2(),
    ])


def getTransforms_val_test():
    resize_h, resize_w = 64, 64

    # DEFAULT NORMALIZATION FOR VAL/TEST
    # use your aerial dataset stats
    mean = [0.34311995, 0.35981545, 0.33854547]
    std  = [0.10209841, 0.09479538, 0.09755876]

    return alb.Compose([
        alb.Resize(resize_h, resize_w),
        alb.Normalize(mean=mean, std=std),
        alb.pytorch.transforms.ToTensorV2()
    ])
