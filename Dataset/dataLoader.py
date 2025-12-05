  
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

    mean = [0.324276, 0.338223, 0.318705]
    std  = [0.135364, 0.124776, 0.124514]

    if not use_aug:
        return alb.Compose([
            alb.Resize(resize_h, resize_w),
            alb.Normalize(mean=mean, std=std),
            alb.pytorch.transforms.ToTensorV2()
        ])

    # OPTIMIZED AUGMENTATION (léger mais efficace)
    return alb.Compose([
        # 1. RESIZE FIRST
        alb.Resize(resize_h, resize_w),
        
        # 2. GEOMETRIC AUGMENTATIONS (léger)
        alb.HorizontalFlip(p=0.5),
        alb.VerticalFlip(p=0.5),
        alb.RandomRotate90(p=0.4),  # ✅ Réduit de 0.5
        
        alb.Rotate(
            limit=10,  # ✅ Réduit de 15° à 10°
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.3  # ✅ Réduit de 0.4
        ),
        
        alb.ShiftScaleRotate(
            shift_limit=0.08,  # ✅ Réduit de 0.1
            scale_limit=0.08,  # ✅ Réduit de 0.1
            rotate_limit=8,    # ✅ Réduit de 10°
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.3
        ),
        
        # 3. COLOR AUGMENTATIONS (très léger)
        alb.ColorJitter(
            brightness=0.15,  # ✅ Réduit de 0.2
            contrast=0.15,    # ✅ Réduit de 0.2
            saturation=0.1,   # ✅ Réduit de 0.15
            hue=0.03,         # ✅ Réduit de 0.05
            p=0.4             # ✅ Réduit de 0.5
        ),
        
        # 4. BLUR (atmospheric effects, très léger)
        alb.GaussianBlur(
            blur_limit=(3, 5),  # ✅ Réduit de (3,7)
            p=0.15              # ✅ Réduit de 0.2
        ),
        
        # 5. NORMALIZE & CONVERT TO TENSOR
        alb.Normalize(mean=mean, std=std),
        alb.pytorch.transforms.ToTensorV2(),
    ])


def getTransforms_val_test():
    resize_h, resize_w = 64, 64

    mean = [0.324276, 0.338223, 0.318705]
    std  = [0.135364, 0.124776, 0.124514]

    return alb.Compose([
        alb.Resize(resize_h, resize_w),
        alb.Normalize(mean=mean, std=std),
        alb.pytorch.transforms.ToTensorV2()
    ])
