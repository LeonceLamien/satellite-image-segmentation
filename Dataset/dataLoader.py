  
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
#
# Do not hesitate to consider other useful transforms!
# -----------------------------------------------------------------------
def getTransforms_train(param): 
    # Extract parameters from YAML configuration
    resize_h = int(param["DATASET"]["RESIZE_SHAPE"].split("x")[0])
    resize_w = int(param["DATASET"]["RESIZE_SHAPE"].split("x")[1])
    mean = param["DATASET"].get("NORMALIZE_MEAN", [0.485, 0.456, 0.406])
    std = param["DATASET"].get("NORMALIZE_STD", [0.229, 0.224, 0.225])
    
    # Extract augmentation parameters (with default values)
    aug_params = param["DATASET"].get("AUGMENTATION", {})
    
    imgTransformsList = [
        # 1. Resize first (before augmentations)
        alb.Resize(height=resize_h, width=resize_w),
        
        # 2. GEOMETRIC AUGMENTATIONS (most important for segmentation)
        alb.HorizontalFlip(p=aug_params.get("HORIZONTAL_FLIP", 0.5)),
        alb.VerticalFlip(p=aug_params.get("VERTICAL_FLIP", 0.5)),
        
        # Multiple of 90° rotations (preserves grid for aerial images)
        alb.RandomRotate90(p=aug_params.get("ROTATE90", 0.5)),
        
        # Small free rotations (simulates slight camera misalignment)
        alb.Rotate(
            limit=aug_params.get("ROTATION_LIMIT", 15),
            border_mode=cv2.BORDER_CONSTANT,
            p=aug_params.get("ROTATION_PROB", 0.4)
        ),
        
        # Combined shift, scale, rotate (simulates camera zoom/displacement)
        alb.ShiftScaleRotate(
            shift_limit=aug_params.get("SHIFT_LIMIT", 0.1),
            scale_limit=aug_params.get("SCALE_LIMIT", 0.1),
            rotate_limit=aug_params.get("SHIFT_SCALE_ROTATE_LIMIT", 10),
            border_mode=cv2.BORDER_CONSTANT,
            p=aug_params.get("SHIFT_SCALE_ROTATE_PROB", 0.4)
        ),
        
        # 3. COLOR AUGMENTATIONS (simulates weather/lighting/season conditions)
        alb.ColorJitter(
            brightness=aug_params.get("BRIGHTNESS", 0.2),
            contrast=aug_params.get("CONTRAST", 0.2),
            saturation=aug_params.get("SATURATION", 0.2),
            hue=aug_params.get("HUE", 0.1),
            p=aug_params.get("COLOR_JITTER_PROB", 0.5)
        ),
        
        # 4. OPTIONAL AUGMENTATIONS (elastic deformations, occlusions, shadows)
        # Elastic transform - simulates slight terrain deformations
        alb.ElasticTransform(
            alpha=1,
            sigma=50,
            border_mode=cv2.BORDER_CONSTANT,
            p=aug_params.get("ELASTIC_PROB", 0.0)
        ),
        
        # Coarse Dropout - simulates clouds, shadows, occlusions
        alb.CoarseDropout(
            max_holes=aug_params.get("COARSE_DROPOUT_HOLES", 4),
            max_height=int(resize_h * aug_params.get("COARSE_DROPOUT_SIZE", 0.05)),
            max_width=int(resize_w * aug_params.get("COARSE_DROPOUT_SIZE", 0.05)),
            p=aug_params.get("COARSE_DROPOUT_PROB", 0.0)
        ),
        
        # Random Shadow - simulates cloud shadows on aerial images
        alb.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1),
            num_shadows_limit=(1, 2),
            shadow_dimension=5,
            p=aug_params.get("SHADOW_PROB", 0.0)
        ),
        
        # 5. NOISE AUGMENTATIONS (simulates sensor quality)
        alb.GaussNoise(
            var_limit=(aug_params.get("GAUSS_NOISE_VAR", [10.0, 50.0])[0], 
                       aug_params.get("GAUSS_NOISE_VAR", [10.0, 50.0])[1]),
            p=aug_params.get("GAUSS_NOISE_PROB", 0.2)
        ),
        
        alb.GaussianBlur(
            blur_limit=aug_params.get("BLUR_LIMIT", (3, 3)),
            p=aug_params.get("BLUR_PROB", 0.15)
        ),
        
        # 6. NORMALIZATION AND CONVERSION (always last)
        alb.Normalize(mean=mean, std=std), 
        alb.pytorch.transforms.ToTensorV2(), 
    ]
    
    return alb.Compose(imgTransformsList)

def getTransforms_val_test(param): 
    resize_h = int(param["DATASET"]["RESIZE_SHAPE"].split("x")[0])
    resize_w = int(param["DATASET"]["RESIZE_SHAPE"].split("x")[1])
    mean = param["DATASET"].get("NORMALIZE_MEAN", [0.485, 0.456, 0.406])
    std = param["DATASET"].get("NORMALIZE_STD", [0.229, 0.224, 0.225])
    
    imgTransformsList = [
        alb.Resize(height=resize_h, width=resize_w), 
        alb.Normalize(mean=mean, std=std),  
        alb.pytorch.transforms.ToTensorV2(), 
    ]
    return alb.Compose(imgTransformsList)
