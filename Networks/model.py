from Dataset.dataLoader import *
from Dataset.makeGraph import *
from Networks.Architectures.basicNetwork import *

import numpy as np
np.random.seed(2885)
import os
import copy

import torch
torch.manual_seed(2885)
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim


# --------------------------------------------------------------------------------
# CREATE A FOLDER IF IT DOES NOT EXIST
# INPUT: 
#     - desiredPath (str): path to the folder to create
# --------------------------------------------------------------------------------
def createFolder(desiredPath): 
    if not os.path.exists(desiredPath):
        os.makedirs(desiredPath)


######################################################################################
#
# CLASS DESCRIBING THE INSTANTIATION, TRAINING AND EVALUATION OF THE MODEL 
# An instance of Network_Class has been created in the main.py file
# 
######################################################################################

class Network_Class: 
    # --------------------------------------------------------------------------------
    # INITIALISATION OF THE MODEL
    # INPUTS: 
    #     - param (dic): dictionnary containing the parameters defined in the 
    #                    configuration (yaml) file
    #     - imgDirectory (str): path to the folder containing the images 
    #     - maskDirectory (str): path to the folder containing the masks
    #     - resultsPath (str): path to the folder containing the results of the 
    #                          experiement
    # --------------------------------------------------------------------------------
    def __init__(self, param, imgDirectory, maskDirectory, resultsPath):
        # ----------------
        # USEFUL VARIABLES 
        # ----------------
        self.imgDirectory  = imgDirectory
        self.maskDirectory = maskDirectory
        self.resultsPath   = resultsPath
        self.epoch         = param["TRAINING"]["EPOCH"]
        self.device        = param["TRAINING"]["DEVICE"]
        self.lr            = param["TRAINING"]["LEARNING_RATE"]
        self.batchSize     = param["TRAINING"]["BATCH_SIZE"]

        # -----------------------------------
        # NETWORK ARCHITECTURE INITIALISATION
        # -----------------------------------
        self.model = ImprovedU_net(param).to(self.device)

        # -------------------
        # TRAINING PARAMETERS
        # -------------------
        self.criterion = nn.CrossEntropyLoss()  # ← AJOUTÉ: pour segmentation multi-classes
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)  # ← AJOUTÉ

        # ----------------------------------------------------
        # DATASET INITIALISATION (from the dataLoader.py file)
        # ----------------------------------------------------
        self.dataSetTrain    = LLNDataset(imgDirectory, maskDirectory, "train", param)
        self.dataSetVal      = LLNDataset(imgDirectory, maskDirectory, "val",   param)
        self.dataSetTest     = LLNDataset(imgDirectory, maskDirectory, "test",  param)
        self.trainDataLoader = DataLoader(self.dataSetTrain, batch_size=self.batchSize, shuffle=True,  num_workers=4)
        self.valDataLoader   = DataLoader(self.dataSetVal,   batch_size=self.batchSize, shuffle=False, num_workers=4)
        self.testDataLoader  = DataLoader(self.dataSetTest,  batch_size=self.batchSize, shuffle=False, num_workers=4)

        # Pour suivre les métriques
        self.train_losses = []
        self.val_losses = []

    # ---------------------------------------------------------------------------
    # LOAD PRETRAINED WEIGHTS (to run evaluation without retraining the model...)
    # ---------------------------------------------------------------------------
    def loadWeights(self): 
        self.model.load_state_dict(torch.load(self.resultsPath + '/_Weights/wghts.pkl', weights_only=True))

    # -----------------------------------
    # TRAINING LOOP (implémentation réelle)
    # -----------------------------------
    def train(self): 
        best_loss = float('inf')
        
        # train for a given number of epochs
        for epoch in range(self.epoch):
            # Training phase
            self.model.train()
            running_loss = 0.0
            
            for batch_idx, (images, masks, _, _) in enumerate(self.trainDataLoader):
                images = images.to(self.device)
                masks = masks.to(self.device).long()  # Conversion en long pour CrossEntropy
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            # Calcul de la loss moyenne sur l'epoch
            avg_train_loss = running_loss / len(self.trainDataLoader)
            self.train_losses.append(avg_train_loss)
            
            # Validation phase
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            print(f"Epoch [{epoch+1}/{self.epoch}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")
            
            # Sauvegarde des meilleurs poids
            if val_loss < best_loss:
                best_loss = val_loss
                modelWts = copy.deepcopy(self.model.state_dict())
        
        # Print learning curves
        self.plot_learning_curves()
        
        # Save the model weights
        wghtsPath  = self.resultsPath + '/_Weights/'
        createFolder(wghtsPath)
        torch.save(modelWts, wghtsPath + '/wghts.pkl')
        print("✅ Training completed and weights saved!")

    def validate(self):
        """Validation step"""
        self.model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for images, masks, _, _ in self.valDataLoader:
                images = images.to(self.device)
                masks = masks.to(self.device).long()
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                running_val_loss += loss.item()
        
        return running_val_loss / len(self.valDataLoader)

    def plot_learning_curves(self):
        """Plot and save learning curves"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True)
        
        curves_path = os.path.join(self.resultsPath, "learning_curves.png")
        plt.savefig(curves_path)
        plt.close()
        print(f"📊 Learning curves saved to {curves_path}")

    # -------------------------------------------------
    # EVALUATION PROCEDURE (implémentation complète)
    # -------------------------------------------------
    def evaluate(self):
        self.model.eval()  # ← CORRIGÉ: utiliser eval() au lieu de train(False)
         
        allMasks, allMasksPreds, allTileNames, allResizedImgs = [], [], [], []
        
        with torch.no_grad():  # ← AJOUTÉ: désactive le calcul des gradients pour l'évaluation
            for (images, masks, tileNames, resizedImgs) in self.testDataLoader:
                images = images.to(self.device)
                outputs = self.model(images)

                # Move back to CPU for processing
                outputs = outputs.to('cpu')
                
                # Get predicted masks (class with highest probability)
                masksPreds = torch.argmax(outputs, dim=1)

                allMasks.extend(masks.data.numpy())
                allMasksPreds.extend(masksPreds.data.numpy())
                allResizedImgs.extend(resizedImgs.data.numpy())
                allTileNames.extend(tileNames)
        
        allMasks = np.array(allMasks)
        allMasksPreds = np.array(allMasksPreds)
        allResizedImgs = np.array(allResizedImgs)
            
        # Qualitative Evaluation
        savePath = os.path.join(self.resultsPath, "Test")
        createFolder(savePath)
        reconstruct_from_tiles(allResizedImgs, allMasksPreds, allMasks, allTileNames, savePath)
    
        # Quantitative Evaluation
        self.quantitative_evaluation(allMasks, allMasksPreds)

    def quantitative_evaluation(self, true_masks, pred_masks):
        """Calculate quantitative metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
        import matplotlib.pyplot as plt
        
        # Flatten arrays for metric calculation
        true_flat = true_masks.flatten()
        pred_flat = pred_masks.flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(true_flat, pred_flat)
        precision = precision_score(true_flat, pred_flat, average='weighted', zero_division=0)
        recall = recall_score(true_flat, pred_flat, average='weighted', zero_division=0)
        f1 = f1_score(true_flat, pred_flat, average='weighted', zero_division=0)
        iou = jaccard_score(true_flat, pred_flat, average='weighted', zero_division=0)
        
        print("\n📈 Quantitative Evaluation Results:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"IoU:       {iou:.4f}")
        
        # Save metrics to file
        metrics_path = os.path.join(self.resultsPath, "test_metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write("Quantitative Evaluation Results:\n")
            f.write(f"Accuracy:  {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall:    {recall:.4f}\n")
            f.write(f"F1-Score:  {f1:.4f}\n")
            f.write(f"IoU:       {iou:.4f}\n")
        
        print(f"✅ Metrics saved to {metrics_path}")  