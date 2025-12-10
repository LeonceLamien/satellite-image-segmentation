from Dataset.dataLoader import *
from Dataset.makeGraph import *
from Networks.Architectures.basicNetwork import *
from Networks.Architectures.improvedUNet import *
from Networks.Architectures.deepLab import *

from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from sklearn.preprocessing import label_binarize
        
import numpy as np
np.random.seed(2885)

import matplotlib.pyplot as plt
import seaborn as sns
import os
import copy

import torch
torch.manual_seed(2885)
from torch.utils.data import DataLoader
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

class FocalLoss(nn.Module):
    """Focal Loss pour gérer le déséquilibre des classes"""
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Poids des classes
        self.gamma = gamma  # Paramètre de focalisation
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

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
        print("resultsPath", resultsPath)
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
        self.optimizer_name = param["TRAINING"].get("OPTIMIZER", "Adam")
        self.early_stopping = param["TRAINING"].get("EARLY_STOPPING", False)
        self.patience = param["TRAINING"].get("PATIENCE_EARLY_STOPPING", 10)
        self.weight_decay = param["TRAINING"].get("WEIGHT_DECAY", 0.0)
        self.loss_function = param["TRAINING"].get("LOSS_FUNCTION", "CrossEntropy")
        self.use_scheduler = param["TRAINING"].get("LR_SCHEDULER", None)

        # -----------------------------------
        # NETWORK ARCHITECTURE INITIALISATION
        # -----------------------------------
        model_arch = param["MODEL"].get("ARCHITECTURE", "DeepLabV3Plus")
        if model_arch == "DeepLabV3Plus":
            self.model = DeepLabV3Plus(param).to(self.device)
        elif model_arch == "ImprovedUNet":
            self.model = ImprovedUNet(param).to(self.device)
        else:
            print(f"Unknown Model {model_arch}")
            self.model = None

        # -------------------
        # TRAINING PARAMETERS
        # -------------------
        # Loss function
        if self.loss_function == "FocalLoss":
            focal_gamma = param["TRAINING"].get("FOCAL_GAMMA", 2.0)
            class_weights = param["TRAINING"].get("CLASS_WEIGHTS", None)
            if class_weights:
                class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            self.criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
            print(f"Using Focal Loss (gamma={focal_gamma}, weights={class_weights})")
        else:
            class_weights = param["TRAINING"].get("CLASS_WEIGHTS", None)
            if class_weights:
                class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"Using CrossEntropy Loss (weights={class_weights})")
        
        # Optimizer with weight decay
        if self.optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == "SDG":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            print(f"Unknown Optimizer {self.optimizer_name}")
            self.optimizer = None
        
        # Learning rate scheduler
        self.scheduler = None
        if self.use_scheduler == "ReduceLROnPlateau":
            lr_patience = param["TRAINING"].get("LR_PATIENCE", 5)
            lr_factor = param["TRAINING"].get("LR_FACTOR", 0.5)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=lr_patience, factor=lr_factor, verbose=True
            )
            print(f"Using ReduceLROnPlateau scheduler (patience={lr_patience}, factor={lr_factor})")

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
    # TRAINING LOOP
    # -----------------------------------
    def train(self): 
        best_loss = float('inf')
        patience_counter = 0
        
        # train for a given number of epochs
        for epoch in range(self.epoch):
            # Training phase
            self.model.train()
            running_loss = 0.0
            
            for _, (images, masks, _, _) in enumerate(self.trainDataLoader):
                images = images.to(self.device)
                masks = masks.to(self.device).long()  
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            avg_train_loss = running_loss / len(self.trainDataLoader)
            self.train_losses.append(avg_train_loss)
            
            # Validation phase
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            print(f"Epoch [{epoch+1}/{self.epoch}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                modelWts = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Early stopping check
            if self.early_stopping and patience_counter >= self.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs (patience: {self.patience})")
                print(f"Best validation loss: {best_loss:.4f}")
                break
        
        # Print learning curves
        self.plot_learning_curves()
        
        # Save the model weights
        wghtsPath  = self.resultsPath + '/_Weights/'
        createFolder(wghtsPath)
        torch.save(modelWts, wghtsPath + '/wghts.pkl')
        print("Training completed and weights saved!")

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
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True)
        
        createFolder(self.resultsPath)
        curves_path = os.path.join(self.resultsPath, "learning_curves.png")
        plt.savefig(curves_path)
        plt.close()
        print(f"Learning curves saved to {curves_path}")

    # -------------------------------------------------
    # EVALUATION PROCEDURE 
    # -------------------------------------------------
    def evaluate(self):
        createFolder(self.resultsPath)

        self.model.eval()
        
        allMasks, allMasksPreds, allTileNames, allResizedImgs = [], [], [], []
        allOutputs = [] 
        
        with torch.no_grad():
            for (images, masks, tileNames, resizedImgs) in self.testDataLoader:
                images = images.to(self.device)
                outputs = self.model(images)

                # Move back to CPU for processing
                outputs = outputs.to('cpu')
                
                # Store raw outputs for ROC AUC calculation
                allOutputs.extend(outputs.data.numpy())  
                
                # Get predicted masks (class with highest probability)
                masksPreds = torch.argmax(outputs, dim=1)

                allMasks.extend(masks.data.numpy())
                allMasksPreds.extend(masksPreds.data.numpy())
                allResizedImgs.extend(resizedImgs.data.numpy())
                allTileNames.extend(tileNames)
        
        allMasks = np.array(allMasks)
        allMasksPreds = np.array(allMasksPreds)
        allResizedImgs = np.array(allResizedImgs)
        allOutputs = np.array(allOutputs)  
            
        # Qualitative Evaluation
        savePath = os.path.join(self.resultsPath, "Test")
        createFolder(savePath)
        reconstruct_from_tiles(allResizedImgs, allMasksPreds, allMasks, allTileNames, savePath)
 
        # Quantitative Evaluation
        self.quantitative_evaluation(allMasks, allMasksPreds, allOutputs)  

        
    def quantitative_evaluation(self, true_masks, pred_masks, outputs=None):
        """Calculate quantitative metrics including ROC AUC and Confusion Matrix"""        
        # Flatten arrays for metric calculation
        true_flat = true_masks.flatten()
        pred_flat = pred_masks.flatten()
        
        # Calculate basic metrics
        accuracy = accuracy_score(true_flat, pred_flat)
        precision = precision_score(true_flat, pred_flat, average='weighted', zero_division=0)
        recall = recall_score(true_flat, pred_flat, average='weighted', zero_division=0)
        f1 = f1_score(true_flat, pred_flat, average='weighted', zero_division=0)
        iou = jaccard_score(true_flat, pred_flat, average='weighted', zero_division=0)
        
        print("\n Quantitative Evaluation Results:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"IoU:       {iou:.4f}")
        
        # Calculate and plot confusion matrix
        self.plot_confusion_matrix(true_flat, pred_flat)
        
        self.plot_normalized_confusion_matrix(true_flat, pred_flat)
        
        # Calculate per-class metrics
        self.calculate_per_class_metrics(true_flat, pred_flat)
        
        # ROC AUC Calculation
        if outputs is not None:
            print("\nCalculating ROC AUC...")
            
            # Reshape outputs and true masks for ROC calculation
            # outputs shape: [N, 5, H, W] -> [N*H*W, 5]
            outputs_flat = outputs.transpose(0, 2, 3, 1).reshape(-1, 5)  # [N*H*W, 5]
            true_flat_roc = true_masks.flatten()  # [N*H*W]
            
            # Binarize true labels for multi-class ROC
            n_classes = 5
            true_binary = label_binarize(true_flat_roc, classes=range(n_classes))
            
            # Calculate ROC AUC for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(true_binary[:, i], outputs_flat[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(true_binary.ravel(), outputs_flat.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            
            # Compute macro-average ROC curve and ROC area
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= n_classes
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            
            # Print ROC AUC results
            print("\n🎯 ROC AUC Results:")
            for i in range(n_classes):
                print(f"Class {i} AUC: {roc_auc[i]:.4f}")
            print(f"Micro-average AUC: {roc_auc['micro']:.4f}")
            print(f"Macro-average AUC: {roc_auc['macro']:.4f}")
            
            # Plot ROC curves
            self.plot_roc_curves(fpr, tpr, roc_auc, n_classes)
            
            # Add ROC AUC to metrics file
            metrics_path = os.path.join(self.resultsPath, "test_metrics.txt")
            with open(metrics_path, 'w') as f:
                f.write("Quantitative Evaluation Results:\n")
                f.write(f"Accuracy:  {accuracy:.4f}\n")
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall:    {recall:.4f}\n")
                f.write(f"F1-Score:  {f1:.4f}\n")
                f.write(f"IoU:       {iou:.4f}\n")
                f.write("\nROC AUC Results:\n")
                for i in range(n_classes):
                    f.write(f"Class {i} AUC: {roc_auc[i]:.4f}\n")
                f.write(f"Micro-average AUC: {roc_auc['micro']:.4f}\n")
                f.write(f"Macro-average AUC: {roc_auc['macro']:.4f}\n")
            
        else:
            # Save basic metrics to file (if no outputs provided)
            metrics_path = os.path.join(self.resultsPath, "test_metrics.txt")
            with open(metrics_path, 'w') as f:
                f.write("Quantitative Evaluation Results:\n")
                f.write(f"Accuracy:  {accuracy:.4f}\n")
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall:    {recall:.4f}\n")
                f.write(f"F1-Score:  {f1:.4f}\n")
                f.write(f"IoU:       {iou:.4f}\n")
        
        print(f" Metrics saved to {metrics_path}")

    def calculate_per_class_metrics(self, true_labels, pred_labels, class_names=None):
        """Calculate and display per-class metrics"""
        if class_names is None:
            class_names = [f'Class {i}' for i in range(5)]
        
        # Calculate per-class metrics
        precision_per_class = precision_score(true_labels, pred_labels, average=None, zero_division=0)
        recall_per_class = recall_score(true_labels, pred_labels, average=None, zero_division=0)
        f1_per_class = f1_score(true_labels, pred_labels, average=None, zero_division=0)
        iou_per_class = jaccard_score(true_labels, pred_labels, average=None, zero_division=0)
        
        print("\n🎯 Per-Class Metrics:")
        print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'IoU':<10}")
        print("-" * 50)
        for i in range(len(class_names)):
            print(f"{class_names[i]:<10} {precision_per_class[i]:<10.4f} {recall_per_class[i]:<10.4f} {f1_per_class[i]:<10.4f} {iou_per_class[i]:<10.4f}")
        
        # Save per-class metrics to file
        metrics_path = os.path.join(self.resultsPath, "per_class_metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write("Per-Class Metrics:\n")
            f.write(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'IoU':<10}\n")
            f.write("-" * 50 + "\n")
            for i in range(len(class_names)):
                f.write(f"{class_names[i]:<10} {precision_per_class[i]:<10.4f} {recall_per_class[i]:<10.4f} {f1_per_class[i]:<10.4f} {iou_per_class[i]:<10.4f}\n")
        
        print(f"Per-class metrics saved to {metrics_path}")

    def plot_confusion_matrix(self, true_labels, pred_labels, class_names=None):
        """Plot and save confusion matrix"""
        if class_names is None:
            class_names = [f'Class {i}' for i in range(5)]
        
        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot confusion matrix with absolute counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, 
                    yticklabels=class_names,
                    cbar_kws={'label': 'Number of Pixels'})
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Labels', fontsize=12)
        plt.ylabel('True Labels', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save the plot
        cm_path = os.path.join(self.resultsPath, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {cm_path}")
        
    def plot_normalized_confusion_matrix(self, true_labels, pred_labels, class_names=None):
        """Plot normalized confusion matrix (optional)"""
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(5)]
        
        # Calculate normalized confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot normalized confusion matrix
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, 
                    yticklabels=class_names,
                    cbar_kws={'label': 'Proportion'})
        
        plt.title('Normalized Confusion Matrix (by True Labels)', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Labels', fontsize=12)
        plt.ylabel('True Labels', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save the plot
        cm_norm_path = os.path.join(self.resultsPath, "confusion_matrix_normalized.png")
        plt.savefig(cm_norm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Normalized confusion matrix saved to {cm_norm_path}")
        
    
    def plot_roc_curves(self, fpr, tpr, roc_auc, n_classes):
        """Plot ROC curves for all classes"""        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot individual classes
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
        for i, color in zip(range(n_classes), colors):
            ax1.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
        
        ax1.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves - Individual Classes')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        
        # Plot micro and macro averages
        ax2.plot(fpr["micro"], tpr["micro"],
                label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})',
                color='deeppink', linestyle=':', linewidth=4)
        
        ax2.plot(fpr["macro"], tpr["macro"],
                label=f'Macro-average (AUC = {roc_auc["macro"]:.2f})',
                color='navy', linestyle=':', linewidth=4)
        
        for i in range(n_classes):
            ax2.plot(fpr[i], tpr[i], lw=1, alpha=0.3)
        
        ax2.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curves - Average Methods')
        ax2.legend(loc="lower right")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        roc_path = os.path.join(self.resultsPath, "roc_curves.png")
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ROC curves saved to {roc_path}")
