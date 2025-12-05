import torch
import torch.nn as nn
import torch.nn.functional as F

######################################################################################
#
# FOCAL LOSS - Alternative aux class weights pour gérer le déséquilibre
# Focal Loss (ICCV 2017) : Focus automatique sur exemples difficiles
# SANS pénaliser les classes majoritaires
#
######################################################################################

class FocalLoss(nn.Module):
    """
    Focal Loss pour segmentation sémantique
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    - gamma > 0 : Réduit le poids des exemples faciles
    - alpha : Balance les classes (optionnel)
    
    AVANTAGES vs Class Weights:
    - Focus sur pixels difficiles (frontières, petits objets)
    - Ne pénalise PAS les prédictions correctes des classes majoritaires
    - Pas de confusion entre classes similaires (0 vs 4)
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        INPUTS:
            - alpha (list/tensor): Poids par classe [C0, C1, C2, C3, C4]
                                   Si None, pas de pondération
            - gamma (float): Exposant de modulation (2.0 recommandé)
                            0 = CrossEntropy classique
                            2 = Focus fort sur difficiles
            - reduction (str): 'mean', 'sum', ou 'none'
        """
        super(FocalLoss, self).__init__()
        
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.FloatTensor(alpha)
            else:
                self.alpha = alpha
        else:
            self.alpha = None
            
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        INPUTS:
            - inputs: [B, C, H, W] (logits)
            - targets: [B, H, W] (class indices)
        """
        # CrossEntropy standard
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Probabilités prédites
        p = torch.exp(-ce_loss)  # p_t
        
        # Focal term: (1 - p_t)^gamma
        focal_weight = (1 - p) ** self.gamma
        
        # Focal Loss = focal_weight * CE
        focal_loss = focal_weight * ce_loss
        
        # Alpha weighting (optionnel)
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            
            # Gather alpha pour chaque pixel selon sa vraie classe
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


######################################################################################
#
# DICE LOSS - Complémentaire à Focal Loss pour la segmentation
# Optimise directement l'IoU
#
######################################################################################

class DiceLoss(nn.Module):
    """
    Dice Loss pour segmentation
    
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    Loss = 1 - Dice
    
    AVANTAGES:
    - Optimise directement l'IoU
    - Gère naturellement le déséquilibre
    - Complémentaire à Focal Loss
    """
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        """
        INPUTS:
            - inputs: [B, C, H, W] (logits)
            - targets: [B, H, W] (class indices)
        """
        num_classes = inputs.size(1)
        
        # Softmax sur les logits
        probs = F.softmax(inputs, dim=1)
        
        # One-hot encoding des targets
        targets_one_hot = F.one_hot(targets, num_classes)  # [B, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
        
        # Dice par classe
        dice_scores = []
        for c in range(num_classes):
            pred_c = probs[:, c]
            target_c = targets_one_hot[:, c]
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        # Moyenne des Dice
        dice_mean = torch.stack(dice_scores).mean()
        
        return 1.0 - dice_mean


######################################################################################
#
# COMBINED LOSS - Focal + Dice
# Recommandé pour segmentation déséquilibrée
#
######################################################################################

class CombinedLoss(nn.Module):
    """
    Focal Loss + Dice Loss
    
    Loss = alpha * Focal + (1-alpha) * Dice
    
    RECOMMANDATIONS:
    - alpha=0.7 : Plus de focus sur Focal (pixel-wise accuracy)
    - alpha=0.5 : Balance entre les deux
    """
    
    def __init__(self, focal_alpha=None, focal_gamma=2.0, dice_weight=0.3):
        super(CombinedLoss, self).__init__()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice = DiceLoss()
        self.dice_weight = dice_weight
        
    def forward(self, inputs, targets):
        focal_loss = self.focal(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        
        return (1 - self.dice_weight) * focal_loss + self.dice_weight * dice_loss


######################################################################################
#
# HELPER FUNCTION - Récupérer la loss depuis config
#
######################################################################################

def get_loss_from_config(param, class_weights=None):
    """
    Crée la fonction de loss depuis la config YAML
    
    INPUTS:
        - param: Config YAML
        - class_weights: Poids des classes (pour CrossEntropy/Focal)
    
    OUTPUTS:
        - criterion: Loss function
    """
    loss_type = param["TRAINING"].get("LOSS_TYPE", "crossentropy")
    
    if loss_type == "crossentropy":
        if class_weights is not None:
            return nn.CrossEntropyLoss(weight=class_weights)
        else:
            return nn.CrossEntropyLoss()
            
    elif loss_type == "focal":
        gamma = param["TRAINING"].get("FOCAL_GAMMA", 2.0)
        return FocalLoss(alpha=class_weights, gamma=gamma)
        
    elif loss_type == "dice":
        return DiceLoss()
        
    elif loss_type == "combined":
        gamma = param["TRAINING"].get("FOCAL_GAMMA", 2.0)
        dice_weight = param["TRAINING"].get("DICE_WEIGHT", 0.3)
        return CombinedLoss(
            focal_alpha=class_weights,
            focal_gamma=gamma,
            dice_weight=dice_weight
        )
        
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
