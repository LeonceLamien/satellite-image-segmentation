# 📊 Comparaison des Configurations - Stratégie Anti-Overfitting

## 🎯 Objectif
Réduire l'overfitting observé dans LargeExp (gap train-val de 0.10)

---

## 📈 Résultats Actuels

### DefaultExp (Baseline)
- **Train Loss**: 0.3681
- **Val Loss**: 0.4976
- **Gap**: ~0.13
- **Test Accuracy**: Non testé
- **Configuration**: 
  - Backbone: 64 channels
  - ASPP: 256 channels
  - Dropout: 0.2
  - Optimizer: Adam (LR=0.001)
  - Epochs: 30

### LargeExp (Large Model)
- **Train Loss**: 0.3334
- **Val Loss**: 0.4342
- **Gap**: ~0.10 ⚠️ OVERFITTING
- **Test Accuracy**: 0.7142
- **Test IoU**: 0.5745
- **Configuration**:
  - Backbone: 256 channels (4x plus gros)
  - ASPP: 512 channels
  - Dropout: 0.1 (trop bas!)
  - Weight Decay: 0.0 (aucune régularisation L2)
  - Optimizer: Adam (LR=0.0005)
  - Epochs: 30
  - **Problème**: Le modèle mémorise le train set

---

## 🔧 Solution: LargeExp_Regularized

### Techniques de Régularisation Appliquées

#### 1. **Data Augmentation** (Principal changement)
```yaml
USE_AUGMENTATION: TRUE
HORIZONTAL_FLIP: 0.5      # Flip horizontal 50% du temps
VERTICAL_FLIP: 0.3        # Flip vertical 30% du temps
ROTATE_LIMIT: 25          # Rotation jusqu'à ±25°
SHIFT_SCALE_ROTATE: 0.5   # Translation + zoom + rotation
COLOR_JITTER: 0.6         # Variation couleur/contraste
HUE_SAT_VALUE: 0.6        # Variation teinte/saturation
BRIGHTNESS_CONTRAST: 0.5  # Variation luminosité
GAUSSIAN_BLUR: 0.3        # Flou gaussien léger
GAUSSIAN_NOISE: 0.2       # Bruit gaussien
```

**Impact attendu**: Force le modèle à apprendre des features invariantes aux transformations, réduit la mémorisation.

#### 2. **Dropout Augmenté**
```yaml
DROPOUT: 0.3          # 0.1 → 0.3 (3x plus)
ASPP_DROPOUT: 0.5     # 0.3 → 0.5
```

**Impact**: Empêche la co-adaptation des neurones, force la redondance.

#### 3. **L2 Regularization (Weight Decay)**
```yaml
WEIGHT_DECAY: 0.0001  # 0.0 → 0.0001
OPTIMIZER: AdamW      # Adam → AdamW (meilleure implémentation du weight decay)
```

**Impact**: Pénalise les poids élevés, préfère les solutions plus simples.

#### 4. **Class Weights**
```yaml
CLASS_WEIGHTS: [1.0, 3.0, 1.5, 1.2, 1.0]
```

**Rationale**:
- Class 0 (36%): weight=1.0 (classe majoritaire)
- Class 1 (0.84%): weight=3.0 (classe très minoritaire, recall=49%)
- Class 2 (6%): weight=1.5 (minoritaire)
- Class 3 (13%): weight=1.2 (légèrement sous-représentée)
- Class 4 (44%): weight=1.0 (classe majoritaire)

**Impact**: Améliore le recall des classes minoritaires sans perdre trop sur les majoritaires.

#### 5. **Learning Rate Ajusté**
```yaml
LEARNING_RATE: 0.0003  # 0.0005 → 0.0003 (40% plus bas)
```

**Impact**: Apprentissage plus stable avec augmentation, évite l'overfitting rapide.

#### 6. **Plus d'Epochs + Patience**
```yaml
EPOCH: 40           # 30 → 40
PATIENCE: 12        # 10 → 12
```

**Impact**: Donne plus de temps au modèle pour apprendre avec augmentation (qui ralentit la convergence).

---

## 📊 Résultats Attendus

### LargeExp_Regularized (Prédictions)
- **Train Loss**: ~0.38-0.40 (légèrement plus élevé que LargeExp)
- **Val Loss**: ~0.40-0.42 (devrait rester proche)
- **Gap**: ~0.02-0.04 ✓ (Objectif: < 0.05)
- **Test Accuracy**: ~0.73-0.75 (amélioration de +1-3%)
- **Test IoU**: ~0.58-0.60 (amélioration de +1-3%)
- **Class 1 Recall**: ~0.55-0.60 (amélioration grâce aux class weights)

### Métriques de Succès
✓ **Gap train-val < 0.05**: Overfitting résolu
✓ **Test Accuracy > 0.72**: Meilleure généralisation
✓ **Class 1 Recall > 0.50**: Classes minoritaires améliorées
✓ **Val loss stable**: Pas de divergence après epoch 15

---

## 🚀 Plan d'Exécution

### Étape 1: Entraîner LargeExp_Regularized
```bash
python main.py --config Todo_List/LargeExp_Regularized.yaml
```

### Étape 2: Analyser les Résultats
Comparer les learning curves:
- Si gap < 0.05 et test accuracy > LargeExp → ✓ Succès
- Si gap encore élevé → Augmenter encore dropout ou augmentation
- Si val loss trop élevé → Réduire regularization

### Étape 3: Fine-tuning (si nécessaire)
Ajustements possibles:
- **Si underfitting**: Réduire dropout, weight_decay, ou augmentation
- **Si encore overfitting**: Augmenter dropout (0.4), weight_decay (0.0005), ou augmentation
- **Si Class 1 encore faible**: Augmenter CLASS_WEIGHTS[1] à 4.0 ou 5.0

---

## 📝 Notes Techniques

### Data Augmentation Best Practices
- ✓ Geometric transforms (flip, rotate): Préservent la structure sémantique
- ✓ Color jitter: Robustesse aux variations d'illumination
- ⚠️ Gaussian noise: Attention au niveau (trop = détruit l'information)
- ⚠️ Gaussian blur: Utile mais modéré (risque de perdre les détails fins)

### Dropout Strategy
- Encoder/Decoder: 0.3 (élevé pour éviter co-adaptation)
- ASPP: 0.5 (très élevé car c'est le bottleneck)
- Final layers: 0.1-0.2 (plus bas pour préserver l'information)

### Weight Decay Guidelines
- 0.0001: Standard pour vision tasks
- 0.0005-0.001: Si overfitting persistant
- 0.00001: Si underfitting

---

## 🎯 Conclusion

**Philosophie**: On sacrifie un peu de performance sur le train set (0.33 → 0.38) pour gagner en généralisation sur test/val.

**Espoir**: Réduire le gap de 0.10 → 0.03-0.04 tout en maintenant ou améliorant le test accuracy.

**Prochaine Étape**: Lance `LargeExp_Regularized.yaml` et observe si le modèle généralise mieux ! 🚀
