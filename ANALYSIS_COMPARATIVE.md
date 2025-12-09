# 📊 Analyse Comparative des Configurations

## 🎯 Résultats Complets

### Configuration 1: DefaultExp (Baseline - Petit Modèle)
```yaml
Backbone: 64 channels
Dropout: 0.2
Weight Decay: 0.0
Augmentation: OFF
```

**Résultats**:
- ✅ **Accuracy: 0.7341** (meilleur)
- ✅ **IoU: 0.5919** (meilleur)
- ✅ **Micro AUC: 0.9384** (meilleur)
- ✅ **Macro AUC: 0.9006** (meilleur)
- Train Loss: 0.3681
- Val Loss: 0.4976
- **Gap: 0.13** (overfitting modéré)

**Per-Class Performance**:
- Class 0: Non spécifié
- Class 1: Non spécifié (probablement faible)
- Class 4: Recall ~0.60

---

### Configuration 2: LargeExp (Large Model Sans Régularisation)
```yaml
Backbone: 256 channels
Dropout: 0.1
Weight Decay: 0.0
Augmentation: OFF
```

**Résultats**:
- Accuracy: 0.7142
- IoU: 0.5745
- Micro AUC: 0.9025
- Macro AUC: 0.8723
- **Train Loss: 0.3334** (très bas)
- **Val Loss: 0.4342**
- **Gap: 0.10** ⚠️ OVERFITTING

**Per-Class Performance**:
- Class 0: Recall 0.7489
- Class 1: Recall 0.4933 (faible)
- Class 2: Recall 0.6653
- Class 3: Recall 0.8098
- Class 4: Recall 0.6012

---

### Configuration 3: LargeExp_Regularized (Régularisation Trop Forte)
```yaml
Backbone: 256 channels
Dropout: 0.3
Weight Decay: 0.0001
Augmentation: STRONG (8 transforms)
Class Weights: [1.0, 3.0, 1.5, 1.2, 1.0]
```

**Résultats**:
- ❌ **Accuracy: 0.7278** (-0.63% vs DefaultExp)
- ❌ **IoU: 0.5743** (-1.76% vs DefaultExp)
- ❌ **Micro AUC: 0.9199** (-1.85% vs DefaultExp)
- ❌ **Macro AUC: 0.8652** (-3.54% vs DefaultExp)
- **Train Loss: 0.6108** (trop élevé!)
- **Val Loss: 0.5301**
- **Gap: 0.08** (meilleur mais underfitting)

**Per-Class Performance**:
- Class 0: Recall 0.8338 ✓ (excellent)
- Class 1: Recall 0.3979 ❌ (pire que LargeExp!)
- Class 2: Recall 0.6362
- Class 3: Recall 0.6915
- Class 4: Recall 0.4472 ❌❌ (EFFONDREMENT!)

**Problèmes Identifiés**:
1. 🔴 **Régularisation trop forte**: Train loss trop élevé (0.61)
2. 🔴 **Class 4 effondrée**: Recall 0.60 → 0.44 (-26%)
3. 🔴 **Class weights déséquilibrés**: Favorisent trop Class 0 au détriment de Class 4
4. 🔴 **Augmentation trop agressive**: 8 transforms empêchent l'apprentissage

---

## 🔍 Diagnostic Détaillé

### Problème 1: Régularisation Excessive

**Signes**:
- Train loss élevé (0.61 vs 0.33 dans LargeExp)
- Écart train-val réduit (0.08) mais au prix de la performance
- Test accuracy PIRE que le petit modèle

**Cause**:
```yaml
DROPOUT: 0.3              # Trop élevé
ASPP_DROPOUT: 0.5         # Trop élevé
WEIGHT_DECAY: 0.0001      # Trop élevé
+ 8 augmentations fortes  # Trop agressif
```

**Effet**: Le modèle ne peut pas **apprendre** correctement, même les patterns simples.

---

### Problème 2: Class Weights Déséquilibrés

**Configuration précédente**:
```yaml
CLASS_WEIGHTS: [1.0, 3.0, 1.5, 1.2, 1.0]
#               C0   C1   C2   C3   C4
```

**Résultats**:
- Class 0 (36% dataset, weight=1.0): Recall 0.75 → **0.83** ✓ (amélioration)
- Class 1 (0.84% dataset, weight=3.0): Recall 0.49 → **0.40** ❌ (dégradation!)
- Class 4 (44% dataset, weight=1.0): Recall 0.60 → **0.45** ❌❌ (effondrement!)

**Analyse**:
- Class 0 et Class 4 sont toutes deux majoritaires (36% + 44% = 80% du dataset)
- Les pénaliser également (weight=1.0) crée un déséquilibre
- Class 0 bénéficie du boost des minoritaires, Class 4 souffre

---

### Problème 3: Augmentation Trop Agressive

**8 transforms actifs**:
1. HorizontalFlip (0.5)
2. VerticalFlip (0.3)
3. Rotate (±25°)
4. ShiftScaleRotate (0.5)
5. ColorJitter (0.6)
6. HueSaturationValue (0.6)
7. BrightnessContrast (0.5)
8. GaussianBlur (0.3)
9. GaussianNoise (0.2)

**Probabilité cumulative**: Une image a ~90% de chances de subir au moins 3-4 transformations!

**Effet**: 
- Les images sont trop déformées
- Le modèle perd du temps à apprendre des variations extrêmes
- Convergence lente (40 epochs insuffisants)

---

## ✅ Solution: LargeExp_Balanced

### Stratégie Équilibrée

#### 1. **Régularisation MODÉRÉE**
```yaml
DROPOUT: 0.2              # 0.3 → 0.2 (entre 0.1 et 0.3)
ASPP_DROPOUT: 0.4         # 0.5 → 0.4
WEIGHT_DECAY: 0.00005     # 0.0001 → 0.00005 (divisé par 2)
```

#### 2. **Augmentation RÉDUITE**
```yaml
ROTATE_LIMIT: 15          # 25 → 15
SHIFT_SCALE_ROTATE: 0.3   # 0.5 → 0.3
COLOR_JITTER: 0.4         # 0.6 → 0.4
HUE_SAT_VALUE: 0.4        # 0.6 → 0.4
BRIGHTNESS_CONTRAST: 0.3  # 0.5 → 0.3
GAUSSIAN_BLUR: 0.15       # 0.3 → 0.15
GAUSSIAN_NOISE: 0.1       # 0.2 → 0.1
```

#### 3. **Class Weights AJUSTÉS**
```yaml
CLASS_WEIGHTS: [0.9, 4.0, 1.3, 1.1, 0.8]
#               C0   C1   C2   C3   C4
```

**Rationale**:
- **Class 0** (36%): 1.0 → **0.9** (légèrement réduit, déjà performante)
- **Class 1** (0.84%): 3.0 → **4.0** (augmenté, cible prioritaire)
- **Class 2** (6%): 1.5 → **1.3** (légèrement réduit)
- **Class 3** (13%): 1.2 → **1.1** (légèrement réduit)
- **Class 4** (44%): 1.0 → **0.8** (RÉDUIT pour éviter l'effondrement)

**Objectif**: Équilibrer les 5 classes sans sacrifier Class 4.

#### 4. **Hyperparamètres Optimisés**
```yaml
LEARNING_RATE: 0.0004     # 0.0003 → 0.0004 (20% plus élevé)
EPOCH: 50                 # 40 → 50 (25% plus d'epochs)
PATIENCE: 15              # 12 → 15 (plus de patience)
```

---

## 🎯 Objectifs LargeExp_Balanced

### Métriques Cibles

| Métrique | DefaultExp | LargeExp | LargeExp_Reg | **Target** |
|----------|-----------|----------|--------------|------------|
| Accuracy | 0.7341 | 0.7142 | 0.7278 | **> 0.735** ✓ |
| IoU | 0.5919 | 0.5745 | 0.5743 | **> 0.590** ✓ |
| Micro AUC | 0.9384 | 0.9025 | 0.9199 | **> 0.930** ✓ |
| Train Loss | 0.3681 | 0.3334 | 0.6108 | **0.45-0.50** |
| Val Loss | 0.4976 | 0.4342 | 0.5301 | **0.48-0.52** |
| Gap | 0.13 | 0.10 | 0.08 | **< 0.05** ✓ |

### Per-Class Targets

| Class | Rappel Dataset | LargeExp_Reg | **Target** |
|-------|---------------|--------------|------------|
| Class 0 | 36% | 0.8338 | **0.80-0.82** |
| Class 1 | 0.84% | 0.3979 | **> 0.50** 🎯 |
| Class 2 | 6% | 0.6362 | **> 0.65** |
| Class 3 | 13% | 0.6915 | **> 0.70** |
| Class 4 | 44% | 0.4472 | **> 0.58** 🎯 |

---

## 📈 Prédictions

### Scénario Optimiste (✓ Success)
- Train loss converge vers 0.47
- Val loss stable à 0.49
- Gap: 0.02 (excellent)
- Test Accuracy: 0.74-0.75 (+1-2% vs DefaultExp)
- Class 1 Recall: 0.52-0.55
- Class 4 Recall: 0.58-0.60

### Scénario Réaliste (✓ Good)
- Train loss: 0.50
- Val loss: 0.52
- Gap: 0.02-0.04
- Test Accuracy: 0.73-0.74 (~= DefaultExp)
- Class 1 Recall: 0.48-0.52
- Class 4 Recall: 0.55-0.58

### Scénario Pessimiste (⚠️ Needs Adjustment)
- Train loss: 0.55
- Val loss: 0.57
- Gap: 0.02
- Test Accuracy: 0.72 (< DefaultExp)
- → Réduire encore plus la régularisation

---

## 🚀 Plan d'Action

### Étape 1: Lancer LargeExp_Balanced
```bash
python main.py --config Todo_List/LargeExp_Balanced.yaml
```

### Étape 2: Monitoring Pendant l'Entraînement
Observer à chaque epoch:
- ✓ Train loss descend progressivement (pas trop lent, pas trop rapide)
- ✓ Val loss suit train loss (gap < 0.05)
- ✓ Pas d'effondrement de val loss après epoch 20

**Signes de succès à epoch 20**:
- Train loss ~0.55-0.60
- Val loss ~0.57-0.62
- Gap < 0.05

### Étape 3: Ajustements si Nécessaire

**Si underfitting (train loss > 0.60 à epoch 30)**:
- Réduire DROPOUT à 0.15
- Réduire augmentation (probabilities /2)
- Augmenter LEARNING_RATE à 0.0005

**Si overfitting (gap > 0.08)**:
- Augmenter WEIGHT_DECAY à 0.0001
- Augmenter PATIENCE à 20

**Si Class 1 encore faible (< 0.45)**:
- Augmenter CLASS_WEIGHTS[1] à 5.0 ou 6.0

**Si Class 4 effondrement (< 0.50)**:
- Réduire CLASS_WEIGHTS[4] à 0.7 ou 0.6

---

## 📝 Conclusion

**Lesson Learned**: Trop de régularisation = Underfitting = Pire performance

**La clé**: Trouver l'équilibre entre:
- Capacité du modèle (256 channels ✓)
- Régularisation (modérée)
- Augmentation (douce)
- Class weights (équilibrés)

**Philosophie**: On veut un modèle qui **apprend bien** (train loss bas) TOUT EN **généralisant bien** (gap < 0.05).

LargeExp_Balanced devrait atteindre cet équilibre ! 🎯
