# Documentation des Paramètres

Ce document décrit tous les paramètres disponibles pour configurer vos expériences. **Incluez uniquement les paramètres spécifiques au modèle/configuration que vous utilisez** dans vos fichiers YAML.

---

## 📊 DATASET - Paramètres du Dataset

Tous ces paramètres sont utilisés quelle que soit l'architecture choisie.

| Paramètre | Type | Valeur par défaut | Description | Obligatoire |
|-----------|------|-------------------|-------------|-------------|
| `RESIZE_SHAPE` | string | - | Dimension de redimensionnement des images au format "HxW" (ex: "64x64", "256x256") | ✅ Oui |
| `NORMALIZE_MEAN` | list[float] | `[0.485, 0.456, 0.406]` | Moyenne pour la normalisation (ImageNet par défaut) | ❌ Non |
| `NORMALIZE_STD` | list[float] | `[0.229, 0.224, 0.225]` | Écart-type pour la normalisation (ImageNet par défaut) | ❌ Non |
| `NUM_WORKERS` | int | `2` | Nombre de workers pour le DataLoader | ❌ Non |

### Exemple DATASET
```yaml
DATASET:
  RESIZE_SHAPE: 256x256
  NUM_WORKERS: 4
```

---

## 🧠 MODEL - Paramètres du Modèle

### Paramètres Communs (tous les modèles)

| Paramètre | Type | Valeur par défaut | Description | Obligatoire |
|-----------|------|-------------------|-------------|-------------|
| `ARCHITECTURE` | string | `"DeepLabV3Plus"` | Type d'architecture : `"DeepLabV3Plus"`, `"ImprovedU_net"`, `"Net"` | ❌ Non |
| `NUM_CLASSES` | int | `5` | Nombre de classes pour la segmentation | ❌ Non |

---

### Architecture : **Net** (réseau simple baseline)

Architecture basique avec 3 couches convolutionnelles.

| Paramètre | Type | Valeur par défaut | Description | Obligatoire |
|-----------|------|-------------------|-------------|-------------|
| `NB_CHANNEL` | int | - | Nombre de canaux dans les couches cachées | ✅ Oui |

#### Exemple pour Net
```yaml
MODEL:
  ARCHITECTURE: Net
  NB_CHANNEL: 64
  NUM_CLASSES: 5
```

---

### Architecture : **ImprovedU_net** (U-Net avec attention optionnelle)

Architecture U-Net améliorée avec mécanismes d'attention optionnels.

| Paramètre | Type | Valeur par défaut | Description | Obligatoire |
|-----------|------|-------------------|-------------|-------------|
| `NB_CHANNEL` | int | - | Nombre de canaux (non utilisé dans l'implémentation actuelle, mais requis) | ✅ Oui |
| `DROPOUT` | float | - | Taux de dropout appliqué au bottleneck (ex: 0.5) | ✅ Oui |
| `USE_ATTENTION` | bool | - | Active/désactive les mécanismes d'attention gates | ✅ Oui |

**Note :** L'architecture utilise des canaux fixes (64, 128, 256, 512, 1024) indépendamment de `NB_CHANNEL`.

#### Exemple pour ImprovedU_net
```yaml
MODEL:
  ARCHITECTURE: ImprovedU_net
  NB_CHANNEL: 64
  DROPOUT: 0.5
  USE_ATTENTION: true
  NUM_CLASSES: 5
```

---

### Architecture : **DeepLabV3Plus** (DeepLabV3+ avec ASPP)

Architecture DeepLabV3+ avec module ASPP (Atrous Spatial Pyramid Pooling).

| Paramètre | Type | Valeur par défaut | Description | Obligatoire |
|-----------|------|-------------------|-------------|-------------|
| `BACKBONE_CHANNELS` | int | `64` | Nombre de canaux de base pour le backbone (les couches suivantes utilisent 2x et 4x) | ❌ Non |
| `ASPP_OUTPUT_CHANNELS` | int | `256` | Nombre de canaux en sortie du module ASPP | ❌ Non |
| `ASPP_DROPOUT` | float | `0.5` | Taux de dropout dans le module ASPP | ❌ Non |
| `ASPP_DILATIONS` | list[int] | `[6, 12, 18]` | Taux de dilatation pour les 3 branches atrous du module ASPP | ❌ Non |

#### Exemple pour DeepLabV3Plus
```yaml
MODEL:
  ARCHITECTURE: DeepLabV3Plus
  BACKBONE_CHANNELS: 256
  ASPP_OUTPUT_CHANNELS: 256
  ASPP_DROPOUT: 0.5
  ASPP_DILATIONS: [6, 12, 18]
  NUM_CLASSES: 5
```

---

## 🎯 TRAINING - Paramètres d'Entraînement

### Paramètres de Base

| Paramètre | Type | Valeur par défaut | Description | Obligatoire |
|-----------|------|-------------------|-------------|-------------|
| `LEARNING_RATE` | float | - | Taux d'apprentissage initial | ✅ Oui |
| `EPOCH` | int | - | Nombre d'époques d'entraînement | ✅ Oui |
| `BATCH_SIZE` | int | - | Taille des batchs | ✅ Oui |
| `DEVICE` | string | - | Device pour l'entraînement : `"cuda"` ou `"cpu"` | ✅ Oui |
| `RANDOM_SEED` | int | `2885` | Seed pour la reproductibilité | ❌ Non |

---

### Optimiseurs

| Paramètre | Type | Valeur par défaut | Description | Obligatoire |
|-----------|------|-------------------|-------------|-------------|
| `OPTIMIZER` | string | `"Adam"` | Type d'optimiseur : `"Adam"`, `"SGD"`, `"AdamW"` | ❌ Non |

#### Optimiseur spécifique : **SGD**

| Paramètre | Type | Valeur par défaut | Description | Obligatoire si SGD |
|-----------|------|-------------------|-------------|-------------------|
| `MOMENTUM` | float | `0.9` | Momentum pour SGD | ❌ Non |
| `WEIGHT_DECAY` | float | `0.0001` | Weight decay (L2 regularization) pour SGD | ❌ Non |

#### Optimiseur spécifique : **AdamW**

| Paramètre | Type | Valeur par défaut | Description | Obligatoire si AdamW |
|-----------|------|-------------------|-------------|---------------------|
| `WEIGHT_DECAY` | float | `0.01` | Weight decay pour AdamW | ❌ Non |

---

### Learning Rate Scheduler (optionnel)

| Paramètre | Type | Valeur par défaut | Description | Obligatoire |
|-----------|------|-------------------|-------------|-------------|
| `USE_SCHEDULER` | bool | `false` | Active/désactive le scheduler | ❌ Non |
| `SCHEDULER_TYPE` | string | `"StepLR"` | Type de scheduler : `"StepLR"`, `"ReduceLROnPlateau"`, `"CosineAnnealingLR"` | Si USE_SCHEDULER=true |

#### Scheduler spécifique : **StepLR**

| Paramètre | Type | Valeur par défaut | Description | Obligatoire si StepLR |
|-----------|------|-------------------|-------------|----------------------|
| `SCHEDULER_STEP_SIZE` | int | `10` | Nombre d'époques avant de réduire le LR | ❌ Non |
| `SCHEDULER_GAMMA` | float | `0.1` | Facteur multiplicatif de réduction du LR | ❌ Non |

**Note :** `ReduceLROnPlateau` et `CosineAnnealingLR` utilisent des paramètres codés en dur (patience=5, factor=0.5 pour Plateau ; T_max=EPOCH pour Cosine).

---

## 📋 Exemples de Configurations Complètes

### Configuration Minimaliste (Net)
```yaml
DATASET:
  RESIZE_SHAPE: 64x64

MODEL:
  ARCHITECTURE: Net
  NB_CHANNEL: 64

TRAINING:
  LEARNING_RATE: 0.001
  EPOCH: 30
  BATCH_SIZE: 16
  DEVICE: cuda
```

### Configuration ImprovedU_net avec Attention
```yaml
DATASET:
  RESIZE_SHAPE: 128x128
  NUM_WORKERS: 4

MODEL:
  ARCHITECTURE: ImprovedU_net
  NB_CHANNEL: 64
  DROPOUT: 0.5
  USE_ATTENTION: true
  NUM_CLASSES: 5

TRAINING:
  LEARNING_RATE: 0.0001
  EPOCH: 50
  BATCH_SIZE: 8
  DEVICE: cuda
  OPTIMIZER: AdamW
  WEIGHT_DECAY: 0.01
  RANDOM_SEED: 42
```

### Configuration DeepLabV3Plus avec Scheduler
```yaml
DATASET:
  RESIZE_SHAPE: 256x256
  NORMALIZE_MEAN: [0.485, 0.456, 0.406]
  NORMALIZE_STD: [0.229, 0.224, 0.225]
  NUM_WORKERS: 4

MODEL:
  ARCHITECTURE: DeepLabV3Plus
  BACKBONE_CHANNELS: 256
  ASPP_OUTPUT_CHANNELS: 256
  ASPP_DROPOUT: 0.5
  ASPP_DILATIONS: [6, 12, 18]
  NUM_CLASSES: 5

TRAINING:
  LEARNING_RATE: 0.001
  EPOCH: 100
  BATCH_SIZE: 16
  DEVICE: cuda
  OPTIMIZER: SGD
  MOMENTUM: 0.9
  WEIGHT_DECAY: 0.0001
  USE_SCHEDULER: true
  SCHEDULER_TYPE: StepLR
  SCHEDULER_STEP_SIZE: 20
  SCHEDULER_GAMMA: 0.1
  RANDOM_SEED: 2885
```

---

## 💡 Conseils d'Utilisation

### ✅ À FAIRE
- **Inclure uniquement les paramètres nécessaires** pour votre architecture/configuration
- Pour Net : inclure `NB_CHANNEL`
- Pour ImprovedU_net : inclure `NB_CHANNEL`, `DROPOUT`, `USE_ATTENTION`
- Pour DeepLabV3Plus : inclure `BACKBONE_CHANNELS`, `ASPP_OUTPUT_CHANNELS`, `ASPP_DROPOUT`, `ASPP_DILATIONS`
- Si vous utilisez SGD : inclure `MOMENTUM` et/ou `WEIGHT_DECAY` si vous voulez modifier les valeurs par défaut
- Si vous utilisez un scheduler : activer `USE_SCHEDULER` et spécifier le type

### ❌ À ÉVITER
- Ne pas inclure `BACKBONE_CHANNELS` si vous utilisez Net ou ImprovedU_net
- Ne pas inclure `DROPOUT` ou `USE_ATTENTION` si vous utilisez DeepLabV3Plus ou Net
- Ne pas inclure `NB_CHANNEL` si vous utilisez DeepLabV3Plus (non utilisé)
- Ne pas inclure les paramètres de scheduler si `USE_SCHEDULER: false`

---

## 🔍 Tableau Récapitulatif par Architecture

| Paramètre | Net | ImprovedU_net | DeepLabV3Plus |
|-----------|-----|---------------|---------------|
| `NB_CHANNEL` | ✅ Requis | ✅ Requis* | ❌ Non utilisé |
| `DROPOUT` | ❌ | ✅ Requis | ❌ |
| `USE_ATTENTION` | ❌ | ✅ Requis | ❌ |
| `BACKBONE_CHANNELS` | ❌ | ❌ | ✅ Utilisé |
| `ASPP_OUTPUT_CHANNELS` | ❌ | ❌ | ✅ Utilisé |
| `ASPP_DROPOUT` | ❌ | ❌ | ✅ Utilisé |
| `ASPP_DILATIONS` | ❌ | ❌ | ✅ Utilisé |
| `NUM_CLASSES` | ✅ Utilisé | ✅ Utilisé | ✅ Utilisé |

*Note : Pour ImprovedU_net, `NB_CHANNEL` est requis mais non utilisé dans l'implémentation actuelle (architecture avec canaux fixes).
