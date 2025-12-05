#!/usr/bin/env python3
"""
Script pour tester le calcul des class weights
Usage: python test_class_weights.py
"""

import sys
import os
from os.path import dirname, abspath

# Setup paths
rootDirectory = dirname(abspath(__file__))
sys.path.insert(0, rootDirectory)

from Dataset.classWeights import compare_weight_methods

# Paths
datasetDirectory = os.path.join(rootDirectory, "Dataset")
imgDirectory = os.path.join(datasetDirectory, "images")
maskDirectory = os.path.join(datasetDirectory, "annotations")

print("\n" + "="*80)
print("🧪 TEST: CLASS WEIGHTS COMPUTATION")
print("="*80)
print(f"\n📁 Images directory: {imgDirectory}")
print(f"📁 Masks directory: {maskDirectory}\n")

# Compare all methods
compare_weight_methods(
    imgDirectory=imgDirectory,
    maskDirectory=maskDirectory,
    setName="train",
    num_classes=5,
    beta=0.999
)

print("\n" + "="*80)
print("✅ TEST COMPLETED")
print("="*80)
print("\n💡 Next steps:")
print("   1. Regardez les poids calculés ci-dessus")
print("   2. Choisissez la méthode qui vous convient")
print("   3. Ajoutez dans votre YAML:")
print("      TRAINING:")
print("        USE_CLASS_WEIGHTS: true")
print("        CLASS_WEIGHT_METHOD: effective_samples  # ou autre")
print("   4. Relancez l'entraînement: python main.py -exp DefaultExp_WithWeights")
print("\n" + "="*80 + "\n")
