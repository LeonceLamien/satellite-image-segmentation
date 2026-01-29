"""
Part A - Semantic Segmentation
===============================

Train and evaluate segmentation models on satellite imagery.

Usage:
    python main.py -exp DeepLab_Adam
    python main.py -exp ImprovedUNet_SGD
"""

import argparse
import os
import sys
from pathlib import Path
from termcolor import colored

# Add parent directory to path for common modules
sys.path.append(str(Path(__file__).parent.parent))

from common.config import load_config
from common.data import LLNDataset
from torch.utils.data import DataLoader
from trainer import SegmentationTrainer

# Paths
root_directory = Path(__file__).parent
dataset_directory = root_directory / "Dataset"
img_directory = str(dataset_directory / "images")
mask_directory = str(dataset_directory / "annotations")

parser = argparse.ArgumentParser(description='Part A - Semantic Segmentation')
parser.add_argument('-exp', type=str, default='DeepLab_Adam',
                   help='Experiment name (YAML file in Todo_List folder)')


def main(args):
    """Main training and evaluation procedure."""
    print("\n" + "="*70)
    print(f"PART A - SEMANTIC SEGMENTATION: {args.exp}")
    print("="*70)
    
    # Load configuration
    config_path = root_directory / "Todo_List" / f"{args.exp}.yaml"
    config = load_config(str(config_path))
    print(f"\n✓ Configuration loaded from {config_path}")
    
    # Results path
    results_path = root_directory / "Results" / args.exp
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = config.training.get('DEVICE', 'cpu')
    print(f"✓ Using device: {device}")
    
    # Create datasets
    print(f"\n{'='*70}")
    print("Loading Datasets")
    print("="*70)
    
    train_dataset = LLNDataset(img_directory, mask_directory, "train", config.data)
    val_dataset = LLNDataset(img_directory, mask_directory, "val", config.data)
    test_dataset = LLNDataset(img_directory, mask_directory, "test", config.data)
    
    print(f"✓ Train set: {len(train_dataset)} images")
    print(f"✓ Val set:   {len(val_dataset)} images")
    print(f"✓ Test set:  {len(test_dataset)} images")
    
    # Create dataloaders
    batch_size = config.training.get('BATCH_SIZE', 16)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create trainer
    trainer = SegmentationTrainer(
        config=config.data,
        img_directory=img_directory,
        mask_directory=mask_directory,
        results_path=str(results_path),
        device=device
    )
    
    # Train
    print(f"\n{'='*70}")
    print(colored('TRAINING', 'red', attrs=['bold']))
    print("="*70)
    trainer.train(train_loader, val_loader)
    print(colored('\n✓ Training completed!', 'green', attrs=['bold']))
    
    # Evaluate
    print(f"\n{'='*70}")
    print(colored('EVALUATION', 'red', attrs=['bold']))
    print("="*70)
    trainer.load_weights()
    trainer.evaluate(test_loader)
    print(colored('\n✓ Evaluation completed!', 'green', attrs=['bold']))
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {results_path}")
    print("="*70)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
