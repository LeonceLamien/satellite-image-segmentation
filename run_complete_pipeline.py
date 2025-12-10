"""
Complete automated pipeline for Part B self-supervised learning.

Runs:
1. Training on selected proxy task
2. Visualization of task results
3. Feature extraction and clustering
4. Clustering visualization

Usage:
    python run_complete_pipeline.py --task denoising --device cuda
    python run_complete_pipeline.py --task colorization --device cpu
    python run_complete_pipeline.py --task context_prediction --device cuda
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print("\n" + "=" * 80)
    print(f"▶️  {description}")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error in: {description}")
        print(f"Command failed with return code {result.returncode}")
        sys.exit(1)
    
    print(f"\n✅ Completed: {description}")
    return result


def main():
    parser = argparse.ArgumentParser(description='Run complete SSL pipeline')
    parser.add_argument('--task', type=str, required=True,
                        choices=['denoising', 'colorization', 'context_prediction'],
                        help='Proxy task to run')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--n_clusters', type=int, default=5,
                        help='Number of clusters for KMeans (default: 5)')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training if model already exists')
    parser.add_argument('--skip_task_viz', action='store_true',
                        help='Skip task visualization')
    
    args = parser.parse_args()
    
    # Get workspace root
    workspace_root = Path(__file__).parent
    os.chdir(workspace_root)
    
    # Set PYTHONPATH
    env = os.environ.copy()
    env['PYTHONPATH'] = str(workspace_root)
    
    print("\n" + "🚀" * 40)
    print(f"COMPLETE PIPELINE: {args.task.upper()}")
    print("🚀" * 40)
    print(f"\nTask: {args.task}")
    print(f"Device: {args.device}")
    print(f"Clusters: {args.n_clusters}")
    print(f"Workspace: {workspace_root}")
    
    # Task configuration mapping
    task_configs = {
        'denoising': 'configs/denoising.yaml',
        'colorization': 'configs/colorization.yaml',
        'context_prediction': 'configs/context_pred.yaml'
    }
    
    config_path = task_configs[args.task]
    exp_dir = f'Results/{args.task}_exp'
    model_path = f'{exp_dir}/_Weights/best_model.pth'
    
    # =========================================================================
    # STEP 1: TRAINING
    # =========================================================================
    if args.skip_training and os.path.exists(model_path):
        print(f"\n⏩ Skipping training (model exists: {model_path})")
    else:
        cmd = [
            sys.executable, 'main_partB.py',
            '--task', args.task,
            '--config', config_path,
            '--device', args.device
        ]
        run_command(cmd, f"STEP 1/4: Training {args.task} model")
    
    # =========================================================================
    # STEP 2: TASK VISUALIZATION
    # =========================================================================
    if not args.skip_task_viz:
        cmd = [
            sys.executable, 'visualize_single_task.py',
            '--task', args.task,
            '--exp', exp_dir,
            '--config', config_path
        ]
        run_command(cmd, f"STEP 2/4: Visualizing {args.task} results")
    else:
        print(f"\n⏩ Skipping task visualization")
    
    # =========================================================================
    # STEP 3: CLUSTERING
    # =========================================================================
    cmd = [
        sys.executable, '-m', 'clustering.cluster_features',
        '--exp', exp_dir,
        '--task', args.task,
        '--n_clusters', str(args.n_clusters),
        '--device', args.device
    ]
    run_command(cmd, f"STEP 3/4: Extracting features and clustering")
    
    # =========================================================================
    # STEP 4: CLUSTERING VISUALIZATION
    # =========================================================================
    cmd = [
        sys.executable, '-m', 'clustering.visualize_clusters',
        '--exp', exp_dir,
        '--task', args.task,
        '--n_clusters', str(args.n_clusters)
    ]
    run_command(cmd, f"STEP 4/4: Visualizing clustering results")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "🎉" * 40)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("🎉" * 40)
    
    print(f"\n📁 Results saved in: {exp_dir}/")
    print("\n📊 Generated files:")
    print(f"   • {exp_dir}/_Weights/best_model.pth")
    print(f"   • {exp_dir}/training_curves.png")
    print(f"   • visualization_{args.task}.png")
    print(f"   • {exp_dir}/cluster_labels_n{args.n_clusters}.npy")
    print(f"   • {exp_dir}/clustering_model_n{args.n_clusters}.pkl")
    print(f"   • {exp_dir}/clustering_vs_gt_n{args.n_clusters}.png")
    
    print("\n💡 Next steps:")
    print(f"   1. Check visualizations to verify results")
    print(f"   2. Try different number of clusters:")
    print(f"      python run_complete_pipeline.py --task {args.task} --n_clusters 8 --skip_training")
    print(f"   3. Run other tasks:")
    other_tasks = [t for t in ['denoising', 'colorization', 'context_prediction'] if t != args.task]
    for task in other_tasks:
        print(f"      python run_complete_pipeline.py --task {task} --device {args.device}")
    print()


if __name__ == "__main__":
    main()
