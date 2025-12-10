#!/usr/bin/env python3
"""
Script to create submission.zip for Part B
Date: 10 décembre 2025
"""

import os
import zipfile
import shutil
from pathlib import Path

def create_submission():
    print("=" * 50)
    print("Creating Part B Submission Archive")
    print("=" * 50)
    print()
    
    # Paths
    base_dir = Path(__file__).parent
    zip_path = base_dir / "submission.zip"
    
    # Remove old submission
    if zip_path.exists():
        zip_path.unlink()
        print("Removed old submission.zip")
    
    print("Creating archive...")
    print()
    
    # Files and directories to include
    files_to_include = [
        "main.py",
        "run_complete_pipeline.py", 
        "visualize_single_task.py"
    ]
    
    dirs_to_include = {
        "Todo_List": ["*.yaml"],
        "proxy_tasks": ["*.py"],
        "clustering": ["*.py"],
        "Dataset": ["*.py", "*.json"]  # Exclude images/ and annotations/
    }
    
    # Create zip archive
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        
        # Add individual files
        for file in files_to_include:
            file_path = base_dir / file
            if file_path.exists():
                zipf.write(file_path, file)
                print(f"  ✓ {file}")
            else:
                print(f"  {file} not found, skipping")
        
        print()
        
        # Add directories
        for dir_name, patterns in dirs_to_include.items():
            dir_path = base_dir / dir_name
            if dir_path.exists():
                print(f"  ✓ {dir_name}/")
                
                for root, dirs, files in os.walk(dir_path):
                    root_path = Path(root)
                    
                    # Skip __pycache__ directories
                    if '__pycache__' in root_path.parts:
                        continue
                    
                    # Skip images/ and annotations/ in Dataset
                    if dir_name == "Dataset" and root_path.name in ["images", "annotations"]:
                        continue
                    
                    for file in files:
                        file_path = root_path / file
                        
                        # Check if file matches any pattern
                        if any(file_path.match(pattern) for pattern in patterns):
                            arcname = file_path.relative_to(base_dir)
                            zipf.write(file_path, arcname)
                            print(f"    • {arcname}")
            else:
                print(f"  {dir_name}/ not found, skipping")
        
        print()
      
        print("  ✓ SUBMISSION_README.txt")
    
    print()
    print("=" * 50)
    print("Submission created successfully!")
    print("=" * 50)
    print()
    print(f"File: {zip_path.name}")
    print(f"Size: {zip_path.stat().st_size / 1024:.1f} KB")
    print()
    
    # List contents
    print("Contents:")
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        file_list = zipf.namelist()
        for _, name in enumerate(sorted(file_list)[:30]):
            print(f"  {name}")
        if len(file_list) > 30:
            print(f"  ... and {len(file_list) - 30} more files")
        print()
        print(f"Total files: {len(file_list)}")
    
    print()
    print("Ready to submit!")
    print()

if __name__ == "__main__":
    create_submission()
