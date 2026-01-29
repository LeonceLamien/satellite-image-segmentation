#!/usr/bin/env python3
"""
Script to create a project .zip without the dataset (images and annotations)
"""

import os
import zipfile
from pathlib import Path

def create_submission_zip(output_name="submission.zip"):
    """
    Creates a .zip file with the project code, without images and annotations
    """
    
    # Files and folders to include
    items_to_include = [
        "visualize_augmentations.py",
        "visualize_segmentation.py",
        "main.py",
        "check_mask_colors.py",
        "Todo_List/",
        "Networks/",
        "Dataset/dataLoader.py",
        "Dataset/makeGraph.py",
        "Dataset/train.json",
        "Dataset/val.json",
        "Dataset/test.json",
        "Dataset/__pycache__/",
    ]
    
    # Folders to completely exclude
    exclude_dirs = [
        "Dataset/images",
        "Dataset/annotations",
        "Results",
        ".git",
        "__pycache__",
        "augmentation_examples",
    ]
    
    print("Creating submission file...")
    print(f"Output file: {output_name}\n")
    
    with zipfile.ZipFile(output_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        files_added = 0
        
        for item in items_to_include:
            item_path = Path(item)
            
            # If it's a file
            if item_path.is_file():
                zipf.write(item, item)
                print(f"✓ Added: {item}")
                files_added += 1
            
            # If it's a folder
            elif item_path.is_dir():
                for root, dirs, files in os.walk(item):
                    # Filter out excluded folders
                    dirs[:] = [d for d in dirs if not any(excl in os.path.join(root, d) for excl in exclude_dirs)]
                    
                    for file in files:
                        file_path = os.path.join(root, file)
                        
                        # Check that the file is not in an excluded folder
                        if not any(excl in file_path for excl in exclude_dirs):
                            zipf.write(file_path, file_path)
                            print(f"✓ Added: {file_path}")
                            files_added += 1
            
            else:
                print(f"Not found: {item}")
    
    # Get file size
    file_size = os.path.getsize(output_name)
    size_mb = file_size / (1024 * 1024)
    
    print(f"\nArchive created successfully!")
    print(f"{files_added} files added")
    print(f"Size: {size_mb:.2f} MB")
    print(f"File: {output_name}")
    
    # List zip contents
    print(f"\nContents of .zip file:")
    with zipfile.ZipFile(output_name, 'r') as zipf:
        for info in zipf.filelist[:20]:  # Display first 20
            print(f"   - {info.filename} ({info.file_size} bytes)")
        if len(zipf.filelist) > 20:
            print(f"   ... and {len(zipf.filelist) - 20} other files")
    
    print(f"\nReady for submission!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create submission .zip file')
    parser.add_argument('-o', '--output', type=str, default='submission.zip',
                       help='Output .zip filename (default: submission.zip)')
    
    args = parser.parse_args()
    
    create_submission_zip(args.output)
