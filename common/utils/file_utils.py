"""
File Utilities
==============

Common file and directory operations.
"""

import os
from pathlib import Path
from typing import Union


def create_folder(path: Union[str, Path]) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    """
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists and return Path object.
    
    Args:
        path: Directory path
    
    Returns:
        Path object
    """
    path = Path(path)
    create_folder(path)
    return path
