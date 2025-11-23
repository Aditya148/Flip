"""
Script to prepare Flip SDK for PyPI publication.

This script:
1. Removes setup.py (conflicts with pyproject.toml)
2. Cleans build artifacts
3. Builds the package
"""

import os
import shutil
from pathlib import Path


def remove_file(filepath):
    """Remove a file if it exists."""
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            print(f"✅ Removed {filepath}")
            return True
        except Exception as e:
            print(f"❌ Failed to remove {filepath}: {e}")
            return False
    return True


def remove_directory(dirpath):
    """Remove a directory if it exists."""
    if os.path.exists(dirpath):
        try:
            shutil.rmtree(dirpath)
            print(f"✅ Removed {dirpath}/")
            return True
        except Exception as e:
            print(f"❌ Failed to remove {dirpath}: {e}")
            return False
    return True


def main():
    """Prepare package for building."""
    print("\n" + "=" * 60)
    print("Preparing Flip SDK for PyPI Publication")
    print("=" * 60 + "\n")
    
    # Step 1: Remove setup.py
    print("Step 1: Removing setup.py...")
    if not remove_file("setup.py"):
        print("\n⚠️  Please manually delete setup.py and run this script again")
        return 1
    
    # Step 2: Clean build artifacts
    print("\nStep 2: Cleaning build artifacts...")
    remove_directory("dist")
    remove_directory("build")
    
    # Remove egg-info directories
    for item in Path(".").glob("*.egg-info"):
        remove_directory(str(item))
    
    print("\n" + "=" * 60)
    print("✅ Preparation complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run: python -m build")
    print("2. Check: twine check dist/*")
    print("3. Upload: twine upload dist/*")
    print("\nSee PUBLISHING.md for detailed instructions.")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
