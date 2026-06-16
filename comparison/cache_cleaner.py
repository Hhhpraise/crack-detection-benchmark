"""
YOLO CACHE CLEANER UTILITY
Finds and deletes all YOLO cache files from training datasets
Clears space on C: drive even when datasets are on D: drive
"""

import os
import shutil
import glob
from pathlib import Path


class YOLOCacheCleaner:
    def __init__(self):
        self.cache_locations = []
        self.total_size_freed = 0

    def get_folder_size(self, folder_path):
        """Calculate total size of a folder"""
        total = 0
        try:
            for entry in os.scandir(folder_path):
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += self.get_folder_size(entry.path)
        except Exception as e:
            print(f"   Error calculating size: {e}")
        return total

    def format_size(self, size_bytes):
        """Convert bytes to human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"

    def find_yolo_caches(self, search_paths):
        """Find all YOLO cache files in specified paths"""
        print("\n🔍 Scanning for YOLO cache files...")

        cache_files = []

        for search_path in search_paths:
            search_path = Path(search_path)

            if not search_path.exists():
                print(f"   ⚠️ Path not found: {search_path}")
                continue

            print(f"\n   Searching in: {search_path}")

            # Find .cache files in labels folders
            for cache_file in search_path.rglob("*.cache"):
                size = os.path.getsize(cache_file)
                cache_files.append({
                    'path': cache_file,
                    'size': size,
                    'type': 'cache'
                })
                print(f"      Found: {cache_file.name} ({self.format_size(size)})")

        return cache_files

    def find_ultralytics_temp_files(self):
        """Find Ultralytics temporary files on C: drive"""
        print("\n🔍 Scanning for Ultralytics temporary files on C: drive...")

        temp_locations = [
            Path.home() / "AppData" / "Local" / "Temp",  # User temp
            Path("C:/Windows/Temp"),  # System temp
            Path.home() / ".cache" / "torch",  # PyTorch cache
            Path.home() / ".cache" / "huggingface",  # HuggingFace cache
            Path.home() / "AppData" / "Roaming" / "Ultralytics",  # Ultralytics settings
        ]

        temp_files = []

        for temp_path in temp_locations:
            if not temp_path.exists():
                continue

            print(f"\n   Checking: {temp_path}")

            try:
                # Find YOLO-related temp files
                for pattern in ["*yolo*", "*ultralytics*", "*.pt.lock", "*train_batch*"]:
                    for temp_file in temp_path.glob(pattern):
                        if temp_file.is_file():
                            size = os.path.getsize(temp_file)
                            temp_files.append({
                                'path': temp_file,
                                'size': size,
                                'type': 'temp'
                            })
                            print(f"      Found temp: {temp_file.name} ({self.format_size(size)})")
                        elif temp_file.is_dir():
                            size = self.get_folder_size(temp_file)
                            temp_files.append({
                                'path': temp_file,
                                'size': size,
                                'type': 'temp_dir'
                            })
                            print(f"      Found temp dir: {temp_file.name} ({self.format_size(size)})")
            except Exception as e:
                print(f"      Error scanning {temp_path}: {e}")

        return temp_files

    def find_runs_folders(self, search_paths):
        """Find YOLO runs/experiments folders"""
        print("\n🔍 Scanning for YOLO runs folders...")

        runs_folders = []

        for search_path in search_paths:
            search_path = Path(search_path)

            if not search_path.exists():
                continue

            # Find runs folders (but not the main experiments folder)
            for runs_dir in search_path.rglob("runs"):
                if runs_dir.is_dir():
                    size = self.get_folder_size(runs_dir)
                    runs_folders.append({
                        'path': runs_dir,
                        'size': size,
                        'type': 'runs'
                    })
                    print(f"      Found runs folder: {runs_dir} ({self.format_size(size)})")

        return runs_folders

    def delete_items(self, items, item_type="cache"):
        """Delete specified items"""
        if not items:
            print(f"\n   No {item_type} files found to delete")
            return

        total_size = sum(item['size'] for item in items)

        print(f"\n🗑️ Deleting {len(items)} {item_type} items ({self.format_size(total_size)})...")

        deleted_count = 0
        deleted_size = 0

        for item in items:
            try:
                item_path = item['path']

                if item_path.is_file():
                    item_path.unlink()
                    print(f"   ✓ Deleted file: {item_path.name}")
                elif item_path.is_dir():
                    shutil.rmtree(item_path)
                    print(f"   ✓ Deleted folder: {item_path.name}")

                deleted_count += 1
                deleted_size += item['size']

            except Exception as e:
                print(f"   ✗ Failed to delete {item_path}: {e}")

        print(f"\n   Successfully deleted {deleted_count}/{len(items)} items")
        print(f"   Space freed: {self.format_size(deleted_size)}")

        self.total_size_freed += deleted_size

    def clean_all(self, dataset_paths, delete_runs=False):
        """Clean all YOLO cache and temp files"""
        print("\n" + "=" * 100)
        print("YOLO CACHE CLEANER")
        print("=" * 100)

        # Find dataset caches
        cache_files = self.find_yolo_caches(dataset_paths)

        # Find temp files on C: drive
        temp_files = self.find_ultralytics_temp_files()

        # Find runs folders (optional)
        runs_folders = []
        if delete_runs:
            runs_folders = self.find_runs_folders(dataset_paths)

        # Summary
        total_items = len(cache_files) + len(temp_files) + len(runs_folders)
        total_size = sum(item['size'] for item in cache_files + temp_files + runs_folders)

        print("\n" + "=" * 100)
        print("SUMMARY")
        print("=" * 100)
        print(f"\nFound:")
        print(f"   Cache files: {len(cache_files)}")
        print(f"   Temp files: {len(temp_files)}")
        if delete_runs:
            print(f"   Runs folders: {len(runs_folders)}")
        print(f"\nTotal items: {total_items}")
        print(f"Total size: {self.format_size(total_size)}")

        if total_items == 0:
            print("\n✓ No cache files found - system is clean!")
            return

        # Confirm deletion
        print("\n" + "=" * 100)
        confirm = input("\nDelete all these files? (y/n): ").strip().lower()

        if confirm != 'y':
            print("\n❌ Cleanup cancelled")
            return

        # Delete files
        print("\n" + "=" * 100)
        print("CLEANING UP")
        print("=" * 100)

        self.delete_items(cache_files, "cache")
        self.delete_items(temp_files, "temp")

        if delete_runs:
            self.delete_items(runs_folders, "runs")

        # Final summary
        print("\n" + "=" * 100)
        print("CLEANUP COMPLETE")
        print("=" * 100)
        print(f"\n✅ Total space freed: {self.format_size(self.total_size_freed)}")
        print("\n💡 Tip: Run this script after each training session to keep your drive clean!")


def quick_clean(dataset_paths):
    """Quick clean without prompts - for use in training scripts"""
    cleaner = YOLOCacheCleaner()

    print("\n🧹 Quick cleanup: Deleting YOLO cache files...")

    cache_files = cleaner.find_yolo_caches(dataset_paths)

    if cache_files:
        cleaner.delete_items(cache_files, "cache")
        print(f"✅ Freed: {cleaner.format_size(cleaner.total_size_freed)}")
    else:
        print("✓ No cache files found")


def main():
    """Interactive cache cleaner"""
    print("=" * 100)
    print("YOLO CACHE CLEANER - Interactive Mode")
    print("=" * 100)

    # Default paths - UPDATE THESE!
    default_dataset_paths = [
        "unified_crack_dataset",
        "unified_crack_dataset_seg",
        os.environ.get("CRACK_WORK_IMAGES", "D:/work images"),
    ]

    print("\n📁 Default search paths:")
    for path in default_dataset_paths:
        print(f"   - {path}")

    use_defaults = input("\nUse these paths? (y/n): ").strip().lower()

    if use_defaults == 'y':
        dataset_paths = default_dataset_paths
    else:
        print("\nEnter paths to search (one per line, empty line to finish):")
        dataset_paths = []
        while True:
            path = input("Path: ").strip()
            if not path:
                break
            dataset_paths.append(path)

    # Ask about runs folders
    delete_runs = input("\nAlso delete 'runs' folders? (y/n): ").strip().lower() == 'y'

    # Clean
    cleaner = YOLOCacheCleaner()
    cleaner.clean_all(dataset_paths, delete_runs=delete_runs)


if __name__ == "__main__":
    main()