import json
import re


def extract_number_from_filename(filename):
    """Extract numeric ID from filename like '4001.png' -> 4001"""
    match = re.search(r'(\d+)\.png', filename)
    return int(match.group(1)) if match else None


def fix_and_merge_annotations(json_files, output_file):
    """
    Fix image IDs to match filenames and merge multiple COCO annotation files.

    Args:
        json_files: List of paths to JSON files to merge
        output_file: Path to save the merged JSON file
    """
    merged_data = {
        "licenses": [],
        "info": {},
        "categories": [],
        "images": [],
        "annotations": []
    }

    annotation_id_offset = 0

    for file_idx, json_file in enumerate(json_files):
        print(f"\nProcessing {json_file}...")

        with open(json_file, 'r') as f:
            data = json.load(f)

        # Copy metadata from first file only
        if file_idx == 0:
            merged_data["licenses"] = data.get("licenses", [])
            merged_data["info"] = data.get("info", {})
            merged_data["categories"] = data.get("categories", [])

        # Create mapping from old image_id to new image_id
        id_mapping = {}

        # Process images
        for img in data["images"]:
            old_id = img["id"]
            filename = img["file_name"]

            # Extract the actual ID from filename
            new_id = extract_number_from_filename(filename)

            if new_id is None:
                print(f"Warning: Could not extract ID from filename {filename}")
                continue

            # Store mapping
            id_mapping[old_id] = new_id

            # Update image entry
            img["id"] = new_id
            merged_data["images"].append(img)

            print(f"  Mapped image ID {old_id} -> {new_id} ({filename})")

        # Process annotations with updated IDs
        for ann in data["annotations"]:
            old_image_id = ann["image_id"]

            # Update to new image_id
            if old_image_id in id_mapping:
                ann["image_id"] = id_mapping[old_image_id]

                # Update annotation ID to be globally unique
                ann["id"] = annotation_id_offset + ann["id"]

                merged_data["annotations"].append(ann)
            else:
                print(f"  Warning: Annotation {ann['id']} references missing image_id {old_image_id}")

        # Update offset for next file
        if data["annotations"]:
            max_ann_id = max(ann["id"] for ann in data["annotations"])
            annotation_id_offset += max_ann_id

        print(f"  Processed {len(data['images'])} images and {len(data['annotations'])} annotations")

    # Sort by ID for cleaner output
    merged_data["images"].sort(key=lambda x: x["id"])
    merged_data["annotations"].sort(key=lambda x: x["id"])

    # Save merged file
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)

    print(f"\n✓ Merged data saved to {output_file}")
    print(f"  Total images: {len(merged_data['images'])}")
    print(f"  Total annotations: {len(merged_data['annotations'])}")
    print(f"  Image ID range: {merged_data['images'][0]['id']} to {merged_data['images'][-1]['id']}")


# Usage
if __name__ == "__main__":
    json_files = [
        "instances_default1.json",  # Should contain images 1-2000
        "instances_default2.json",  # Should contain images 2001-4000
        "instances_default3.json"  # Should contain images 4001-5000
    ]

    output_file = "instances_merged.json"

    fix_and_merge_annotations(json_files, output_file)

    # Verify the result
    print("\n" + "=" * 50)
    print("VERIFICATION")
    print("=" * 50)

    with open(output_file, 'r') as f:
        result = json.load(f)

    # Check for any mismatches
    print("\nChecking for ID/filename mismatches:")
    mismatches = []
    for img in result["images"][:10]:  # Show first 10 as sample
        expected_filename = f"{img['id']:04d}.png"
        if img["file_name"] != expected_filename:
            mismatches.append(f"  ID {img['id']} -> {img['file_name']} (expected {expected_filename})")
            print(f"  ✗ ID {img['id']} -> {img['file_name']} (expected {expected_filename})")
        else:
            print(f"  ✓ ID {img['id']} -> {img['file_name']}")

    if not mismatches:
        print("\n✓ All IDs match their filenames correctly!")
    else:
        print(f"\n✗ Found {len(mismatches)} mismatches")