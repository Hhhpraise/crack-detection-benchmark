import cv2
import numpy as np
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from scipy import ndimage
from skimage.morphology import skeletonize, medial_axis
import pandas as pd

# ===== USER CONFIGURATION =====
DETECTION_MODEL = 'yolo12s_cbam_ca_crack.pt'  # Your detection model
SEGMENTATION_MODEL = 'yolo12s_seg_cbam_ca_crack.pt'  # Your segmentation model
IMAGE_PATH = 'dataset/images/train/0138.png'
CONFIDENCE = 0.5
IMG_SIZE = 192
SAVE_RESULTS = True
RESULTS_DIR = 'combined_results'
USE_BOTH_MODELS = True  # Set to False to use only segmentation
OVERLAY_ALPHA = 0.6  # Transparency for segmentation overlay
SAVE_SEMANTIC_MASK = True  # New option to save semantic segmentation mask

# Crack measurement parameters
PIXELS_PER_MM = 10  # Adjust this based on your camera calibration and image resolution


# ==============================

def calculate_crack_dimensions(mask, pixels_per_mm=PIXELS_PER_MM):
    """
    Calculate crack dimensions including length, width, and area
    """
    # Ensure mask is binary
    if mask.dtype != np.bool_:
        binary_mask = mask > 0
    else:
        binary_mask = mask

    # Calculate area
    area_pixels = np.sum(binary_mask)
    area_mm2 = area_pixels / (pixels_per_mm ** 2)

    # Skeletonize to calculate length
    try:
        skeleton = skeletonize(binary_mask)
        length_pixels = np.sum(skeleton)
        length_mm = length_pixels / pixels_per_mm
    except:
        # Fallback method if skeletonization fails
        length_pixels = area_pixels  # Rough approximation
        length_mm = length_pixels / pixels_per_mm

    # Calculate average width
    if length_pixels > 0:
        avg_width_pixels = area_pixels / length_pixels
        avg_width_mm = avg_width_pixels / pixels_per_mm
    else:
        avg_width_pixels = 0
        avg_width_mm = 0

    # Calculate maximum width using distance transform
    if area_pixels > 0:
        # Distance transform to find width
        distance_transform = ndimage.distance_transform_edt(binary_mask)
        max_width_pixels = np.max(distance_transform) * 2  # Multiply by 2 for full width
        max_width_mm = max_width_pixels / pixels_per_mm

        # Calculate width distribution
        width_values = distance_transform[binary_mask] * 2
        width_std_mm = np.std(width_values) / pixels_per_mm
    else:
        max_width_pixels = 0
        max_width_mm = 0
        width_std_mm = 0

    return {
        'length_pixels': length_pixels,
        'length_mm': length_mm,
        'avg_width_pixels': avg_width_pixels,
        'avg_width_mm': avg_width_mm,
        'max_width_pixels': max_width_pixels,
        'max_width_mm': max_width_mm,
        'width_std_mm': width_std_mm,
        'area_pixels': area_pixels,
        'area_mm2': area_mm2
    }


def create_semantic_mask(seg_results, image_shape):
    """
    Create a semantic segmentation mask with white cracks on black background
    using polygon coordinates for precise alignment
    """
    # Create black background
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    if seg_results and hasattr(seg_results[0], 'masks') and seg_results[0].masks is not None:
        # Get the original image dimensions
        orig_h, orig_w = image_shape[:2]

        for i, mask_data in enumerate(seg_results[0].masks.xy):
            # Convert polygon points to integer coordinates
            polygon = np.array(mask_data, np.int32).reshape((-1, 1, 2))

            # Fill the polygon with white (255)
            cv2.fillPoly(mask, [polygon], 255)

    return mask


def create_overlay_mask(mask, color=(0, 255, 0)):
    """Create colored overlay from binary mask"""
    overlay = np.zeros((*mask.shape, 3), dtype=np.uint8)
    overlay[mask > 0] = color
    return overlay


def visualize_combined_results(image, det_results, seg_results, crack_dimensions=None, save_path=None):
    """Visualize both detection and segmentation results with precise mask alignment"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Original image
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Detection results
    if det_results and len(det_results[0].boxes) > 0:
        det_img = det_results[0].plot()
        axes[0, 1].imshow(cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title(f'Detection Results ({len(det_results[0].boxes)} cracks)',
                             fontsize=14, fontweight='bold')
    else:
        axes[0, 1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Detection Results (No cracks found)',
                             fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # Segmentation results
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if seg_results and hasattr(seg_results[0], 'masks') and seg_results[0].masks is not None:
        # Create mask using polygon coordinates for precise alignment
        combined_mask = np.zeros(rgb_image.shape[:2], dtype=np.uint8)
        for mask_poly in seg_results[0].masks.xy:
            polygon = np.array(mask_poly, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(combined_mask, [polygon], 255)

        # Create overlay
        overlay = create_overlay_mask(combined_mask, color=(255, 0, 0))  # Red for cracks
        seg_img = cv2.addWeighted(rgb_image, 1 - OVERLAY_ALPHA, overlay, OVERLAY_ALPHA, 0)

        axes[1, 0].imshow(seg_img)
        axes[1, 0].set_title(f'Segmentation Results ({len(seg_results[0].masks)} masks)',
                             fontsize=14, fontweight='bold')
    else:
        axes[1, 0].imshow(rgb_image)
        axes[1, 0].set_title('Segmentation Results (No masks found)',
                             fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    # Combined visualization
    combined_img = rgb_image.copy()

    # Add segmentation masks using polygon coordinates
    if seg_results and hasattr(seg_results[0], 'masks') and seg_results[0].masks is not None:
        combined_mask = np.zeros(rgb_image.shape[:2], dtype=np.uint8)
        for mask_poly in seg_results[0].masks.xy:
            polygon = np.array(mask_poly, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(combined_mask, [polygon], 255)

        overlay = create_overlay_mask(combined_mask, color=(255, 0, 0))
        combined_img = cv2.addWeighted(combined_img, 1 - OVERLAY_ALPHA, overlay, OVERLAY_ALPHA, 0)

    # Add detection boxes
    if det_results and len(det_results[0].boxes) > 0:
        ax = axes[1, 1]
        ax.imshow(combined_img)
        for box in det_results[0].boxes:
            xyxy = box.xyxy[0].tolist()
            conf = box.conf.item()
            rect = Rectangle((xyxy[0], xyxy[1]), xyxy[2] - xyxy[0], xyxy[3] - xyxy[1],
                             linewidth=2, edgecolor='yellow', facecolor='none')
            ax.add_patch(rect)
            ax.text(xyxy[0], xyxy[1] - 10, f'Crack {conf:.2%}',
                    color='yellow', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
    else:
        axes[1, 1].imshow(combined_img)

    # Add crack dimensions to the title if available
    title = 'Combined Results (Segmentation + Detection)'
    if crack_dimensions:
        title += f"\nLength: {crack_dimensions['length_mm']:.2f}mm, " \
                 f"Avg Width: {crack_dimensions['avg_width_mm']:.2f}mm, " \
                 f"Area: {crack_dimensions['area_mm2']:.2f}mm²"

    axes[1, 1].set_title(title, fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to: {save_path}")

    plt.close()


def calculate_crack_metrics(seg_results, image_shape):
    """Calculate crack coverage metrics from segmentation results using polygon coordinates"""
    if not seg_results or not hasattr(seg_results[0], 'masks') or seg_results[0].masks is None:
        return {
            'total_crack_pixels': 0,
            'crack_percentage': 0.0,
            'total_image_pixels': image_shape[0] * image_shape[1],
            'num_crack_regions': 0,
            'crack_dimensions': None
        }

    total_pixels = image_shape[0] * image_shape[1]
    combined_mask = np.zeros(image_shape[:2], dtype=np.uint8)

    # Use polygon coordinates for precise mask creation
    for mask_poly in seg_results[0].masks.xy:
        polygon = np.array(mask_poly, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(combined_mask, [polygon], 255)

    crack_pixels = np.sum(combined_mask > 0)
    crack_percentage = (crack_pixels / total_pixels) * 100

    # Calculate crack dimensions
    crack_dimensions = calculate_crack_dimensions(combined_mask)

    return {
        'total_crack_pixels': int(crack_pixels),
        'crack_percentage': crack_percentage,
        'total_image_pixels': total_pixels,
        'num_crack_regions': len(seg_results[0].masks),
        'crack_dimensions': crack_dimensions
    }


def calculate_individual_crack_dimensions(seg_results, image_shape):
    """Calculate dimensions for each individual crack segment"""
    if not seg_results or not hasattr(seg_results[0], 'masks') or seg_results[0].masks is None:
        return []

    individual_dimensions = []

    for i, mask_poly in enumerate(seg_results[0].masks.xy):
        # Create mask for this individual crack
        individual_mask = np.zeros(image_shape[:2], dtype=np.uint8)
        polygon = np.array(mask_poly, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(individual_mask, [polygon], 255)

        # Calculate dimensions
        dimensions = calculate_crack_dimensions(individual_mask)
        dimensions['id'] = i + 1
        individual_dimensions.append(dimensions)

    return individual_dimensions


def main():
    print(f"\n{'=' * 60}")
    print("🔬 COMBINED CRACK ANALYSIS")
    print(f"{'=' * 60}")
    print(f"📁 Testing image: {IMAGE_PATH}")
    print(f"🎯 Detection model: {DETECTION_MODEL}")
    print(f"🎨 Segmentation model: {SEGMENTATION_MODEL}")
    print(f"⚙️ Confidence: {CONFIDENCE}, Image Size: {IMG_SIZE}")
    print(f"📏 Pixels per mm: {PIXELS_PER_MM}")
    print(f"{'=' * 60}")

    try:
        # Create results directory
        os.makedirs(RESULTS_DIR, exist_ok=True)

        # Get base image name for saving
        img_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]

        # Load image
        print("📖 Loading image...")
        img = cv2.imread(IMAGE_PATH)
        if img is None:
            print(f"❌ Error: Failed to load image at {IMAGE_PATH}")
            return

        print(f"✅ Image loaded: {img.shape[1]}x{img.shape[0]} pixels")

        det_results = None
        seg_results = None

        # Load and run detection model
        if USE_BOTH_MODELS and os.path.exists(DETECTION_MODEL):
            print("\n🎯 Running detection model...")
            det_model = YOLO(DETECTION_MODEL)
            det_results = det_model.predict(
                source=img,
                conf=CONFIDENCE,
                save=False,
                imgsz=IMG_SIZE,
                verbose=False
            )
            print(f"✅ Detection complete: {len(det_results[0].boxes)} boxes found")

        # Load and run segmentation model
        if os.path.exists(SEGMENTATION_MODEL):
            print("\n🎨 Running segmentation model...")
            seg_model = YOLO(SEGMENTATION_MODEL)
            seg_results = seg_model.predict(
                source=img,
                conf=CONFIDENCE,
                save=False,
                imgsz=IMG_SIZE,
                verbose=False
            )

            mask_count = len(seg_results[0].masks) if (
                    hasattr(seg_results[0], 'masks') and seg_results[0].masks is not None) else 0
            print(f"✅ Segmentation complete: {mask_count} masks found")
        else:
            print(f"⚠️ Segmentation model not found: {SEGMENTATION_MODEL}")
            return

        # Analyze results
        print(f"\n{'=' * 60}")
        print("📊 ANALYSIS RESULTS")
        print(f"{'=' * 60}")

        # Detection results
        if det_results and len(det_results[0].boxes) > 0:
            print(f"🎯 Detection Results:")
            print(f"   └─ Found {len(det_results[0].boxes)} crack detection(s)")

            for i, box in enumerate(det_results[0].boxes):
                xyxy = box.xyxy[0].tolist()
                conf = box.conf.item()
                width = xyxy[2] - xyxy[0]
                height = xyxy[3] - xyxy[1]
                area = width * height

                print(f"     Detection {i + 1}:")
                print(f"       ├─ Confidence: {conf:.2%}")
                print(f"       ├─ Box: ({xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f})")
                print(f"       └─ Area: {area:.0f} pixels ({width:.0f}×{height:.0f})")
        else:
            print("🎯 Detection Results: No cracks detected")

        # Segmentation results
        metrics = calculate_crack_metrics(seg_results, img.shape)
        print(f"\n🎨 Segmentation Results:")
        print(f"   ├─ Crack regions: {metrics['num_crack_regions']}")
        print(f"   ├─ Crack pixels: {metrics['total_crack_pixels']:,}")
        print(f"   ├─ Total pixels: {metrics['total_image_pixels']:,}")
        print(f"   └─ Crack coverage: {metrics['crack_percentage']:.3f}%")

        # Crack dimensions
        if metrics['crack_dimensions']:
            dims = metrics['crack_dimensions']
            print(f"\n📏 Crack Dimensions:")
            print(f"   ├─ Length: {dims['length_mm']:.2f} mm")
            print(f"   ├─ Average width: {dims['avg_width_mm']:.2f} mm")
            print(f"   ├─ Maximum width: {dims['max_width_mm']:.2f} mm")
            print(f"   ├─ Width std: {dims['width_std_mm']:.2f} mm")
            print(f"   └─ Area: {dims['area_mm2']:.2f} mm²")

        # Individual crack dimensions
        individual_dims = calculate_individual_crack_dimensions(seg_results, img.shape)
        if individual_dims:
            print(f"\n📐 Individual Crack Dimensions:")
            for dim in individual_dims:
                print(f"   Crack {dim['id']}:")
                print(f"     ├─ Length: {dim['length_mm']:.2f} mm")
                print(f"     ├─ Average width: {dim['avg_width_mm']:.2f} mm")
                print(f"     ├─ Maximum width: {dim['max_width_mm']:.2f} mm")
                print(f"     └─ Area: {dim['area_mm2']:.2f} mm²")

        # Damage assessment
        print(f"\n🔍 Damage Assessment:")
        if metrics['crack_percentage'] > 5.0:
            severity = "SEVERE"
            severity_emoji = "🔴"
        elif metrics['crack_percentage'] > 2.0:
            severity = "MODERATE"
            severity_emoji = "🟡"
        elif metrics['crack_percentage'] > 0.5:
            severity = "MINOR"
            severity_emoji = "🟢"
        elif metrics['crack_percentage'] > 0.0:
            severity = "MINIMAL"
            severity_emoji = "🔵"
        else:
            severity = "NONE"
            severity_emoji = "⚪"

        print(f"   └─ Severity: {severity_emoji} {severity} ({metrics['crack_percentage']:.3f}% coverage)")

        # Save results
        if SAVE_RESULTS:
            print(f"\n💾 Saving results...")

            # Create detailed report
            report_path = os.path.join(RESULTS_DIR, f"{img_name}_analysis_report.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("CRACK ANALYSIS REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Image: {IMAGE_PATH}\n")
                f.write(f"Image size: {img.shape[1]}x{img.shape[0]} pixels\n")
                f.write(f"Analysis date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Confidence threshold: {CONFIDENCE}\n")
                f.write(f"Pixels per mm: {PIXELS_PER_MM}\n\n")

                f.write("DETECTION RESULTS:\n")
                f.write("-" * 20 + "\n")
                if det_results and len(det_results[0].boxes) > 0:
                    f.write(f"Total detections: {len(det_results[0].boxes)}\n\n")
                    for i, box in enumerate(det_results[0].boxes):
                        xyxy = box.xyxy[0].tolist()
                        conf = box.conf.item()
                        f.write(f"Detection {i + 1}:\n")
                        f.write(f"  Confidence: {conf:.2%}\n")
                        f.write(f"  Bounding box: ({xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f})\n")
                        f.write(f"  Size: {xyxy[2] - xyxy[0]:.0f}x{xyxy[3] - xyxy[1]:.0f} pixels\n\n")
                else:
                    f.write("No cracks detected\n\n")

                f.write("SEGMENTATION RESULTS:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Crack regions found: {metrics['num_crack_regions']}\n")
                f.write(f"Total crack pixels: {metrics['total_crack_pixels']:,}\n")
                f.write(f"Total image pixels: {metrics['total_image_pixels']:,}\n")
                f.write(f"Crack coverage: {metrics['crack_percentage']:.3f}%\n\n")

                if metrics['crack_dimensions']:
                    dims = metrics['crack_dimensions']
                    f.write("CRACK DIMENSIONS:\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"Length: {dims['length_mm']:.2f} mm\n")
                    f.write(f"Average width: {dims['avg_width_mm']:.2f} mm\n")
                    f.write(f"Maximum width: {dims['max_width_mm']:.2f} mm\n")
                    f.write(f"Width standard deviation: {dims['width_std_mm']:.2f} mm\n")
                    f.write(f"Area: {dims['area_mm2']:.2f} mm²\n\n")

                if individual_dims:
                    f.write("INDIVIDUAL CRACK DIMENSIONS:\n")
                    f.write("-" * 30 + "\n")
                    for dim in individual_dims:
                        f.write(f"Crack {dim['id']}:\n")
                        f.write(f"  Length: {dim['length_mm']:.2f} mm\n")
                        f.write(f"  Average width: {dim['avg_width_mm']:.2f} mm\n")
                        f.write(f"  Maximum width: {dim['max_width_mm']:.2f} mm\n")
                        f.write(f"  Area: {dim['area_mm2']:.2f} mm²\n\n")

                f.write("DAMAGE ASSESSMENT:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Damage severity: {severity}\n")
                f.write(f"Crack coverage: {metrics['crack_percentage']:.3f}%\n")

            print(f"📄 Report saved to: {report_path}")

            # Save semantic segmentation mask (white cracks on black background)
            if SAVE_SEMANTIC_MASK and seg_results:
                semantic_mask = create_semantic_mask(seg_results, img.shape)
                mask_path = os.path.join(RESULTS_DIR, f"{img_name}_semantic_mask.png")
                cv2.imwrite(mask_path, semantic_mask)
                print(f"⚪ Semantic mask saved to: {mask_path}")

            # Save visualization
            viz_path = os.path.join(RESULTS_DIR, f"{img_name}_combined_analysis.png")
            visualize_combined_results(img, det_results, seg_results,
                                       metrics['crack_dimensions'], viz_path)

            # Save individual result images
            if det_results and len(det_results[0].boxes) > 0:
                det_img = det_results[0].plot()
                det_path = os.path.join(RESULTS_DIR, f"{img_name}_detection_only.png")
                cv2.imwrite(det_path, det_img)
                print(f"🎯 Detection image saved to: {det_path}")

            if seg_results and hasattr(seg_results[0], 'masks') and seg_results[0].masks is not None:
                seg_img = seg_results[0].plot()
                seg_path = os.path.join(RESULTS_DIR, f"{img_name}_segmentation_only.png")
                cv2.imwrite(seg_path, seg_img)
                print(f"🎨 Segmentation image saved to: {seg_path}")

            # Save crack dimensions to CSV
            if individual_dims:
                df = pd.DataFrame(individual_dims)
                csv_path = os.path.join(RESULTS_DIR, f"{img_name}_crack_dimensions.csv")
                df.to_csv(csv_path, index=False)
                print(f"📊 Crack dimensions saved to: {csv_path}")

        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 All results saved to: {RESULTS_DIR}/")

    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


def batch_process_images(image_folder, output_folder="batch_results"):
    """Process multiple images in batch"""
    print(f"\n🔄 BATCH PROCESSING MODE")
    print(f"{'=' * 60}")

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Get all image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(image_folder)
                   if f.lower().endswith(image_extensions)]

    if not image_files:
        print(f"❌ No images found in {image_folder}")
        return

    print(f"📁 Found {len(image_files)} images to process")

    # Load models once
    det_model = YOLO(DETECTION_MODEL) if USE_BOTH_MODELS and os.path.exists(DETECTION_MODEL) else None
    seg_model = YOLO(SEGMENTATION_MODEL) if os.path.exists(SEGMENTATION_MODEL) else None

    if seg_model is None:
        print(f"❌ Segmentation model not found: {SEGMENTATION_MODEL}")
        return

    # Process each image
    results_summary = []

    for i, img_file in enumerate(image_files, 1):
        print(f"\n🔄 Processing {i}/{len(image_files)}: {img_file}")

        img_path = os.path.join(image_folder, img_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ Skipping {img_file} (failed to load)")
            continue

        try:
            # Run detection
            det_results = None
            if det_model:
                det_results = det_model.predict(source=img, conf=CONFIDENCE, save=False, imgsz=IMG_SIZE, verbose=False)

            # Run segmentation
            seg_results = seg_model.predict(source=img, conf=CONFIDENCE, save=False, imgsz=IMG_SIZE, verbose=False)

            # Calculate metrics
            metrics = calculate_crack_metrics(seg_results, img.shape)
            individual_dims = calculate_individual_crack_dimensions(seg_results, img.shape)

            # Save individual results
            img_name = os.path.splitext(img_file)[0]

            # Create subfolder for this image
            img_output_folder = os.path.join(output_folder, img_name)
            os.makedirs(img_output_folder, exist_ok=True)

            # Save semantic segmentation mask
            if SAVE_SEMANTIC_MASK and seg_results:
                semantic_mask = create_semantic_mask(seg_results, img.shape)
                mask_path = os.path.join(img_output_folder, f"{img_name}_semantic_mask.png")
                cv2.imwrite(mask_path, semantic_mask)

            # Save visualization
            viz_path = os.path.join(img_output_folder, f"{img_name}_analysis.png")
            visualize_combined_results(img, det_results, seg_results, metrics['crack_dimensions'], viz_path)

            # Save crack dimensions to CSV
            if individual_dims:
                df = pd.DataFrame(individual_dims)
                csv_path = os.path.join(img_output_folder, f"{img_name}_crack_dimensions.csv")
                df.to_csv(csv_path, index=False)

            # Add to summary
            det_count = len(det_results[0].boxes) if det_results and len(det_results[0].boxes) > 0 else 0

            crack_length = metrics['crack_dimensions']['length_mm'] if metrics['crack_dimensions'] else 0
            crack_area = metrics['crack_dimensions']['area_mm2'] if metrics['crack_dimensions'] else 0

            results_summary.append({
                'filename': img_file,
                'detections': det_count,
                'mask_regions': metrics['num_crack_regions'],
                'crack_pixels': metrics['total_crack_pixels'],
                'coverage_percent': metrics['crack_percentage'],
                'crack_length_mm': crack_length,
                'crack_area_mm2': crack_area
            })

            print(f"✅ Processed: {det_count} detections, {metrics['num_crack_regions']} masks, "
                  f"{metrics['crack_percentage']:.2f}% coverage, {crack_length:.2f}mm length")

        except Exception as e:
            print(f"❌ Error processing {img_file}: {e}")
            continue

    # Save batch summary
    summary_path = os.path.join(output_folder, "batch_summary.csv")
    df_summary = pd.DataFrame(results_summary)
    df_summary.to_csv(summary_path, index=False)

    print(f"\n📊 Batch processing complete!")
    print(f"   Processed: {len(results_summary)}/{len(image_files)} images")
    print(f"   Results saved to: {output_folder}/")
    print(f"   Summary CSV: {summary_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        # Batch processing mode
        folder = sys.argv[2] if len(sys.argv) > 2 else "test_images"
        batch_process_images(folder)
    else:
        # Single image mode
        main()