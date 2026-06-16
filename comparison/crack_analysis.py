import cv2
import numpy as np
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import sys
from scipy import ndimage
from skimage.morphology import skeletonize
import warnings

warnings.filterwarnings('ignore')

# ===== CALIBRATION PARAMETERS =====
# Camera calibration based on field-of-view estimation
CAMERA_DISTANCE_M = 1.0  # Camera distance from surface (meters)
HORIZONTAL_FOV_DEG = 65.0  # Conservative mid-range FOV (degrees)

# CRITICAL: Use ORIGINAL high-resolution image dimensions
# The 227×227 patches were extracted FROM 4032×3024 images
# The calibration is based on the original capture, not the patch size
IMAGE_WIDTH_PIXELS = 4032  # Original high-res image width
IMAGE_HEIGHT_PIXELS = 3024  # Original high-res image height

# Calculate spatial calibration
# At 1m distance with 65° FOV:
# Horizontal coverage = 2 * tan(FOV/2) * distance = 2 * tan(32.5°) * 1m ≈ 1.28m = 1280mm
# Pixels per mm = IMAGE_WIDTH / horizontal_coverage_mm
HORIZONTAL_COVERAGE_MM = 2 * np.tan(np.radians(HORIZONTAL_FOV_DEG / 2)) * CAMERA_DISTANCE_M * 1000
PIXELS_PER_MM = IMAGE_WIDTH_PIXELS / HORIZONTAL_COVERAGE_MM  # Approximately 3.16 pixels/mm
MM_PER_PIXEL = 1 / PIXELS_PER_MM  # Approximately 0.316 mm/pixel

print(f"Camera Calibration:")
print(f"  Distance: {CAMERA_DISTANCE_M}m")
print(f"  Horizontal FOV: {HORIZONTAL_FOV_DEG}°")
print(f"  Original image size: {IMAGE_WIDTH_PIXELS}×{IMAGE_HEIGHT_PIXELS} pixels")
print(f"  Horizontal coverage: {HORIZONTAL_COVERAGE_MM:.2f}mm")
print(f"  Spatial resolution: {PIXELS_PER_MM:.2f} pixels/mm ({MM_PER_PIXEL:.3f} mm/pixel)")
print(f"  Each 227×227 patch represents: {227 * MM_PER_PIXEL:.1f}mm × {227 * MM_PER_PIXEL:.1f}mm")
print()

# ===== USER CONFIGURATION =====
DETECTION_MODEL = 'yolo12s_cbam_ca_crack.pt'
SEGMENTATION_MODEL = 'yolo12s_seg_cbam_ca_crack.pt'
CONFIDENCE = 0.5
IMG_SIZE = 224
BATCH_OUTPUT_DIR = 'batch_analysis_results'


def calculate_crack_dimensions(mask, pixels_per_mm=PIXELS_PER_MM, mm_per_pixel=MM_PER_PIXEL):
    """
    Calculate crack dimensions using calibrated spatial resolution

    Methodology:
    - Length: Calculated via skeletonization (centerline of crack)
    - Width: Computed using distance transform (perpendicular measurements)
    - Area: Total crack pixels converted to mm²
    - All measurements use calibrated FOV-based conversion factor
    """
    # Ensure mask is binary
    if mask.dtype != bool:
        binary_mask = mask > 0
    else:
        binary_mask = mask

    # Calculate area
    area_pixels = np.sum(binary_mask)
    area_mm2 = area_pixels * (mm_per_pixel ** 2)

    # Skeletonize to calculate length
    try:
        skeleton = skeletonize(binary_mask)
        length_pixels = np.sum(skeleton)
        length_mm = length_pixels * mm_per_pixel
    except Exception as e:
        # Fallback: estimate length from area (assuming linear crack)
        length_pixels = np.sqrt(area_pixels)
        length_mm = length_pixels * mm_per_pixel

    # Calculate average width
    if length_pixels > 0:
        avg_width_pixels = area_pixels / length_pixels
        avg_width_mm = avg_width_pixels * mm_per_pixel
    else:
        avg_width_pixels = 0
        avg_width_mm = 0

    # Calculate maximum width using distance transform
    if area_pixels > 0:
        # Distance transform gives distance to nearest edge
        distance_transform = ndimage.distance_transform_edt(binary_mask)
        max_width_pixels = np.max(distance_transform) * 2  # Diameter = 2 × radius
        max_width_mm = max_width_pixels * mm_per_pixel

        # Calculate standard deviation of width
        width_values = distance_transform[binary_mask] * 2
        width_std_pixels = np.std(width_values)
        width_std_mm = width_std_pixels * mm_per_pixel
    else:
        max_width_pixels = 0
        max_width_mm = 0
        width_std_mm = 0

    # Calculate coverage percentage
    total_pixels = mask.shape[0] * mask.shape[1]
    coverage_percent = (area_pixels / total_pixels) * 100 if total_pixels > 0 else 0

    return {
        'length_mm': length_mm,
        'avg_width_mm': avg_width_mm,
        'max_width_mm': max_width_mm,
        'width_std_mm': width_std_mm,
        'area_mm2': area_mm2,
        'coverage_percent': coverage_percent,
        'length_pixels': length_pixels,
        'avg_width_pixels': avg_width_pixels,
        'max_width_pixels': max_width_pixels,
        'area_pixels': area_pixels
    }


def create_semantic_mask(seg_results, image_shape):
    """
    Create a semantic segmentation mask with white cracks on black background
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    if seg_results and hasattr(seg_results[0], 'masks') and seg_results[0].masks is not None:
        for i, mask_data in enumerate(seg_results[0].masks.xy):
            polygon = np.array(mask_data, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [polygon], 255)

    return mask


def analyze_single_image(image_path, seg_model, det_model=None):
    """
    Analyze a single image and return calibrated measurements
    """
    print(f"  Processing: {os.path.basename(image_path)}")

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"  ❌ Failed to load image: {image_path}")
        return None

    # Get actual image dimensions
    actual_height, actual_width = img.shape[:2]

    # ===== CRITICAL FIX: DO NOT ADJUST CALIBRATION =====
    # The calibration is based on the ORIGINAL capture conditions (4032×3024 at 1m).
    # The 227×227 patches were extracted/cropped from these original images.
    # When you crop an image, the physical scale DOES NOT CHANGE - the same number
    # of pixels still represents the same physical distance.
    #
    # Example: If you photograph a 1-meter ruler from 1m away, then crop out just
    # the 10cm section, those pixels still represent 10cm - you didn't shrink the
    # ruler by cropping the photo!
    #
    # Therefore: Use the ORIGINAL calibration factor regardless of patch size.

    adjusted_pixels_per_mm = PIXELS_PER_MM  # Fixed at 3.16 pixels/mm
    adjusted_mm_per_pixel = MM_PER_PIXEL  # Fixed at 0.316 mm/pixel

    # Log image info (but don't adjust calibration based on it)
    if actual_width != IMAGE_WIDTH_PIXELS or actual_height != IMAGE_HEIGHT_PIXELS:
        patch_size_mm = actual_width * MM_PER_PIXEL
        print(
            f"    Image: {actual_width}×{actual_height}px patch (from {IMAGE_WIDTH_PIXELS}×{IMAGE_HEIGHT_PIXELS}px original)")
        print(f"    Patch represents: ~{patch_size_mm:.1f}mm × {patch_size_mm:.1f}mm of concrete surface")

    # Run segmentation model
    seg_results = seg_model.predict(
        source=img,
        conf=CONFIDENCE,
        save=False,
        imgsz=IMG_SIZE,
        verbose=False
    )

    # Create combined mask from all segments
    semantic_mask = create_semantic_mask(seg_results, img.shape)

    # Calculate crack dimensions with calibrated measurements
    if np.sum(semantic_mask) > 0:
        dimensions = calculate_crack_dimensions(
            semantic_mask,
            adjusted_pixels_per_mm,
            adjusted_mm_per_pixel
        )

        # Calculate detection confidence
        if det_model is not None:
            det_results = det_model.predict(
                source=img,
                conf=CONFIDENCE,
                save=False,
                imgsz=IMG_SIZE,
                verbose=False
            )
            if len(det_results[0].boxes) > 0:
                confidences = [box.conf.item() for box in det_results[0].boxes]
                dimensions['confidence_percent'] = np.mean(confidences) * 100
            else:
                dimensions['confidence_percent'] = 0
        else:
            # Use segmentation confidence if available
            if hasattr(seg_results[0], 'boxes') and seg_results[0].boxes is not None and len(seg_results[0].boxes) > 0:
                confidences = [box.conf.item() for box in seg_results[0].boxes]
                dimensions['confidence_percent'] = np.mean(confidences) * 100 if confidences else 0
            else:
                dimensions['confidence_percent'] = 0

        dimensions['image_name'] = os.path.basename(image_path)
        dimensions['calibration_px_per_mm'] = adjusted_pixels_per_mm

        return dimensions
    else:
        # No crack detected
        return {
            'image_name': os.path.basename(image_path),
            'length_mm': 0,
            'avg_width_mm': 0,
            'max_width_mm': 0,
            'width_std_mm': 0,
            'area_mm2': 0,
            'coverage_percent': 0,
            'confidence_percent': 0,
            'calibration_px_per_mm': adjusted_pixels_per_mm
        }


def batch_analyze_images(image_folder):
    """
    Analyze all images in a folder using calibrated measurements
    """
    print("\n" + "=" * 100)
    print("BATCH CRACK ANALYSIS - CALIBRATED MEASUREMENTS")
    print("=" * 100 + "\n")

    # Create output directory
    os.makedirs(BATCH_OUTPUT_DIR, exist_ok=True)

    # Load models
    print("📦 Loading models...")
    try:
        seg_model = YOLO(SEGMENTATION_MODEL)
        print(f"  ✅ Segmentation model loaded: {SEGMENTATION_MODEL}")
    except Exception as e:
        print(f"  ❌ Failed to load segmentation model: {e}")
        return None

    try:
        det_model = YOLO(DETECTION_MODEL)
        print(f"  ✅ Detection model loaded: {DETECTION_MODEL}")
    except Exception as e:
        print(f"  ⚠️  Detection model not loaded (confidence scores will use segmentation): {e}")
        det_model = None

    # Get list of images
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(image_folder)
                   if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"❌ No images found in {image_folder}")
        return None

    print(f"\n📁 Found {len(image_files)} images to process")
    print(f"🔍 Confidence threshold: {CONFIDENCE}")
    print(f"📏 Calibration: {PIXELS_PER_MM:.2f} pixels/mm ({MM_PER_PIXEL:.3f} mm/pixel)")
    print("\n" + "-" * 100)

    # Process each image
    results = []
    for idx, image_file in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Processing: {image_file}")
        image_path = os.path.join(image_folder, image_file)

        result = analyze_single_image(image_path, seg_model, det_model)
        if result:
            results.append(result)

            # Display measurements
            if result['length_mm'] > 0:
                print(f"  ✅ Crack detected:")
                print(f"     Length: {result['length_mm']:.2f} mm")
                print(f"     Avg Width: {result['avg_width_mm']:.2f} mm")
                print(f"     Max Width: {result['max_width_mm']:.2f} mm")
                print(f"     Area: {result['area_mm2']:.2f} mm²")
                print(f"     Coverage: {result['coverage_percent']:.2f}%")
                print(f"     Confidence: {result['confidence_percent']:.2f}%")
            else:
                print(f"  ⚪ No crack detected")

    print("\n" + "=" * 100)

    if not results:
        print("❌ No results to save")
        return None

    # Create DataFrame
    df = pd.DataFrame(results)

    # Reorder and rename columns for output
    df_output = pd.DataFrame({
        'Image ID': range(1, len(df) + 1),
        'Image Name': df['image_name'],
        'Length (mm)': df['length_mm'].round(2),
        'Avg Width (mm)': df['avg_width_mm'].round(2),
        'Max Width (mm)': df['max_width_mm'].round(2),
        'Width StdDev (mm)': df['width_std_mm'].round(2),
        'Area (mm²)': df['area_mm2'].round(2),
        'Coverage (%)': df['coverage_percent'].round(2),
        'Confidence (%)': df['confidence_percent'].round(2),
        'Calibration (px/mm)': df['calibration_px_per_mm'].round(2)
    })

    # Save CSV
    csv_path = os.path.join(BATCH_OUTPUT_DIR, 'crack_measurements_results.csv')
    df_output.to_csv(csv_path, index=False)
    print(f"\n💾 Results saved to: {csv_path}")

    # Try to save Excel if openpyxl is available
    try:
        excel_path = os.path.join(BATCH_OUTPUT_DIR, 'crack_measurements_results.xlsx')
        df_output.to_excel(excel_path, index=False, sheet_name='Crack Measurements')
        print(f"💾 Excel file saved to: {excel_path}")
    except ImportError:
        print("⚠️  Excel export skipped (install openpyxl for .xlsx support)")

    # Print summary statistics
    crack_df = df_output[df_output['Length (mm)'] > 0]
    if len(crack_df) > 0:
        print("\n" + "=" * 100)
        print("SUMMARY STATISTICS (Calibrated Measurements)")
        print("=" * 100)
        print(f"\nTotal images analyzed: {len(df_output)}")
        print(f"Images with cracks: {len(crack_df)} ({len(crack_df) / len(df_output) * 100:.1f}%)")
        print(f"Images without cracks: {len(df_output) - len(crack_df)}")

        print(f"\n📏 CRACK LENGTHS:")
        print(f"  Mean: {crack_df['Length (mm)'].mean():.2f} mm")
        print(f"  Std Dev: {crack_df['Length (mm)'].std():.2f} mm")
        print(f"  Min: {crack_df['Length (mm)'].min():.2f} mm")
        print(f"  Max: {crack_df['Length (mm)'].max():.2f} mm")

        print(f"\n📐 CRACK WIDTHS (Average):")
        print(f"  Mean: {crack_df['Avg Width (mm)'].mean():.2f} mm")
        print(f"  Std Dev: {crack_df['Avg Width (mm)'].std():.2f} mm")
        print(f"  Min: {crack_df['Avg Width (mm)'].min():.2f} mm")
        print(f"  Max: {crack_df['Avg Width (mm)'].max():.2f} mm")

        print(f"\n📐 CRACK WIDTHS (Maximum):")
        print(f"  Mean: {crack_df['Max Width (mm)'].mean():.2f} mm")
        print(f"  Std Dev: {crack_df['Max Width (mm)'].std():.2f} mm")
        print(f"  Min: {crack_df['Max Width (mm)'].min():.2f} mm")
        print(f"  Max: {crack_df['Max Width (mm)'].max():.2f} mm")

        print(f"\n📦 CRACK AREAS:")
        print(f"  Mean: {crack_df['Area (mm²)'].mean():.2f} mm²")
        print(f"  Std Dev: {crack_df['Area (mm²)'].std():.2f} mm²")
        print(f"  Min: {crack_df['Area (mm²)'].min():.2f} mm²")
        print(f"  Max: {crack_df['Area (mm²)'].max():.2f} mm²")

        print(f"\n🎯 DETECTION CONFIDENCE:")
        print(f"  Mean: {crack_df['Confidence (%)'].mean():.2f}%")
        print(f"  Min: {crack_df['Confidence (%)'].min():.2f}%")
        print(f"  Max: {crack_df['Confidence (%)'].max():.2f}%")

    # Generate detailed report
    report_path = os.path.join(BATCH_OUTPUT_DIR, 'analysis_report.txt')
    generate_detailed_report(df_output, report_path)
    print(f"\n📄 Detailed report saved to: {report_path}")

    return df_output


def generate_detailed_report(df, output_path):
    """
    Generate a detailed text report with calibration methodology
    """
    crack_df = df[df['Length (mm)'] > 0]

    with open(output_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("CRACK DETECTION AND MEASUREMENT ANALYSIS REPORT\n")
        f.write("CALIBRATED MEASUREMENTS USING FOV-BASED SPATIAL RESOLUTION\n")
        f.write("=" * 100 + "\n\n")

        f.write("CALIBRATION METHODOLOGY\n")
        f.write("-" * 100 + "\n\n")
        f.write(f"Camera Distance: {CAMERA_DISTANCE_M} meter\n")
        f.write(f"Horizontal Field of View: {HORIZONTAL_FOV_DEG}°\n")
        f.write(f"Original Image Resolution: {IMAGE_WIDTH_PIXELS} × {IMAGE_HEIGHT_PIXELS} pixels\n")
        f.write(f"Horizontal Coverage at 1m: {HORIZONTAL_COVERAGE_MM:.2f} mm\n")
        f.write(f"Calibration Factor: {PIXELS_PER_MM:.2f} pixels/mm\n")
        f.write(f"Spatial Resolution: {MM_PER_PIXEL:.3f} mm per pixel\n\n")

        f.write("IMPORTANT NOTES:\n")
        f.write("- The calibration is based on original high-resolution images (4032×3024 pixels)\n")
        f.write("- The analyzed patches (227×227 pixels) were extracted from these originals\n")
        f.write("- Cropping does not change pixel scale - calibration remains constant\n")
        f.write("- Each 227×227 patch represents approximately 72mm × 72mm of concrete surface\n\n")

        f.write("=" * 100 + "\n")
        f.write("ANALYSIS SUMMARY\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Total Images Analyzed: {len(df)}\n")
        f.write(f"Images with Cracks Detected: {len(crack_df)} ({len(crack_df) / len(df) * 100:.1f}%)\n")
        f.write(f"Images without Cracks: {len(df) - len(crack_df)}\n\n")

        if len(crack_df) > 0:
            f.write("=" * 100 + "\n")
            f.write("STATISTICAL SUMMARY (CALIBRATED MEASUREMENTS)\n")
            f.write("=" * 100 + "\n\n")

            f.write("CRACK LENGTH (mm):\n")
            f.write(f"  Mean ± Std Dev: {crack_df['Length (mm)'].mean():.2f} ± {crack_df['Length (mm)'].std():.2f}\n")
            f.write(f"  Range: [{crack_df['Length (mm)'].min():.2f}, {crack_df['Length (mm)'].max():.2f}]\n")
            f.write(f"  Median: {crack_df['Length (mm)'].median():.2f}\n\n")

            f.write("AVERAGE CRACK WIDTH (mm):\n")
            f.write(
                f"  Mean ± Std Dev: {crack_df['Avg Width (mm)'].mean():.2f} ± {crack_df['Avg Width (mm)'].std():.2f}\n")
            f.write(f"  Range: [{crack_df['Avg Width (mm)'].min():.2f}, {crack_df['Avg Width (mm)'].max():.2f}]\n")
            f.write(f"  Median: {crack_df['Avg Width (mm)'].median():.2f}\n\n")

            f.write("MAXIMUM CRACK WIDTH (mm):\n")
            f.write(
                f"  Mean ± Std Dev: {crack_df['Max Width (mm)'].mean():.2f} ± {crack_df['Max Width (mm)'].std():.2f}\n")
            f.write(f"  Range: [{crack_df['Max Width (mm)'].min():.2f}, {crack_df['Max Width (mm)'].max():.2f}]\n")
            f.write(f"  Median: {crack_df['Max Width (mm)'].median():.2f}\n\n")

            f.write("CRACK AREA (mm²):\n")
            f.write(f"  Mean ± Std Dev: {crack_df['Area (mm²)'].mean():.2f} ± {crack_df['Area (mm²)'].std():.2f}\n")
            f.write(f"  Range: [{crack_df['Area (mm²)'].min():.2f}, {crack_df['Area (mm²)'].max():.2f}]\n")
            f.write(f"  Median: {crack_df['Area (mm²)'].median():.2f}\n\n")

            f.write("=" * 100 + "\n")
            f.write("DETAILED MEASUREMENTS BY IMAGE\n")
            f.write("=" * 100 + "\n\n")

            for idx, row in crack_df.iterrows():
                f.write(f"Image {row['Image ID']}: {row['Image Name']}\n")
                f.write(f"  Length: {row['Length (mm)']:.2f} mm\n")
                f.write(f"  Average Width: {row['Avg Width (mm)']:.2f} mm\n")
                f.write(f"  Maximum Width: {row['Max Width (mm)']:.2f} mm\n")
                f.write(f"  Width Std Dev: {row['Width StdDev (mm)']:.2f} mm\n")
                f.write(f"  Area: {row['Area (mm²)']:.2f} mm²\n")
                f.write(f"  Coverage: {row['Coverage (%)']:.2f}%\n")
                f.write(f"  Confidence: {row['Confidence (%)']:.2f}%\n")
                f.write("\n")

        f.write("=" * 100 + "\n")
        f.write("MEASUREMENT METHODOLOGY NOTES\n")
        f.write("=" * 100 + "\n\n")

        f.write("1. CRACK LENGTH:\n")
        f.write("   - Calculated using morphological skeletonization\n")
        f.write("   - Represents the centerline path of the crack\n")
        f.write("   - Accounts for crack curvature and tortuosity\n\n")

        f.write("2. CRACK WIDTH:\n")
        f.write("   - Average Width: Computed as total area divided by length\n")
        f.write("   - Maximum Width: Determined using Euclidean distance transform\n")
        f.write("   - Distance transform measures distance from crack edge to centerline\n")
        f.write("   - Width = 2 × maximum distance (accounts for both sides)\n\n")

        f.write("3. CRACK AREA:\n")
        f.write("   - Total surface area of detected crack pixels\n")
        f.write("   - Converted from pixels² to mm² using calibrated scale factor\n\n")

        f.write("4. WIDTH STANDARD DEVIATION:\n")
        f.write("   - Measures variability of crack width along its length\n")
        f.write("   - Computed from distance transform values at all crack pixels\n")
        f.write("   - High std dev indicates non-uniform crack opening\n\n")

        f.write("5. COVERAGE PERCENTAGE:\n")
        f.write("   - Coverage = (Crack pixels / Total image pixels) × 100%\n")
        f.write("   - Represents proportion of image area occupied by cracks\n")
        f.write("   - Useful for assessing damage density\n\n")

        f.write("6. CONFIDENCE SCORE:\n")
        f.write("   - Represents deep learning model certainty in crack detection\n")
        f.write("   - Based on average confidence of all detections in the image\n")
        f.write("   - Higher confidence indicates more reliable measurements\n")

        f.write("\n" + "=" * 100 + "\n")
        f.write("CALIBRATION VALIDATION NOTES\n")
        f.write("=" * 100 + "\n\n")

        f.write("The FOV-based calibration method provides reasonable estimates for crack measurements\n")
        f.write("based on documented camera distance from the METU dataset paper. Key assumptions:\n\n")
        f.write("  • Original images: 4032×3024 pixels captured at ~1m from surface\n")
        f.write("  • Analyzed patches: 227×227 pixels extracted from originals\n")
        f.write("  • Camera FOV: 65° (conservative mid-range for typical inspection cameras)\n")
        f.write("  • Calibration applies uniformly: cropping doesn't change pixel scale\n\n")

        f.write("Measurement uncertainty is estimated at ±15% due to:\n")
        f.write("  • FOV estimation (actual camera FOV may vary 60-70°)\n")
        f.write("  • Approximate camera distance (~1m)\n")
        f.write("  • Image segmentation accuracy\n\n")

        f.write("For validation or higher accuracy:\n")
        f.write("  • Contact dataset authors for exact camera specifications\n")
        f.write("  • Perform checkerboard calibration if recapturing images\n")
        f.write("  • Include known reference objects for scale verification\n")

        f.write("\n" + "=" * 100 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 100 + "\n")


def visualize_results(df):
    """
    Create visualization plots for the analysis results
    """
    if len(df) == 0 or len(df[df['Length (mm)'] > 0]) == 0:
        print("⚠️  No cracks detected - skipping visualizations")
        return

    # Filter only images with cracks for visualization
    crack_df = df[df['Length (mm)'] > 0].copy()

    # Create visualization directory
    viz_dir = os.path.join(BATCH_OUTPUT_DIR, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    # 1. Bar chart of crack lengths
    plt.figure(figsize=(14, 6))
    colors = ['red' if x > 80 else 'orange' if x > 50 else 'green'
              for x in crack_df['Length (mm)']]
    plt.bar(range(len(crack_df)), crack_df['Length (mm)'], color=colors, alpha=0.7, edgecolor='black')
    plt.axhline(y=80, color='red', linestyle='--', label='Severe threshold (80mm)', alpha=0.5)
    plt.axhline(y=50, color='orange', linestyle='--', label='Moderate threshold (50mm)', alpha=0.5)
    plt.title('Crack Lengths by Image (Calibrated Measurements)', fontsize=14, fontweight='bold')
    plt.xlabel('Image Index', fontsize=12)
    plt.ylabel('Length (mm)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'crack_lengths.png'), dpi=300)
    plt.close()

    # 2. Scatter plot: Length vs Width with severity zones
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(crack_df['Length (mm)'], crack_df['Avg Width (mm)'],
                          c=crack_df['Confidence (%)'], cmap='viridis', s=100, alpha=0.7,
                          edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, label='Confidence (%)')

    # Add severity zones (adjusted for realistic calibration)
    plt.axvline(x=80, color='red', linestyle='--', alpha=0.3, label='Severe length')
    plt.axhline(y=10, color='red', linestyle='--', alpha=0.3, label='Severe width')

    plt.title('Crack Length vs Average Width (Calibrated)', fontsize=14, fontweight='bold')
    plt.xlabel('Length (mm)', fontsize=12)
    plt.ylabel('Average Width (mm)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'length_vs_width.png'), dpi=300)
    plt.close()

    # 3. Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Length histogram
    axes[0, 0].hist(crack_df['Length (mm)'], bins=15, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].axvline(crack_df['Length (mm)'].mean(), color='red', linestyle='--',
                       label=f'Mean: {crack_df["Length (mm)"].mean():.1f}mm')
    axes[0, 0].set_title('Crack Length Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Length (mm)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Width histogram
    axes[0, 1].hist(crack_df['Avg Width (mm)'], bins=15, edgecolor='black', alpha=0.7, color='coral')
    axes[0, 1].axvline(crack_df['Avg Width (mm)'].mean(), color='red', linestyle='--',
                       label=f'Mean: {crack_df["Avg Width (mm)"].mean():.2f}mm')
    axes[0, 1].set_title('Average Width Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Width (mm)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Area histogram
    axes[1, 0].hist(crack_df['Area (mm²)'], bins=15, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[1, 0].axvline(crack_df['Area (mm²)'].mean(), color='red', linestyle='--',
                       label=f'Mean: {crack_df["Area (mm²)"].mean():.1f}mm²')
    axes[1, 0].set_title('Crack Area Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Area (mm²)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Confidence histogram
    axes[1, 1].hist(crack_df['Confidence (%)'], bins=15, edgecolor='black', alpha=0.7, color='gold')
    axes[1, 1].axvline(crack_df['Confidence (%)'].mean(), color='red', linestyle='--',
                       label=f'Mean: {crack_df["Confidence (%)"].mean():.1f}%')
    axes[1, 1].set_title('Confidence Score Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('Confidence (%)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'measurement_distributions.png'), dpi=300)
    plt.close()

    # 4. Box plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    box_data = [crack_df['Length (mm)'], crack_df['Avg Width (mm)'], crack_df['Area (mm²)']]
    box_labels = ['Length (mm)', 'Avg Width (mm)', 'Area (mm²)']
    colors_box = ['steelblue', 'coral', 'lightgreen']

    for i, (data, label, color) in enumerate(zip(box_data, box_labels, colors_box)):
        bp = axes[i].boxplot(data, patch_artist=True)
        bp['boxes'][0].set_facecolor(color)
        axes[i].set_ylabel(label, fontweight='bold')
        axes[i].set_title(f'{label} Statistics', fontweight='bold')
        axes[i].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'box_plots.png'), dpi=300)
    plt.close()

    # 5. Severity classification pie chart (adjusted thresholds)
    severe = len(crack_df[crack_df['Length (mm)'] > 80])
    moderate = len(crack_df[(crack_df['Length (mm)'] > 50) & (crack_df['Length (mm)'] <= 80)])
    minor = len(crack_df[crack_df['Length (mm)'] <= 50])

    if severe + moderate + minor > 0:
        plt.figure(figsize=(8, 8))
        sizes = [severe, moderate, minor]
        labels = [f'Severe (>80mm)\n{severe} cracks',
                  f'Moderate (50-80mm)\n{moderate} cracks',
                  f'Minor (<50mm)\n{minor} cracks']
        colors_pie = ['#ff4444', '#ffaa44', '#44ff44']
        explode = (0.1, 0.05, 0)

        plt.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                autopct='%1.1f%%', shadow=True, startangle=90)
        plt.title('Crack Severity Classification (Calibrated)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'severity_classification.png'), dpi=300)
        plt.close()

    print(f"📈 Visualizations saved to: {viz_dir}/")


def main():
    """
    Main function to handle command line arguments
    """
    if len(sys.argv) < 2:
        print("\nUsage: python crack_analysis_corrected.py <image_folder_path>")
        print("Example: python crack_analysis_corrected.py ./test_images")
        print("\nOptional: Install openpyxl for Excel export")
        print("  pip install openpyxl")
        return

    image_folder = sys.argv[1]

    if not os.path.exists(image_folder):
        print(f"❌ Error: Folder '{image_folder}' does not exist")
        return

    # Run batch analysis with calibrated measurements
    results_df = batch_analyze_images(image_folder)

    # Create visualizations if we have results
    if results_df is not None and len(results_df) > 0:
        print("\n📊 Generating visualizations...")
        visualize_results(results_df)
        print("✅ Visualization complete!")


if __name__ == "__main__":
    main()
