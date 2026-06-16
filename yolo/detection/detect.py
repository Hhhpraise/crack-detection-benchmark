import cv2
from ultralytics import YOLO
import os

# ===== USER CONFIGURATION =====
MODEL_PATH = 'yolo12s_cbam_ca_crack.pt'
IMAGE_PATH = 'dataset/images/train/0001.png'
CONFIDENCE = 0.5
IMG_SIZE = 224
SAVE_RESULT = True
RESULTS_DIR = 'results'  # Main directory for all saved results


# ==============================

def main():
    print(f"\n{'=' * 50}")
    print(f"🔍 Testing image: {IMAGE_PATH}")
    print(f"⚙️ Confidence: {CONFIDENCE}, Image Size: {IMG_SIZE}")
    print(f"{'=' * 50}")

    try:
        # Create results directory if it doesn't exist
        os.makedirs(RESULTS_DIR, exist_ok=True)

        # Get base image name for saving
        img_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
        output_filename = f"{img_name}_result.png"
        output_path = os.path.join(RESULTS_DIR, output_filename)

        # 1. Load the trained model
        print("⏳ Loading model...")
        model = YOLO(MODEL_PATH)
        print("✅ Model loaded successfully")

        # 2. Load and verify image
        print("⏳ Loading image...")
        img = cv2.imread(IMAGE_PATH)
        if img is None:
            print(f"❌ Error: Failed to load image at {IMAGE_PATH}")
            return
        print(f"📐 Original size: {img.shape[1]}x{img.shape[0]} pixels")

        # 3. Run inference (disable built-in saving)
        print("🔍 Detecting cracks...")
        results = model.predict(
            source=img,
            conf=CONFIDENCE,
            save=False,  # We'll handle saving manually
            imgsz=IMG_SIZE
        )

        # 4. Process results
        result = results[0]
        num_detections = len(result.boxes)
        print(f"🟢 Found {num_detections} crack{'s' if num_detections != 1 else ''}")

        # Print detection details
        if num_detections > 0:
            print("\n🔬 Detection Details:")
            for i, box in enumerate(result.boxes):
                xyxy = box.xyxy[0].tolist()
                conf = box.conf.item()
                print(f"Crack {i + 1}:")
                print(f"  - Confidence: {conf:.2%}")
                print(f"  - Coordinates: x1={xyxy[0]:.0f}, y1={xyxy[1]:.0f}, x2={xyxy[2]:.0f}, y2={xyxy[3]:.0f}")
                print(f"  - Size: {xyxy[2] - xyxy[0]:.0f}x{xyxy[3] - xyxy[1]:.0f} pixels")

        # 5. Save results with image-based naming
        if SAVE_RESULT:
            # Generate annotated image and save
            annotated_img = result.plot()  # Get annotated image array
            cv2.imwrite(output_path, annotated_img)
            print(f"💾 Results saved to: {output_path}")

            # Check if this was an overwrite
            if os.path.exists(output_path):
                print("♻️ Overwritten previous result for this image")

        print("✅ Test completed successfully!\n")

    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")


if __name__ == "__main__":
    main()