"""
FIXED UNIFIED YOLO TRAINER - Uses Proper Dual Datasets
Trains YOLO detection and segmentation models with their respective label formats
NOW WITH: Memory optimization, resume support, and error recovery
"""

import os
from ultralytics import YOLO
import yaml
import json
import torch
import gc
from datetime import datetime


class FixedUnifiedYOLOTrainer:
    def __init__(self, detection_yaml, segmentation_yaml, output_dir='yolo_experiments'):
        """
        Initialize unified YOLO trainer with separate datasets

        Args:
            detection_yaml: Path to detection dataset YAML (bbox labels)
            segmentation_yaml: Path to segmentation dataset YAML (polygon labels)
            output_dir: Directory to save all experiment results
        """
        self.detection_yaml = detection_yaml
        self.segmentation_yaml = segmentation_yaml
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Available YOLO models
        self.detection_models = {
            'yolo8s': 'yolov8s.pt',
            'yolo11s': 'yolo11s.pt',
            'yolo12s': 'yolo12s.pt',
        }

        self.segmentation_models = {
            'yolo8s-seg': 'yolov8s-seg.pt',
            'yolo11s-seg': 'yolo11s-seg.pt',
        }

        self.results_log = []

    def clear_gpu_memory(self):
        """Clear GPU memory to prevent OOM errors"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print("   GPU memory cleared")

    def train_detection(self, model_name, epochs=100, imgsz=192, batch=16,
                        device='0', patience=20, **kwargs):
        """Train YOLO detection model with bbox labels"""
        if model_name not in self.detection_models:
            print(f"❌ Model {model_name} not found!")
            return None

        print("\n" + "="*100)
        print(f"🚀 TRAINING DETECTION: {model_name}")
        print("="*100)

        model = YOLO(self.detection_models[model_name])
        exp_name = f"{model_name}_det_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        train_args = {
            'data': self.detection_yaml,  # Uses bbox labels
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'device': device,
            'patience': patience,
            'project': self.output_dir,
            'name': exp_name,
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'Adam',
            'verbose': True,
            'seed': 42,
            'save': True,
            'plots': True,
            'val': True,
            'workers': 8,  # Reduced from 16 to prevent memory issues
        }
        train_args.update(kwargs)

        print(f"\n📋 Configuration:")
        print(f"   Dataset: {self.detection_yaml}")
        print(f"   Epochs: {epochs}, Image size: {imgsz}, Batch: {batch}")
        print(f"   Device: {device}")

        try:
            results = model.train(**train_args)

            best_model_path = os.path.join(self.output_dir, exp_name, 'weights', 'best.pt')
            result_summary = {
                'model_name': model_name,
                'task': 'detection',
                'timestamp': datetime.now().isoformat(),
                'best_model': best_model_path,
                'config': train_args,
                'metrics': {
                    'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
                    'mAP50-95': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
                    'precision': float(results.results_dict.get('metrics/precision(B)', 0)),
                    'recall': float(results.results_dict.get('metrics/recall(B)', 0)),
                }
            }

            self.results_log.append(result_summary)

            print(f"\n✅ {model_name} Detection Training Complete!")
            print(f"   Best model: {best_model_path}")
            print(f"   mAP50: {result_summary['metrics']['mAP50']:.4f}")
            print(f"   mAP50-95: {result_summary['metrics']['mAP50-95']:.4f}")

            # Clear memory after training
            del model
            self.clear_gpu_memory()

            return results

        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()

            # Try to clear memory on failure
            try:
                del model
                self.clear_gpu_memory()
            except:
                pass

            return None

    def train_segmentation(self, model_name, epochs=100, imgsz=192, batch=16,
                           device='0', patience=20, **kwargs):
        """Train YOLO segmentation model with polygon labels"""
        if model_name not in self.segmentation_models:
            print(f"❌ Model {model_name} not found!")
            return None

        print("\n" + "="*100)
        print(f"🚀 TRAINING SEGMENTATION: {model_name}")
        print("="*100)

        # Clear memory before starting
        self.clear_gpu_memory()

        model = YOLO(self.segmentation_models[model_name])
        exp_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        train_args = {
            'data': self.segmentation_yaml,  # Uses polygon labels
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'device': device,
            'patience': patience,
            'project': self.output_dir,
            'name': exp_name,
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'Adam',
            'verbose': True,
            'seed': 42,
            'save': True,
            'plots': True,
            'val': True,
            'workers': 4,  # Reduced further for segmentation (memory intensive)
            'cache': False,  # Don't cache to save memory
        }
        train_args.update(kwargs)

        print(f"\n📋 Configuration:")
        print(f"   Dataset: {self.segmentation_yaml}")
        print(f"   Epochs: {epochs}, Image size: {imgsz}, Batch: {batch}")
        print(f"   Device: {device}")
        print(f"   Workers: {train_args['workers']} (reduced for memory)")

        try:
            results = model.train(**train_args)

            best_model_path = os.path.join(self.output_dir, exp_name, 'weights', 'best.pt')
            result_summary = {
                'model_name': model_name,
                'task': 'segmentation',
                'timestamp': datetime.now().isoformat(),
                'best_model': best_model_path,
                'config': train_args,
                'metrics': {
                    'box_mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
                    'box_mAP50-95': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
                    'mask_mAP50': float(results.results_dict.get('metrics/mAP50(M)', 0)),
                    'mask_mAP50-95': float(results.results_dict.get('metrics/mAP50-95(M)', 0)),
                }
            }

            self.results_log.append(result_summary)

            print(f"\n✅ {model_name} Segmentation Training Complete!")
            print(f"   Best model: {best_model_path}")
            print(f"   Box mAP50: {result_summary['metrics']['box_mAP50']:.4f}")
            print(f"   Mask mAP50: {result_summary['metrics']['mask_mAP50']:.4f}")

            # Clear memory after training
            del model
            self.clear_gpu_memory()

            return results

        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()

            # Try to clear memory on failure
            try:
                del model
                self.clear_gpu_memory()
            except:
                pass

            return None

    def find_latest_detection_model(self, model_prefix='yolo11s_det'):
        """Find the latest trained detection model"""
        import glob
        pattern = os.path.join(self.output_dir, f"{model_prefix}_*/weights/best.pt")
        models = glob.glob(pattern)

        if models:
            latest = sorted(models)[-1]
            print(f"\n✅ Found existing detection model: {latest}")
            return latest
        return None

    def train_both_models(self, det_model='yolo11s', seg_model='yolo11s-seg',
                         epochs=150, imgsz=192, batch=16, patience=30,
                         skip_detection=False, **kwargs):
        """
        Train both detection and segmentation models

        Args:
            det_model: Detection model name
            seg_model: Segmentation model name
            epochs: Training epochs
            imgsz: Image size
            batch: Batch size
            patience: Early stopping patience
            skip_detection: If True, skip detection training and only train segmentation
            **kwargs: Additional training args
        """
        print("\n" + "="*100)
        print("🔥 TRAINING BOTH DETECTION + SEGMENTATION MODELS")
        print("="*100)

        # Check available GPUs
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        if n_gpus >= 2:
            print(f"\n✅ Found {n_gpus} GPUs - Will use GPU 0 and GPU 1")
            det_device = '0'
            seg_device = '1'
        elif n_gpus == 1:
            print(f"\n⚠️ Only 1 GPU found - Training sequentially on same GPU")
            det_device = '0'
            seg_device = '0'
        else:
            print(f"\n⚠️ No GPU found - Training on CPU (this will be slow!)")
            det_device = 'cpu'
            seg_device = 'cpu'

        det_result = None

        # Train or skip detection model
        if skip_detection:
            print(f"\n⏭️ SKIPPING Detection Training (as requested)")
            existing_det = self.find_latest_detection_model(f'{det_model}_det')
            if existing_det:
                print(f"   Will use existing detection model for comparison")
            else:
                print(f"   ⚠️ Warning: No existing detection model found!")
        else:
            print(f"\n📦 STEP 1: Training Detection Model on device {det_device}...")
            det_result = self.train_detection(
                det_model,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=det_device,
                patience=patience,
                **kwargs
            )

        # Train segmentation model
        print(f"\n📦 STEP 2: Training Segmentation Model on device {seg_device}...")

        # Reduce batch size for segmentation if memory issues
        seg_batch = max(8, batch // 2)  # Use half the batch size or minimum 8
        print(f"   Using reduced batch size for segmentation: {seg_batch}")

        seg_result = self.train_segmentation(
            seg_model,
            epochs=epochs,
            imgsz=imgsz,
            batch=seg_batch,  # Reduced batch size
            device=seg_device,
            patience=patience,
            **kwargs
        )

        # Save summary
        self.save_training_summary()

        return det_result, seg_result

    def save_training_summary(self):
        """Save training summary to JSON"""
        summary_path = os.path.join(self.output_dir, 'training_summary.json')

        with open(summary_path, 'w') as f:
            json.dump(self.results_log, f, indent=4)

        print(f"\n✅ Summary saved: {summary_path}")
        self.print_comparison_table()

    def print_comparison_table(self):
        """Print comparison table"""
        if not self.results_log:
            print("No results to compare yet.")
            return

        print("\n" + "="*100)
        print("TRAINING RESULTS SUMMARY")
        print("="*100)

        # Detection models
        det_models = [r for r in self.results_log if r['task'] == 'detection']
        if det_models:
            print("\n🔍 DETECTION MODELS:")
            print(f"{'Model':<20} {'mAP50':<12} {'mAP50-95':<12} {'Precision':<12} {'Recall':<12}")
            print("-"*100)
            for r in det_models:
                m = r['metrics']
                print(f"{r['model_name']:<20} {m.get('mAP50',0):<12.4f} "
                      f"{m.get('mAP50-95',0):<12.4f} {m.get('precision',0):<12.4f} "
                      f"{m.get('recall',0):<12.4f}")

        # Segmentation models
        seg_models = [r for r in self.results_log if r['task'] == 'segmentation']
        if seg_models:
            print("\n🎨 SEGMENTATION MODELS:")
            print(f"{'Model':<20} {'Box mAP50':<15} {'Box mAP50-95':<15} {'Mask mAP50':<15} {'Mask mAP50-95':<15}")
            print("-"*100)
            for r in seg_models:
                m = r['metrics']
                print(f"{r['model_name']:<20} {m.get('box_mAP50',0):<15.4f} "
                      f"{m.get('box_mAP50-95',0):<15.4f} {m.get('mask_mAP50',0):<15.4f} "
                      f"{m.get('mask_mAP50-95',0):<15.4f}")

        print("="*100)


def main():
    """Main execution with dual dataset support"""
    print("="*100)
    print("FIXED YOLO TRAINER - Detection + Segmentation")
    print("Uses proper dual datasets with correct label formats")
    print("="*100)

    # Dataset paths
    DETECTION_YAML = "unified_crack_dataset/data.yaml"  # Bbox labels
    SEGMENTATION_YAML = "unified_crack_dataset_seg/data.yaml"  # Polygon labels

    # Verify both datasets exist
    if not os.path.exists(DETECTION_YAML):
        print(f"\n❌ ERROR: Detection dataset not found: {DETECTION_YAML}")
        print("   Run order.py first to create datasets!")
        return

    if not os.path.exists(SEGMENTATION_YAML):
        print(f"\n❌ ERROR: Segmentation dataset not found: {SEGMENTATION_YAML}")
        print("   Run order.py first to create datasets!")
        return

    print(f"\n📁 Dataset Configuration:")
    print(f"   Detection (bbox): {DETECTION_YAML}")
    print(f"   Segmentation (polygon): {SEGMENTATION_YAML}")

    # Initialize trainer
    trainer = FixedUnifiedYOLOTrainer(
        detection_yaml=DETECTION_YAML,
        segmentation_yaml=SEGMENTATION_YAML,
        output_dir='yolo_experiments'
    )

    # Check if detection model already exists
    existing_det = trainer.find_latest_detection_model('yolo11s_det')

    if existing_det:
        print(f"\n💡 DETECTION MODEL ALREADY TRAINED!")
        print(f"   Found: {existing_det}")

        user_input = input("\n   Skip detection training and only train segmentation? (y/n): ").strip().lower()
        skip_detection = (user_input == 'y')
    else:
        skip_detection = False

    # Training configuration
    EPOCHS = 150
    IMAGE_SIZE = 192
    BATCH_SIZE = 16  # Reduced from 32 to prevent memory issues
    PATIENCE = 30

    print(f"\n⚙️ Training Configuration:")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Image size: {IMAGE_SIZE}")
    print(f"   Batch size: {BATCH_SIZE} (detection), {BATCH_SIZE//2} (segmentation)")
    print(f"   Patience: {PATIENCE}")

    if skip_detection:
        print(f"\n⏭️ Will skip detection training")

    # Train both models (or just segmentation)
    print("\n" + "="*100)
    print("STARTING TRAINING")
    print("="*100)

    trainer.train_both_models(
        det_model='yolo11s',
        seg_model='yolo11s-seg',
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        skip_detection=skip_detection,  # NEW: Skip detection if already trained
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
    )

    print("\n" + "="*100)
    print("✅ TRAINING COMPLETE!")
    print("="*100)

    print("\n📝 Next Steps:")
    print("   1. Check training_summary.json for results")
    print("   2. Best models saved in yolo_experiments/*/weights/best.pt")
    print("   3. Run model_comparison.py to compare all models")

    # Show final model paths
    import glob
    det_models = glob.glob("yolo_experiments/yolo11s_det_*/weights/best.pt")
    seg_models = glob.glob("yolo_experiments/yolo11s-seg_*/weights/best.pt")

    if det_models:
        print(f"\n📦 Detection Model: {sorted(det_models)[-1]}")
    if seg_models:
        print(f"📦 Segmentation Model: {sorted(seg_models)[-1]}")


if __name__ == "__main__":
    main()