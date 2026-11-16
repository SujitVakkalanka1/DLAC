"""
VisionPort — Model Training & Fine-Tuning Pipeline
Trains or fine-tunes Ultralytics YOLO models on traffic detection datasets.
"""

import sys
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

def create_sample_dataset_yaml(data_yaml_path):
    """Creates a sample dataset configuration YAML if not present."""
    sample_content = f"""# VisionPort Custom Traffic Dataset Configuration
path: {str(BASE_DIR / 'data' / 'traffic_dataset').replace('\\', '/')}
train: images/train
val: images/val

# Classes (COCO aligned traffic classes)
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: traffic light
"""
    data_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(sample_content)
    print(f"[*] Generated dataset config template at: {data_yaml_path}")

def train_model(
    model_name="yolo11s.pt",
    data_yaml="data/traffic_dataset.yaml",
    epochs=10,
    imgsz=640,
    batch_size=8,
    device=None,
    update_settings=False
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print(" VISIONPORT: YOLO Model Training & Fine-Tuning")
    print("=" * 72)
    print(f"Base Model Weights : {model_name}")
    print(f"Dataset Config     : {data_yaml}")
    print(f"Epochs             : {epochs}")
    print(f"Image Size         : {imgsz}")
    print(f"Batch Size         : {batch_size}")
    print(f"Compute Device     : {device.upper()}")
    print("=" * 72)

    yaml_file = BASE_DIR / data_yaml
    if not yaml_file.exists():
        print(f"[!] Warning: Dataset file '{yaml_file}' not found.")
        create_sample_dataset_yaml(yaml_file)
        print("[!] Please populate images and labels in 'data/traffic_dataset/' or specify an existing dataset YAML.")
        print("[*] To train on a standard dataset (e.g. COCO8 for testing), use: --data coco8.yaml")
        return None

    # Load base model
    print(f"[*] Initializing model: {model_name}...")
    model = YOLO(model_name)

    # Train
    print("[*] Commencing training...")
    results = model.train(
        data=str(yaml_file),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        workers=0 if sys.platform.startswith("win") else 4,
        project=str(BASE_DIR / "runs" / "train"),
        name="visionport_traffic"
    )

    best_weights = BASE_DIR / "runs" / "train" / "visionport_traffic" / "weights" / "best.pt"
    print("=" * 72)
    print("[*] Training Finished!")
    if best_weights.exists():
        print(f"[+] Optimal weights saved at: {best_weights}")
        if update_settings:
            import json
            cfg_path = BASE_DIR / "config" / "settings.json"
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["model"] = str(best_weights)
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            print(f"[+] Updated {cfg_path} to use newly trained model weights!")
    print("=" * 72)
    return results

def main():
    parser = argparse.ArgumentParser(description="Train / Fine-tune VisionPort YOLO model.")
    parser.add_argument("--model", default="yolo11s.pt", help="Base model weights (e.g. yolo11s.pt, yolo11m.pt)")
    parser.add_argument("--data", default="coco8.yaml", help="Path to dataset YAML (default: coco8.yaml for quick test)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--device", default=None, help="Device ('cpu', 'cuda', '0', etc.)")
    parser.add_argument("--update-config", action="store_true", help="Set the newly trained weights in config/settings.json")
    args = parser.parse_args()

    train_model(
        model_name=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        update_settings=args.update_config
    )

if __name__ == "__main__":
    main()
