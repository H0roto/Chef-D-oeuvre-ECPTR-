import argparse
from ultralytics import YOLO
import os
def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLO Training Script")
    parser.add_argument("--model", type=str, default="YOLO/yolo12s.pt", help="Model file")
    parser.add_argument("--data", type=str, default="YOLO/dataset_yolo_15dB/dataset.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=160)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=0.05)
    parser.add_argument("--flipud", type=float, default=0.5)
    parser.add_argument("--box", type=float, default=15.0)
    parser.add_argument("--cls", type=float, default=2.0)
    parser.add_argument("--project", type=str, default="YOLO/YOLO_Results")
    parser.add_argument("--name", type=str, default="YOLO/YOLO12s_Training_15dB_160")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    abs_model_path = os.path.abspath(args.model)
    abs_data_path = os.path.abspath(args.data)
    abs_project_path = os.path.abspath(args.project)
    print(f"Loading model from: {abs_model_path}")
    model = YOLO(args.model)
    results = model.train(
        data=abs_data_path,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        optimizer=args.optimizer,
        workers=args.workers,
        mosaic=args.mosaic,
        scale=args.scale,
        flipud=args.flipud,
        box=args.box,
        cls=args.cls,
        project=abs_project_path,
        name=args.name,
        exist_ok=True
    )

    print("Training completed!")