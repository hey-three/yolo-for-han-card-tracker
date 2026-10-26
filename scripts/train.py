"""Train the card detector on the project dataset."""
import os
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "weights" / "pretrained" / "yolo11n.pt"
DATA_CONFIG = ROOT / "data.yaml"
OUTPUT_DIR = ROOT / "runs" / "detect"


def main() -> None:
    # Resolve data.yaml relative paths from the project root, not the IDE working directory.
    os.chdir(ROOT)
    model = YOLO(str(MODEL_PATH))
    model.train(
        data=str(DATA_CONFIG),
        project=str(OUTPUT_DIR),
        name="train",
        epochs=100,
        rect=True,
        imgsz=960,
        batch=16,
        device=0,
        workers=0,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=50,
        patience=30,
        save=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()



