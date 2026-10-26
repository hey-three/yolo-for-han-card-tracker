"""Run prediction on a chosen image with a trained checkpoint."""
import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="对单张图片进行扑克牌检测")
    parser.add_argument("--model", default="runs/detect/train/weights/best.pt", help="模型权重路径")
    parser.add_argument("--image", default="data/test_images/jj0.png", help="待预测图片")
    args = parser.parse_args()

    model = YOLO(str(project_path(args.model)))
    model.predict(
        source=str(project_path(args.image)),
        project=str(ROOT / "runs" / "detect"),
        name="predict_one",
        save=True,
        conf=0.5,
        show_labels=False,
        show_conf=False,
    )


if __name__ == "__main__":
    main()

