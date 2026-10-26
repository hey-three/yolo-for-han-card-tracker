"""Generate draft YOLO labels for an image or image directory."""
import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="批量生成预标注")
    parser.add_argument("--model", default="runs/detect/train/weights/best.pt", help="模型权重路径")
    parser.add_argument("--source", default="data/inference_images", help="图片文件或目录")
    args = parser.parse_args()

    model = YOLO(str(project_path(args.model)))
    model.predict(
        source=str(project_path(args.source)),
        project=str(ROOT / "runs" / "detect"),
        name="prelabel",
        save=True,
        save_txt=True,
        conf=0.6,
        show_labels=False,
    )


if __name__ == "__main__":
    main()

