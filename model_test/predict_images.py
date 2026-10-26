"""Predict the sample images in this folder using a manually selected YOLO model."""
from pathlib import Path

from ultralytics import YOLO


TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent

# 修改为你想测试的模型权重路径，例如 runs/detect/train_416/weights/best.pt。
MODEL_PATH = Path(r"runs/detect/train_4163/weights/best.pt")

IMAGE_DIR = TEST_DIR / "images"
OUTPUT_DIR = TEST_DIR / "predictions"
IMAGE_SIZE = 416
CONFIDENCE = 0.25


def main() -> None:
    # 相对路径固定从项目根目录解析，不受 PyCharm 当前工作目录影响。
    model_path = MODEL_PATH if MODEL_PATH.is_absolute() else PROJECT_ROOT / MODEL_PATH
    model_path = model_path.resolve()

    if not model_path.is_file():
        raise FileNotFoundError(
            f"找不到模型权重：{model_path}\n请修改脚本顶部的 MODEL_PATH。"
        )
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"找不到待预测图片目录：{IMAGE_DIR}")

    image_files = [
        path for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]
    if not image_files:
        raise FileNotFoundError(f"图片目录为空：{IMAGE_DIR}")

    model = YOLO(str(model_path))
    model.predict(
        source=str(IMAGE_DIR),
        project=str(TEST_DIR),
        name=OUTPUT_DIR.name,
        exist_ok=True,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        save=True,
        show_labels=True,
        show_conf=True,
    )
    print(f"预测完成：{len(image_files)} 张图片")
    print(f"结果目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()

