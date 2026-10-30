"""预测文件夹里的图片，将带检测框的图片保存到新的输出文件夹。

使用前修改下面的 MODEL_DIR 和 IMAGE_DIR，然后运行：
    python model_test/predict_folder.py

模型目录应包含 best.pt；也可以直接把 MODEL_DIR 设置为 .pt 文件路径。
如果权重旁有 model_config.yaml，会自动读取其中的输入宽高；否则使用下方
IMAGE_WIDTH 和 IMAGE_HEIGHT。预测不需要标签或 classes.txt。
"""

from datetime import datetime
from pathlib import Path

import yaml
from PIL import Image
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 模型所在文件夹（包含 best.pt），也可以写成具体的 .pt 文件路径。
MODEL_DIR = PROJECT_ROOT / "runs" / "my_weights" / "yolov11s960"

# 存放原始图片的文件夹；只处理该文件夹下的图片。
IMAGE_DIR = PROJECT_ROOT / "model_test" / "55"

# 每次运行会在此目录下新建一个文件夹，不覆盖以前的结果。
OUTPUT_DIR = PROJECT_ROOT / "model_test" / "predictions" / "folder"

# 没有 model_config.yaml 时使用的预测画布宽高，建议均为 32 的倍数。
IMAGE_WIDTH = 960
IMAGE_HEIGHT = 608
CONFIDENCE = 0.6  # 只显示置信度不低于此值的检测结果，可设为 0 到 1。
DEVICE = None  # None 由 Ultralytics 自动选择设备；也可设为 0 或 "cpu"。
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def project_path(path: Path) -> Path:
    """相对路径以项目根目录为基准，不依赖运行时的工作目录。"""
    return path if path.is_absolute() else PROJECT_ROOT / path


def input_size(model_path: Path) -> tuple[int, int]:
    """优先读取权重旁的模型配置，返回（高，宽）。"""
    config_path = model_path.parent / "model_config.yaml"
    width, height = IMAGE_WIDTH, IMAGE_HEIGHT
    if config_path.is_file():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
        if not isinstance(config, dict) or not isinstance(config.get("input"), dict):
            raise ValueError(f"模型配置缺少 input 部分：{config_path}")
        width = config["input"].get("width")
        height = config["input"].get("height")
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in (width, height)):
        raise ValueError("预测宽高必须为大于 0 的整数，请检查模型配置或脚本顶部设置。")
    return height, width


def main() -> None:
    model_path = project_path(MODEL_DIR)
    if model_path.is_dir():
        model_path = model_path / "best.pt"
    if not model_path.is_file() or model_path.suffix.lower() != ".pt":
        raise FileNotFoundError(f"找不到 .pt 模型：{model_path}\n请检查脚本顶部的 MODEL_DIR。")

    image_dir = project_path(IMAGE_DIR)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"找不到图片文件夹：{image_dir}\n请检查脚本顶部的 IMAGE_DIR。")
    images = sorted((p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                    key=lambda p: p.name.casefold())
    if not images:
        raise FileNotFoundError(f"图片文件夹中没有支持的图片：{image_dir}")
    if not 0 <= CONFIDENCE <= 1:
        raise ValueError("CONFIDENCE 必须在 0 到 1 之间。")

    height, width = input_size(model_path)
    model = YOLO(str(model_path))
    output_dir = project_path(OUTPUT_DIR) / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"模型：{model_path}\n图片：{image_dir}\n预测画布：宽 {width} × 高 {height}\n输出：{output_dir}")

    for index, image_path in enumerate(images, 1):
        result = model.predict(source=str(image_path), imgsz=(height, width), rect=False,
                               conf=CONFIDENCE, device=DEVICE, verbose=False)[0]
        # result.plot() 使用原图尺寸绘制检测框，保存时转换为 RGB。
        Image.fromarray(result.plot()[:, :, ::-1].copy()).save(output_dir / image_path.name)
        print(f"已完成 {index}/{len(images)}：{image_path.name}")

    print(f"预测完成，结果保存在：{output_dir}")


if __name__ == "__main__":
    main()
