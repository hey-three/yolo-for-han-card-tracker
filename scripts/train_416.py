"""Train YOLO11n at 416px with the user-selected augmentations."""
import os
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "weights" / "pretrained" / "yolo11n.pt"
DATA_CONFIG = ROOT / "data.yaml"
OUTPUT_DIR = ROOT / "runs" / "detect"

# 已确认的图像增强设置。0 表示关闭对应增强。
AUGMENTATION_ARGS = {
    # 颜色变化
    "hsv_h": 0.015,  # 轻微色调偏移
    "hsv_s": 0.3,    # 适度饱和度变化
    "hsv_v": 0.2,    # 适度亮度变化

    # 几何变化
    "degrees": 0.0,      # 旋转
    "translate": 0.0,    # 平移
    "scale": 0.0,        # 缩放
    "shear": 0.0,        # 倾斜
    "perspective": 0.0,  # 透视变形
    "fliplr": 0.0,       # 左右镜像概率
    "flipud": 0.0,       # 上下镜像概率

    # 多图增强
    "mosaic": 0.0,       # 四图拼接概率
    "mixup": 0.0,        # 两图混合概率
    "cutmix": 0.0,       # 区块拼贴概率
    "close_mosaic": 0,   # mosaic 已关闭，因此无需提前关闭

    # 颜色通道交换
    "bgr": 0.0,
}


def main() -> None:
    # Resolve data.yaml relative paths from the project root, not the IDE working directory.
    os.chdir(ROOT)
    model = YOLO(str(MODEL_PATH))
    model.train(
        data=str(DATA_CONFIG),
        project=str(OUTPUT_DIR),
        name="train_416",
        epochs=100,
        imgsz=416,
        batch=32,
        device=0,
        workers=0,
        rect=True,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        patience=30,
        save=True,
        plots=True,  # 保存 loss、Precision、Recall、mAP 等训练曲线和验证图
        verbose=True,
        **AUGMENTATION_ARGS,
    )


if __name__ == "__main__":
    main()



