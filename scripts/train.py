"""使用指定的矩形输入画布训练 YOLO 检测模型。"""
import os
from functools import partial
from pathlib import Path

import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer


ROOT = Path(__file__).resolve().parents[1]

# True：从 MODEL_PATH 指定的 .pt 权重开始新一轮训练。
# False：不加载任何权重，从 MODEL_CONFIG 指定的模型结构随机初始化。
LOAD_WEIGHTS = True
MODEL_PATH = ROOT /"runs"/ "detect" /"train4"/ "weights" / "best.pt"
# MODEL_PATH = ROOT /"runs"/ "my_weights" / "weights3" / "best.pt"
MODEL_CONFIG = "yolo11n.yaml"

DATA_CONFIG = ROOT / "data.yaml"
OUTPUT_DIR = ROOT / "runs" / "detect"

IMAGE_WIDTH = 960  # 实际训练画布的宽，建议设为 32 的倍数。
IMAGE_HEIGHT = 608  # 实际训练画布的高，建议设为 32 的倍数。

# 牌面任务使用轻量颜色增强；0 表示关闭对应增强。
AUGMENTATION_ARGS = {
    # 颜色变化
    "hsv_h": 0.015,  # 色调随机偏移：约为色环的正负 1.5%。
    "hsv_s": 0.3,    # 饱和度随机变化：让颜色更鲜艳或更灰，幅度约为正负 30%。
    "hsv_v": 0.2,    # 亮度随机变化：让图片更亮或更暗，幅度约为正负 20%。

    # 几何变化
    "degrees": 0.0,      # 旋转的最大角度；0 表示不旋转。
    "translate": 0.0,    # 平移的最大幅度，占图片宽或高的比例；0 表示不平移。
    "scale": 0.0,        # 随机缩放的幅度；0 表示关闭随机缩放。
    "shear": 0.0,        # 随机斜切的最大角度；0 表示不斜切。
    "perspective": 0.0,  # 透视变形幅度；0 表示不变形。
    "fliplr": 0.0,       # 左右镜像的概率；0 表示不翻转。
    "flipud": 0.0,       # 上下镜像的概率；0 表示不翻转。

    # 多图增强
    "mosaic": 0.0,      # 将多张图片拼接成一张的概率；0 表示关闭。
    "mixup": 0.0,       # 将两张图片混合的概率；0 表示关闭。
    "cutmix": 0.0,      # 将另一张图片的区域剪贴进来的概率；0 表示关闭。
    "close_mosaic": 0,  # 最后多少轮关闭 Mosaic；当前 Mosaic 已关闭，此值不起作用。

    # 颜色通道交换
    "bgr": 0.0,  # 交换图像颜色通道的概率；0 表示不交换。
}


class FixedResolutionDetectionTrainer(DetectionTrainer):
    """将每个矩形 batch 固定为设置的高和宽。"""

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        dataset = super().build_dataset(img_path, mode, batch)
        if not dataset.rect:
            raise RuntimeError("固定矩形输入画布需要 rect=True。")

        # 固定实际输入尺寸；宽高均为 32 的倍数，可与 YOLO11 特征图步长对齐。
        dataset.batch_shapes[:] = (IMAGE_HEIGHT, IMAGE_WIDTH)
        return dataset


def save_model_config(
    run_dir: Path, *, model_name: str, initial_weights: Path | None, require_weights: bool = True
) -> Path:
    """在本次训练的权重目录中记录预测输入配置。"""
    weights_dir = run_dir / "weights"
    weight_names = [name for name in ("best.pt", "last.pt") if (weights_dir / name).is_file()]
    if not weight_names and require_weights:
        raise FileNotFoundError(f"训练结束后未找到模型权重：{weights_dir}")
    if not weight_names:
        # 训练刚开始时尚未保存权重，先记录之后默认生成的文件名。
        weight_names = ["best.pt", "last.pt"]

    config = {
        "model": model_name,
        "initial_weights": str(initial_weights) if initial_weights is not None else None,
        "weights": weight_names,
        "input": {
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
            "resize": "letterbox",
            # 预测时关闭最小矩形补边，保证使用上面的固定画布尺寸。
            "rect": False,
        },
    }
    config_path = weights_dir / "model_config.yaml"
    with config_path.open("w", encoding="utf-8") as file:
        file.write("# 训练开始时权重文件可能尚未生成；训练结束后会更新权重列表。\n")
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    return config_path


def save_config_at_start(
    trainer: DetectionTrainer, *, model_name: str, initial_weights: Path | None
) -> None:
    """训练准备阶段先保存尺寸，避免中途停止时丢失配置。"""
    config_path = save_model_config(
        Path(trainer.save_dir), model_name=model_name, initial_weights=initial_weights, require_weights=False
    )
    print(f"模型配置已保存：{config_path}")


def main() -> None:
    # 从项目根目录解析 data.yaml，避免依赖 IDE 的当前工作目录。
    os.chdir(ROOT)
    initial_weights = None
    if LOAD_WEIGHTS:
        initial_weights = MODEL_PATH if MODEL_PATH.is_absolute() else ROOT / MODEL_PATH
        initial_weights = initial_weights.resolve()
        if not initial_weights.is_file():
            raise FileNotFoundError(f"找不到要加载的权重：{initial_weights}；请修改脚本顶部的 MODEL_PATH。")
        if initial_weights.suffix.lower() != ".pt":
            raise ValueError(f"训练权重必须是 .pt 文件：{initial_weights}")
        model = YOLO(str(initial_weights))
    else:
        model = YOLO(MODEL_CONFIG)

    model_name = Path(model.model.yaml.get("yaml_file") or MODEL_CONFIG).stem
    model.add_callback(
        "on_pretrain_routine_start",
        partial(save_config_at_start, model_name=model_name, initial_weights=initial_weights),
    )
    model.train(
        # 使用自定义训练器，让每个批次使用上面设置的固定宽、高。
        trainer=FixedResolutionDetectionTrainer,
        # 数据集配置文件，包含训练集、验证集路径和类别名称。
        data=str(DATA_CONFIG),
        # 所有训练结果的父目录，例如 runs/detect。
        project=str(OUTPUT_DIR),
        # 本次运行的目录名；同名目录已存在时，通常会自动编号。
        name="train",
        # 最多训练 200 轮；触发提前停止时可能少于 200 轮。
        epochs=100,
        # 训练接口要求 imgsz 为单值；自定义训练器将实际 batch 固定为
        # (高, 宽) = (IMAGE_HEIGHT, IMAGE_WIDTH)。
        imgsz=IMAGE_WIDTH,
        # 每批处理 16 张图片；显存不足时可减小。
        batch=16,
        # 使用编号为 0 的 GPU；改为 "cpu" 可使用 CPU。
        device=0,
        # 数据加载子进程数；Windows 环境保持 0。
        workers=0,
        # 启用矩形训练；当前自定义训练器需要此选项为 True。
        rect=True,
        # 跟随顶部开关：True 加载指定权重，False 从模型结构随机初始化。
        pretrained=LOAD_WEIGHTS,
        # False 表示开始一次新的训练，不恢复旧运行的轮数和优化器状态。
        resume=False,
        # 设定初始学习率；默认 optimizer=auto 时会自动选择，可能忽略此值。
        lr0=0.01,
        # 学习率下降后的比例：0.01 表示训练末尾约为初始学习率的 1%。
        # 默认使用线性下降；实际初始学习率以训练日志中的优化器信息为准。
        lrf=0.01,
        # 优化器动量目标，用于平滑参数更新。自动优化器会选择初始动量；
        # 对带 momentum 的参数组，预热期间仍会向这里的 0.937 调整。
        momentum=0.937,
        # 权重衰减，用于抑制部分权重过大；框架会按批大小等条件调整实际值。
        weight_decay=0.0005,
        # 预热约 3 轮，逐步调整学习率和动量；当前版本至少预热 100 个批次。
        warmup_epochs=3,
        # 预热起始动量；对带 momentum 的参数组逐步调整到 momentum 的设定值。
        warmup_momentum=0.8,
        # 偏置参数的预热起始学习率；默认 optimizer=auto 会把此值改为 0。
        warmup_bias_lr=0.1,
        # 验证指标连续 30 轮没有改善时提前停止，保留此前最好的权重。
        patience=30,
        # 保存训练权重，例如最后一轮的 last.pt 和验证表现最好的 best.pt。
        save=True,
        # 保存训练曲线、验证结果和混淆矩阵等图表。
        plots=True,
        # 在终端显示较详细的训练信息。
        verbose=True,
        # 应用上面 AUGMENTATION_ARGS 中设置的颜色和图像增强参数。
        **AUGMENTATION_ARGS,
    )
    config_path = save_model_config(
        Path(model.trainer.save_dir), model_name=model_name, initial_weights=initial_weights
    )
    print(f"模型配置已保存：{config_path}")


if __name__ == "__main__":
    main()
