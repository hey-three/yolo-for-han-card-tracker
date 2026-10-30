"""批量预测图片，对照 YOLO 标签生成错误明细、汇总和可视化图片。"""
import csv
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from ultralytics.utils.checks import check_imgsz


TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent

# 修改为你想测试的模型权重路径，相对路径从项目根目录开始。
# MODEL_PATH = Path(r"runs/my_weights/weights1/best.pt")
MODEL_PATH = Path(r"runs/detect/train15/weights/best.pt")

IMAGE_DIR = TEST_DIR / "images"
LABEL_DIR = TEST_DIR / "labels"
CLASSES_FILE = TEST_DIR / "classes.txt"
OUTPUT_DIR = TEST_DIR / "predictions"

# 预测画布的宽和高，可按需修改；建议均为 32 的倍数。
# 非 32 倍数的尺寸会由 Ultralytics 向上调整。
IMAGE_WIDTH =960
IMAGE_HEIGHT = 608
CONFIDENCE = 0.6
MATCH_IOU = 0.5  # 达到这个交并比才认为预测框和标注框是同一个目标。
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class Box:
    """标注框或预测框，坐标为原图像素。"""

    number: int
    name: str
    xyxy: tuple[float, float, float, float]
    confidence: float | None = None


def load_classes(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到类别文件：{path}")
    names = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    if not names or any(not name for name in names):
        raise ValueError(f"类别文件为空或包含空行：{path}")
    if len(names) != len(set(names)):
        raise ValueError(f"类别文件中有重复类别名称：{path}")
    return names


def read_labels(path: Path, classes: list[str]) -> list[tuple[int, float, float, float, float]]:
    """读取 YOLO 检测格式；空文件表示图片中没有标注目标。"""
    annotations = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            fields = line.split()
            if len(fields) != 5:
                raise ValueError("每行应为：类别 ID 中心点 x 中心点 y 宽 高")
            class_id = int(fields[0])
            x, y, width, height = map(float, fields[1:])
            if not 0 <= class_id < len(classes):
                raise ValueError(f"类别 ID {class_id} 超出 classes.txt 的范围")
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y, width, height)):
                raise ValueError("坐标必须是 0 到 1 之间的有限数值")
            if width == 0 or height == 0:
                raise ValueError("标注框的宽和高必须大于 0")
        except ValueError as exc:
            raise ValueError(f"标签格式错误：{path}，第 {line_number} 行：{exc}") from exc
        annotations.append((class_id, x, y, width, height))
    return annotations


def annotation_boxes(annotations, classes: list[str], width: int, height: int) -> list[Box]:
    boxes = []
    for number, (class_id, x, y, w, h) in enumerate(annotations, 1):
        xyxy = (
            max(0.0, (x - w / 2) * width),
            max(0.0, (y - h / 2) * height),
            min(float(width), (x + w / 2) * width),
            min(float(height), (y + h / 2) * height),
        )
        boxes.append(Box(number, classes[class_id], xyxy))
    return boxes


def box_iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def compare_boxes(truth: list[Box], predictions: list[Box], threshold: float):
    """先按位置做一对一匹配，再区分分类正确与分类错误。"""
    if not truth or not predictions:
        return [], [], truth[:], predictions[:]
    overlaps = np.array([[box_iou(a, b) for b in predictions] for a in truth])
    valid = overlaps >= threshold
    # 优先匹配更多目标，再在匹配数量相同时选择总 IoU 更高的组合。
    scores = np.where(valid, min(len(truth), len(predictions)) + 1 + overlaps, 0.0)
    rows, cols = linear_sum_assignment(-scores)
    paired = [(int(i), int(j)) for i, j in zip(rows, cols) if valid[i, j]]
    matched_truth = {i for i, _ in paired}
    matched_predictions = {j for _, j in paired}
    correct, wrong = [], []
    for i, j in paired:
        item = (truth[i], predictions[j], float(overlaps[i, j]))
        (correct if truth[i].name == predictions[j].name else wrong).append(item)
    missed = [box for i, box in enumerate(truth) if i not in matched_truth]
    extra = [box for j, box in enumerate(predictions) if j not in matched_predictions]
    return correct, wrong, missed, extra


def create_run_dir() -> Path:
    """每次运行建立新目录，避免混入或覆盖之前的预测结果。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    for number in range(1000):
        path = OUTPUT_DIR / (name if number == 0 else f"{name}_{number}")
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError("无法创建新的预测输出目录。")


def draw_box(draw: ImageDraw.ImageDraw, box: Box, text: str, color: tuple[int, int, int], font,
             *, line_width: int = 3, text_inside: bool = False) -> None:
    x1, y1, x2, y2 = (round(value) for value in box.xyxy)
    draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
    text_height = draw.textbbox((0, 0), text, font=font)[3]
    left = max(0, x1)
    top = max(0, y1 + 2 if text_inside else y1 - text_height - 2)
    bounds = draw.textbbox((left, top), text, font=font)
    draw.rectangle(bounds, fill=color)
    draw.text((left, top), text, fill=(0, 0, 0), font=font)


def save_error_image(result, path: Path, wrong, missed, extra) -> None:
    """绿色为真实标注，红色为错误预测，橙色为漏检，蓝色为多检。"""
    image = Image.fromarray(result.orig_img[:, :, ::-1].copy())
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", max(12, round(max(image.size) / 100)))
    except OSError:
        font = ImageFont.load_default()
    for truth, prediction, _ in wrong:
        draw_box(draw, truth, f"GT {truth.number}: {truth.name}", (0, 255, 0), font, line_width=5)
        draw_box(draw, prediction, f"P {prediction.number}: {prediction.name} {prediction.confidence:.2f}",
                 (255, 0, 0), font, line_width=2, text_inside=True)
    for truth in missed:
        draw_box(draw, truth, f"MISS GT {truth.number}: {truth.name}", (255, 165, 0), font)
    for prediction in extra:
        draw_box(draw, prediction, f"EXTRA P {prediction.number}: {prediction.name} {prediction.confidence:.2f}", (0, 180, 255), font)
    image.save(path)


def format_box(box: Box | None) -> str:
    return "" if box is None else ", ".join(f"{value:.1f}" for value in box.xyxy)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    # 带 BOM 的 UTF-8 方便 Windows Excel 正确显示中文。
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    model_path = MODEL_PATH if MODEL_PATH.is_absolute() else PROJECT_ROOT / MODEL_PATH
    model_path = model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到模型权重：{model_path}\n请修改脚本顶部的 MODEL_PATH。")
    if not IMAGE_DIR.is_dir() or not LABEL_DIR.is_dir():
        raise FileNotFoundError(f"找不到图片或标签文件夹：{IMAGE_DIR}、{LABEL_DIR}")
    if not 0 <= CONFIDENCE <= 1 or not 0 < MATCH_IOU <= 1:
        raise ValueError("CONFIDENCE 必须在 0 到 1 之间，MATCH_IOU 必须大于 0 且不超过 1。")
    if (not isinstance(IMAGE_WIDTH, int) or isinstance(IMAGE_WIDTH, bool) or IMAGE_WIDTH <= 0
            or not isinstance(IMAGE_HEIGHT, int) or isinstance(IMAGE_HEIGHT, bool) or IMAGE_HEIGHT <= 0):
        raise ValueError("IMAGE_WIDTH 和 IMAGE_HEIGHT 必须是大于 0 的整数。")

    classes = load_classes(CLASSES_FILE)
    images = sorted(
        (p for p in IMAGE_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.name.casefold(),
    )
    if not images:
        raise FileNotFoundError(f"图片目录为空：{IMAGE_DIR}")
    labels = {}
    for path in LABEL_DIR.iterdir():
        if path.is_file() and path.suffix.lower() == ".txt":
            key = path.stem.casefold()
            if key in labels:
                raise ValueError(f"标签文件名重复：{path.name}")
            labels[key] = path
    image_keys = [path.stem.casefold() for path in images]
    if len(image_keys) != len(set(image_keys)):
        raise ValueError("图片文件名重复，无法一一匹配标签。")
    missing = [image.name for image in images if image.stem.casefold() not in labels]
    if missing:
        raise FileNotFoundError(f"有 {len(missing)} 张图片缺少同名标签，前几张：{', '.join(missing[:10])}")
    annotations = {key: read_labels(labels[key], classes) for key in image_keys}
    unused_label_count = len(set(labels) - set(image_keys))

    model = YOLO(str(model_path))
    model_names = model.names
    model_class_names = list(model_names.values()) if isinstance(model_names, dict) else list(model_names)
    if len(model_class_names) != len(set(model_class_names)) or set(model_class_names) != set(classes):
        raise ValueError(
            f"模型类别与 classes.txt 不一致（区分大小写）。\n模型：{model_class_names}\n类别文件：{classes}"
        )
    actual_height, actual_width = check_imgsz(
        (IMAGE_HEIGHT, IMAGE_WIDTH), stride=int(model.model.stride.max()), min_dim=2
    )

    run_dir = create_run_dir()
    annotated_dir, error_dir = run_dir / "images", run_dir / "errors"
    annotated_dir.mkdir()
    error_dir.mkdir()
    totals = Counter()
    per_class = {name: Counter() for name in classes}
    image_rows, error_rows = [], []
    seen = set()
    # 每次处理一张，避免把整个图片列表作为一个大 batch 导致显存不足。
    results = (
        result for image in images
        for result in model.predict(source=str(image), imgsz=(actual_height, actual_width), rect=False,
                                    conf=CONFIDENCE, save=False, stream=True, verbose=False)
    )
    for result in results:
        image_path = Path(result.path)
        key = image_path.stem.casefold()
        if key not in annotations or key in seen:
            raise RuntimeError(f"预测返回了无法匹配或重复的图片：{image_path}")
        seen.add(key)
        height, width = result.orig_shape
        truth = annotation_boxes(annotations[key], classes, width, height)
        predictions = []
        if result.boxes is not None:
            values = zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist())
            for number, (coords, class_id, confidence) in enumerate(values, 1):
                predictions.append(Box(number, model_names[int(class_id)], tuple(coords), float(confidence)))
        correct, wrong, missed, extra = compare_boxes(truth, predictions, MATCH_IOU)

        totals.update({"images": 1, "truth": len(truth), "predictions": len(predictions),
                       "correct": len(correct), "wrong": len(wrong), "missed": len(missed), "extra": len(extra)})
        if wrong or missed or extra:
            totals["images_with_errors"] += 1
        for box in truth:
            per_class[box.name]["truth"] += 1
        for box in predictions:
            per_class[box.name]["predictions"] += 1
        for box, _, _ in correct:
            per_class[box.name]["correct"] += 1
        for box, prediction, _ in wrong:
            per_class[box.name]["wrong"] += 1
            per_class[prediction.name]["wrong_as_prediction"] += 1
        for box in missed:
            per_class[box.name]["missed"] += 1
        for box in extra:
            per_class[box.name]["extra"] += 1

        image_rows.append({"图片": image_path.name, "标注数": len(truth), "预测数": len(predictions),
                           "正确": len(correct), "分类错误": len(wrong), "漏检": len(missed), "多检": len(extra)})

        def record_error(kind: str, gt: Box | None, pred: Box | None, overlap: float | None = None) -> None:
            error_rows.append({"图片": image_path.name, "问题": kind,
                               "标注序号": gt.number if gt else "", "真实类别": gt.name if gt else "",
                               "标注框像素坐标": format_box(gt), "预测序号": pred.number if pred else "",
                               "预测类别": pred.name if pred else "",
                               "置信度": f"{pred.confidence:.4f}" if pred else "",
                               "预测框像素坐标": format_box(pred), "IoU": f"{overlap:.4f}" if overlap is not None else ""})

        for gt, pred, overlap in wrong:
            record_error("分类错误", gt, pred, overlap)
        for gt in missed:
            record_error("漏检", gt, None)
        for pred in extra:
            record_error("多检", None, pred)

        Image.fromarray(result.plot()[:, :, ::-1].copy()).save(annotated_dir / image_path.name)
        if wrong or missed or extra:
            save_error_image(result, error_dir / image_path.name, wrong, missed, extra)
        if totals["images"] % 20 == 0 or totals["images"] == len(images):
            print(f"已检查 {totals['images']} / {len(images)} 张图片")

    if len(seen) != len(images):
        raise RuntimeError(f"预测没有处理完全部图片：已处理 {len(seen)}，预期 {len(images)}。")

    precision = totals["correct"] / totals["predictions"] if totals["predictions"] else None
    recall = totals["correct"] / totals["truth"] if totals["truth"] else None

    def percentage(value: float | None) -> str:
        return f"{value:.2%}" if value is not None else "无可计算目标"

    image_fields = ["图片", "标注数", "预测数", "正确", "分类错误", "漏检", "多检"]
    error_fields = ["图片", "问题", "标注序号", "真实类别", "标注框像素坐标", "预测序号", "预测类别", "置信度", "预测框像素坐标", "IoU"]
    class_fields = ["类别", "标注数", "预测数", "正确", "被错判为其他类别", "其他类别错判为此类别", "漏检", "多检", "查准率", "查全率"]
    class_rows = []
    for name in classes:
        count = per_class[name]
        class_rows.append({"类别": name, "标注数": count["truth"], "预测数": count["predictions"],
                           "正确": count["correct"], "被错判为其他类别": count["wrong"],
                           "其他类别错判为此类别": count["wrong_as_prediction"], "漏检": count["missed"],
                           "多检": count["extra"],
                           "查准率": percentage(count["correct"] / count["predictions"] if count["predictions"] else None),
                           "查全率": percentage(count["correct"] / count["truth"] if count["truth"] else None)})
    write_csv(run_dir / "image_summary.csv", image_fields, image_rows)
    write_csv(run_dir / "errors.csv", error_fields, error_rows)
    write_csv(run_dir / "class_summary.csv", class_fields, class_rows)

    lines = ["模型预测检查报告", f"模型：{model_path}", f"图片目录：{IMAGE_DIR}", f"标签目录：{LABEL_DIR}",
             f"类别文件：{CLASSES_FILE}", f"设置画布：宽 {IMAGE_WIDTH} × 高 {IMAGE_HEIGHT}",
             f"实际预测画布：宽 {actual_width} × 高 {actual_height}",
             f"置信度阈值：{CONFIDENCE}，匹配 IoU 阈值：{MATCH_IOU}", "",
             f"图片：{totals['images']} 张，其中有问题：{totals['images_with_errors']} 张",
             f"标注框：{totals['truth']} 个；预测框：{totals['predictions']} 个",
             f"正确：{totals['correct']}；分类错误：{totals['wrong']}；漏检：{totals['missed']}；多检：{totals['extra']}",
             f"查准率（正确匹配 / 预测框）：{percentage(precision)}",
             f"查全率（正确匹配 / 标注框）：{percentage(recall)}", "",
             "说明：分类错误的预测不计为正确；低于置信度阈值的预测不会参与比较。",
             "这些指标按本脚本的位置匹配规则计算，不是 mAP；错误仍需对照图片和标签人工检查。",
             "错误图例：绿色为分类错误对应的真实框，红色为错误预测；橙色为漏检，蓝色为多检。",
             "框的坐标顺序为 x1, y1, x2, y2，单位是原图像素。", ""]
    if unused_label_count:
        lines.append(f"未使用的标签文件：{unused_label_count} 个（没有同名图片）。")
    lines.append("逐张错误明细：")
    if not error_rows:
        lines.append("未发现错误。")
    for row in error_rows:
        if row["问题"] == "分类错误":
            detail = f"真实 {row['真实类别']}（标注 #{row['标注序号']}）→ 预测 {row['预测类别']}，置信度 {row['置信度']}，IoU {row['IoU']}"
        elif row["问题"] == "漏检":
            detail = f"真实 {row['真实类别']}（标注 #{row['标注序号']}），位置 {row['标注框像素坐标']}"
        else:
            detail = f"预测 {row['预测类别']}（预测 #{row['预测序号']}），置信度 {row['置信度']}，位置 {row['预测框像素坐标']}"
        lines.append(f"{row['图片']} | {row['问题']} | {detail}")
    (run_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(f"预测完成：{totals['images']} 张图片，有问题 {totals['images_with_errors']} 张。")
    print(f"报告及图片目录：{run_dir}")


if __name__ == "__main__":
    main()

