"""Draw YOLO bounding boxes on images for manual label review."""
import argparse
import random
from pathlib import Path
from typing import List, Tuple

import cv2


ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_classes(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]


def draw_labels(image_path: Path, label_path: Path, output_path: Path, colors: List[Tuple[int, int, int]]) -> bool:
    image = cv2.imread(str(image_path))
    if image is None or not label_path.exists():
        return False

    image_height, image_width = image.shape[:2]
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            print(f"跳过格式错误的标签：{label_path.name}:{line_number}")
            continue

        class_id = int(parts[0])
        center_x, center_y, box_width, box_height = map(float, parts[1:])
        x1 = max(0, int((center_x - box_width / 2) * image_width))
        y1 = max(0, int((center_y - box_height / 2) * image_height))
        x2 = min(image_width, int((center_x + box_width / 2) * image_width))
        y2 = min(image_height, int((center_y + box_height / 2) * image_height))
        color = colors[class_id] if 0 <= class_id < len(colors) else (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(output_path), image)


def main() -> None:
    parser = argparse.ArgumentParser(description="把 YOLO 标签框绘制到图片上")
    parser.add_argument("--images", required=True, help="图片目录")
    parser.add_argument("--labels", required=True, help="对应的标签目录")
    parser.add_argument("--output", required=True, help="可视化结果目录")
    parser.add_argument("--classes", default="dataset/classes.txt", help="类别文件")
    args = parser.parse_args()

    image_dir = resolve_path(args.images)
    label_dir = resolve_path(args.labels)
    output_dir = resolve_path(args.output)
    class_file = resolve_path(args.classes)
    class_names = load_classes(class_file)
    rng = random.Random(0)
    colors = [
        (rng.randint(50, 255), rng.randint(50, 255), rng.randint(50, 255))
        for _ in class_names
    ]

    image_paths = [
        path for path in sorted(image_dir.iterdir())
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]
    succeeded = 0
    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        output_path = output_dir / image_path.name
        if draw_labels(image_path, label_path, output_path, colors):
            succeeded += 1
        else:
            print(f"跳过无法读取或缺少标签的图片：{image_path.name}")

    print(f"可视化完成：{succeeded}/{len(image_paths)} 张，结果位于 {output_dir}")


if __name__ == "__main__":
    main()


