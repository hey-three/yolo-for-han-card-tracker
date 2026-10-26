"""Crop card boxes into per-class image folders for manual review."""
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "data" / "inference_images"
LABEL_DIR = ROOT / "data" / "class_crop_review" / "labels"
CLASS_FILE = ROOT / "data" / "class_crop_review" / "classes.txt"
OUTPUT_DIR = ROOT / "data" / "class_crop_review" / "crops"


def main() -> None:
    class_names = [line.strip() for line in CLASS_FILE.read_text(encoding="utf-8").splitlines()]
    for class_name in class_names:
        (OUTPUT_DIR / class_name).mkdir(parents=True, exist_ok=True)

    for image_path in sorted(IMAGE_DIR.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        label_path = LABEL_DIR / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"缺少标签：{image_path.name}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"无法读取图片：{image_path}")
            continue

        height, width = image.shape[:2]
        for index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            class_value, x_center, y_center, box_width, box_height = line.split()
            class_id = int(class_value)
            if not 0 <= class_id < len(class_names):
                print(f"标签类别编号越界：{label_path.name}: {class_id}")
                continue

            x_center, y_center = float(x_center), float(y_center)
            box_width, box_height = float(box_width), float(box_height)
            x1 = max(0, int((x_center - box_width / 2) * width))
            y1 = max(0, int((y_center - box_height / 2) * height))
            x2 = min(width, int((x_center + box_width / 2) * width))
            y2 = min(height, int((y_center + box_height / 2) * height))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]
            output_path = OUTPUT_DIR / class_names[class_id] / f"{image_path.stem}_{index}.jpg"
            if not cv2.imwrite(str(output_path), crop):
                print(f"保存失败：{output_path}")

    print(f"裁剪结果已保存到：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()

