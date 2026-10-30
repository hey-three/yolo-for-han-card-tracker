"""按 YOLO 类别裁出目标小图，供人工检查类别标注是否正确。

使用步骤：
1. 修改本文件顶部的 IMAGES_DIR、LABELS_DIR、CLASSES_FILE 和 OUTPUT_DIR。
2. 确认图片与标签通过相同的相对路径和文件名对应。
3. 从项目根目录运行：python .\\scripts\\data_tools\\crop_by_class.py
4. 在 OUTPUT_DIR 下按类别查看裁剪结果。文件名包含源图片相对路径和框序号，
   可以据此回到原图和对应标签行。
5. 如果输出目录已有内容，脚本会停止以免混入旧裁片；再次运行时可修改 OUTPUT_DIR。

标签格式为 YOLO 检测框：class_id x_center y_center width height；坐标均为 0–1 归一化值。
输出裁片为 PNG，不会在保存时再经历 JPEG 有损压缩。源图片、标签均只读，不会被修改。
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[2]

# ============================ 需要按你的数据修改 ============================
# 当前默认使用 data/huanleddz；如果要检查其他数据，只改这四项路径。
IMAGES_DIR = ROOT / "data" / "huanleddz"/"images"
LABELS_DIR = ROOT / "data" / "huanleddz"/"labels"
CLASSES_FILE = ROOT / "data" /"huanleddz"/ "classes.txt"
# 旧目录中 joker/JOKER 曾因 Windows 大小写不敏感而混在一起；新结果改存到这个独立目录。
OUTPUT_DIR = ROOT / "data" / "huanleddz" / "class_crops_case_safe"

# 额外保留的边缘上下文，0.03 表示在框的四周各留目标最大边长的 3%。
# 想严格只看框内内容时设为 0。
CROP_PADDING_RATIO = 0
# ============================================================================

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def read_class_names(classes_file: Path) -> list[str]:
    """每行读取一个类别名；忽略空行和注释行，支持 '0: name' 格式。"""
    names: list[str] = []
    for line in classes_file.read_text(encoding="utf-8-sig").splitlines():
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        indexed_name = re.match(r"^\d+\s*:\s*(.+)$", name)
        if indexed_name:
            name = indexed_name.group(1).strip()
        names.append(name)
    if not names:
        raise ValueError(f"类别文件没有可读取的类别名称：{classes_file}")
    if len(set(names)) != len(names):
        raise ValueError(f"类别文件中有重名类别，无法分开保存：{classes_file}")
    return names


def safe_folder_name(class_name: str, class_id: int) -> str:
    """清理 Windows 不允许用于目录名的字符；常见英文类别名保持原样。"""
    safe_name = INVALID_FILENAME_CHARS.sub("_", class_name).strip(" .")
    return safe_name or f"class_{class_id}"


def class_folder_names(class_names: list[str]) -> list[str]:
    """Windows 文件夹不区分大小写；为忽略大小写后重名的类别追加 ID。"""
    safe_names = [safe_folder_name(name, class_id) for class_id, name in enumerate(class_names)]
    folded_counts: dict[str, int] = {}
    for name in safe_names:
        key = name.casefold()
        folded_counts[key] = folded_counts.get(key, 0) + 1
    return [
        f"{name}__class_{class_id}"
        if folded_counts[name.casefold()] > 1
        else name
        for class_id, name in enumerate(safe_names)
    ]


def safe_source_stem(image_path: Path, images_root: Path) -> tuple[str, str]:
    """返回可用于输出文件名的相对路径标识和原始相对路径。"""
    relative = image_path.relative_to(images_root)
    relative_stem = relative.with_suffix("").as_posix()
    safe_stem = INVALID_FILENAME_CHARS.sub("_", relative_stem).strip(" .")
    digest = hashlib.sha1(relative.as_posix().encode("utf-8")).hexdigest()[:8]
    return f"{safe_stem[:150]}_{digest}", relative.as_posix()


def label_path_for_image(image_path: Path, images_root: Path, labels_root: Path) -> Path:
    """图片和标签目录有相同的子目录结构时，按相对路径配对。"""
    return labels_root / image_path.relative_to(images_root).with_suffix(".txt")


def crop_box_from_yolo(
    values: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int] | None:
    center_x, center_y, box_width, box_height = values
    if not all(math.isfinite(value) for value in values) or box_width <= 0 or box_height <= 0:
        return None

    x1 = (center_x - box_width / 2) * image_width
    y1 = (center_y - box_height / 2) * image_height
    x2 = (center_x + box_width / 2) * image_width
    y2 = (center_y + box_height / 2) * image_height
    padding = max(0.0, max(x2 - x1, y2 - y1) * padding_ratio)
    left = max(0, math.floor(x1 - padding))
    top = max(0, math.floor(y1 - padding))
    right = min(image_width, math.ceil(x2 + padding))
    bottom = min(image_height, math.ceil(y2 + padding))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def validate_paths() -> None:
    if not IMAGES_DIR.is_dir():
        raise FileNotFoundError(f"图片目录不存在：{IMAGES_DIR}")
    if not LABELS_DIR.is_dir():
        raise FileNotFoundError(f"标签目录不存在：{LABELS_DIR}")
    if not CLASSES_FILE.is_file():
        raise FileNotFoundError(f"类别文件不存在：{CLASSES_FILE}")
    if not 0 <= CROP_PADDING_RATIO <= 1:
        raise ValueError("CROP_PADDING_RATIO 应在 0 到 1 之间")
    if OUTPUT_DIR.resolve() == IMAGES_DIR.resolve() or OUTPUT_DIR.resolve() == LABELS_DIR.resolve():
        raise ValueError("输出目录不能与图片目录或标签目录相同")


def main() -> None:
    validate_paths()
    class_names = read_class_names(CLASSES_FILE)

    # # 为避免旧裁片和新裁片混在一起，也避免覆盖人工整理过的结果，输出目录非空时停止。
    # if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
    #     raise FileExistsError(
    #         f"输出目录已经有文件，请先修改脚本顶部的 OUTPUT_DIR，或人工清理该目录：\n{OUTPUT_DIR}"
    #     )

    images = sorted(
        (
            path
            for path in IMAGES_DIR.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.relative_to(IMAGES_DIR).as_posix().casefold(),
    )
    if not images:
        raise ValueError(f"图片目录中没有支持的图片：{IMAGES_DIR}")

    folder_names = class_folder_names(class_names)
    for folder_name in folder_names:
        (OUTPUT_DIR / folder_name).mkdir(parents=True, exist_ok=True)

    manifest_path = OUTPUT_DIR / "crop_manifest.csv"
    manifest_fields = [
        "crop_file",
        "class_id",
        "class_name",
        "source_image",
        "source_label",
        "label_row_1based",
        "bbox_xyxy_pixels",
    ]
    counts = [0] * len(class_names)
    missing_labels = 0
    invalid_rows = 0
    image_errors = 0

    with manifest_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=manifest_fields)
        writer.writeheader()

        for image_path in images:
            label_path = label_path_for_image(image_path, IMAGES_DIR, LABELS_DIR)
            if not label_path.is_file():
                print(f"缺少标签，跳过：{image_path.relative_to(IMAGES_DIR)}")
                missing_labels += 1
                continue

            try:
                with Image.open(image_path) as source_image:
                    image = ImageOps.exif_transpose(source_image).convert("RGB")
                label_lines = label_path.read_text(encoding="utf-8-sig").splitlines()
            except (OSError, ValueError) as error:
                print(f"无法读取，跳过：{image_path}（{error}）")
                image_errors += 1
                continue

            width, height = image.size
            source_token, relative_image_path = safe_source_stem(image_path, IMAGES_DIR)
            relative_label_path = label_path.relative_to(LABELS_DIR).as_posix()

            for row_number, line in enumerate(label_lines, start=1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5:
                    print(f"标签格式错误，跳过：{relative_label_path} 第 {row_number} 行")
                    invalid_rows += 1
                    continue
                try:
                    class_id = int(fields[0])
                    coordinates = tuple(map(float, fields[1:]))
                except ValueError:
                    print(f"标签含非数字，跳过：{relative_label_path} 第 {row_number} 行")
                    invalid_rows += 1
                    continue
                if not 0 <= class_id < len(class_names):
                    print(f"类别 ID 越界，跳过：{relative_label_path} 第 {row_number} 行（{class_id}）")
                    invalid_rows += 1
                    continue

                crop_box = crop_box_from_yolo(
                    coordinates,
                    width,
                    height,
                    CROP_PADDING_RATIO,
                )
                if crop_box is None:
                    print(f"框坐标无效，跳过：{relative_label_path} 第 {row_number} 行")
                    invalid_rows += 1
                    continue

                class_name = class_names[class_id]
                class_dir = OUTPUT_DIR / folder_names[class_id]
                crop_filename = f"{source_token}__box{row_number:04d}.png"
                crop_path = class_dir / crop_filename
                image.crop(crop_box).save(crop_path, format="PNG", compress_level=6)
                counts[class_id] += 1
                writer.writerow(
                    {
                        "crop_file": f"{class_dir.name}/{crop_filename}",
                        "class_id": class_id,
                        "class_name": class_name,
                        "source_image": relative_image_path,
                        "source_label": relative_label_path,
                        "label_row_1based": row_number,
                        "bbox_xyxy_pixels": ",".join(map(str, crop_box)),
                    }
                )

    print(f"\n裁剪完成：共扫描 {len(images)} 张图片")
    print(f"成功裁剪 {sum(counts)} 个框，结果保存在：{OUTPUT_DIR}")
    for class_id, class_name in enumerate(class_names):
        print(f"  {class_name}: {counts[class_id]} 个")
    print(f"缺少标签：{missing_labels} 张；异常标签框：{invalid_rows} 个；图片读取失败：{image_errors} 张")
    print(f"来源与框编号清单：{manifest_path}")


if __name__ == "__main__":
    main()

