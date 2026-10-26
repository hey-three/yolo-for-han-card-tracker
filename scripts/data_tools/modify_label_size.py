"""Scale YOLO bounding-box width and height without changing the top edge."""
import argparse
from typing import Optional
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def modify_labels(
    input_dir: Path,
    output_dir: Optional[Path],
    height_scale: float,
    width_scale: float,
    backup: bool,
) -> None:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"标注目录不存在：{input_dir}")
    if height_scale <= 0 or width_scale <= 0:
        raise ValueError("宽度和高度缩放比例必须大于 0")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.txt"))
    box_count = 0
    for source in files:
        original_lines = source.read_text(encoding="utf-8").splitlines()
        converted = []
        for line_number, line in enumerate(original_lines, start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"格式错误：{source.name}:{line_number}")

            class_id = parts[0]
            center_x, center_y, width, height = map(float, parts[1:])
            top_edge = center_y - height / 2
            new_width = width * width_scale
            new_height = height * height_scale
            new_center_y = top_edge + new_height / 2
            converted.append(
                f"{class_id} {center_x:.6f} {new_center_y:.6f} "
                f"{new_width:.6f} {new_height:.6f}"
            )
            box_count += 1

        destination = source if output_dir is None else output_dir / source.name
        if output_dir is None and backup:
            source.with_suffix(source.suffix + ".bak").write_text(
                "".join(f"{line}\n" for line in original_lines), encoding="utf-8"
            )
        destination.write_text("".join(f"{line}\n" for line in converted), encoding="utf-8")

    print(f"处理完成：{len(files)} 个文件，{box_count} 个标注框。")
    print(f"输出目录：{output_dir or input_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="缩放 YOLO 标注框的宽度和高度")
    parser.add_argument("--input", required=True, help="输入标注目录")
    parser.add_argument("--output", help="输出目录；省略时原地修改")
    parser.add_argument("--height-scale", type=float, default=0.4)
    parser.add_argument("--width-scale", type=float, default=1.0)
    parser.add_argument("--no-backup", action="store_true", help="原地修改时不备份")
    args = parser.parse_args()

    modify_labels(
        input_dir=resolve_path(args.input),
        output_dir=resolve_path(args.output) if args.output else None,
        height_scale=args.height_scale,
        width_scale=args.width_scale,
        backup=not args.no_backup,
    )


if __name__ == "__main__":
    main()


