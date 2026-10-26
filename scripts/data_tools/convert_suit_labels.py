"""Convert suit-specific card labels to rank-only labels."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "label_conversion" / "suit_agnostic"
LABEL_DIR = DATA_DIR / "labels_raw"
OUTPUT_DIR = DATA_DIR / "labels"
CLASS_FILE = DATA_DIR / "classes_raw.txt"
OUTPUT_CLASS_FILE = DATA_DIR / "classes.txt"
KEEP_RATIO = 0.6

RANK_MAP = {
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
    "J": "J",
    "Q": "Q",
    "K": "K",
    "A": "A",
}
NEW_CLASSES = [
    "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "J", "Q", "K", "A", "ljoker", "bjoker",
]


def convert_class_name(name: str) -> str:
    if name in {"ljoker", "bjoker"}:
        return name
    for suffix, rank in RANK_MAP.items():
        if name.endswith(suffix):
            return rank
    raise ValueError(f"无法识别的原始类别：{name}")


def main() -> None:
    old_classes = [line.strip() for line in CLASS_FILE.read_text(encoding="utf-8").splitlines()]
    new_class_ids = {name: index for index, name in enumerate(NEW_CLASSES)}
    old_to_new = {
        index: new_class_ids[convert_class_name(name)]
        for index, name in enumerate(old_classes)
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CLASS_FILE.write_text("".join(f"{name}\n" for name in NEW_CLASSES), encoding="utf-8")

    for source in sorted(LABEL_DIR.glob("*.txt")):
        converted = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            class_value, x, y, width, height = line.split()
            old_class_id = int(class_value)
            old_class_name = old_classes[old_class_id]

            if old_class_name not in {"ljoker", "bjoker"}:
                original_height = float(height)
                height = original_height * KEEP_RATIO
                y = float(y) - (original_height - height) / 2

            converted.append(
                f"{old_to_new[old_class_id]} {float(x):.6f} {float(y):.6f} "
                f"{float(width):.6f} {float(height):.6f}"
            )

        (OUTPUT_DIR / source.name).write_text(
            "".join(f"{line}\n" for line in converted), encoding="utf-8"
        )

    print(f"转换完成：{LABEL_DIR} -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

