"""Measure a single CPU prediction after one warm-up pass."""
from pathlib import Path
from time import perf_counter

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "runs" / "detect" / "s模型, imgsz = 960" / "weights" / "best.pt"
IMAGE_PATH = ROOT / "data" / "test_images" / "jj0.png"


def main() -> None:
    model = YOLO(str(MODEL_PATH))
    model.predict(str(IMAGE_PATH), conf=0.5, device="cpu", verbose=False)

    started = perf_counter()
    model.predict(str(IMAGE_PATH), conf=0.5, device="cpu", verbose=False)
    elapsed = perf_counter() - started
    print(f"CPU 单次推理耗时：{elapsed:.4f} 秒")


if __name__ == "__main__":
    main()

