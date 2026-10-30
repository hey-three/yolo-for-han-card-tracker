"""YOLO 矩形框标注查看与编辑工具。

使用方法
========
运行本文件后，依次选择图片文件夹、对应的 YOLO 标签文件夹和类别 txt 文件。
类别 txt 每行一个类别名，行号从 0 开始就是 YOLO 标签里的 class_id。
图片与标签按相对路径和同名文件配对；例如 images/train/a.jpg 对应
labels/train/a.txt。标注文件使用 YOLO 检测格式：class_id x_center y_center width height，
后四项均为相对于整张原图的 0–1 归一化坐标。

鼠标操作
========
* 在没有框的位置按住左键拖动：用顶部选择的类别新画一个框。
* 点击已有框：选中它；在框内拖动可移动框，拖白色控制点可改变框的大小。
* 按住 Ctrl 滚动鼠标滚轮：以鼠标指向的位置为中心放大或缩小。
* 放大后按住鼠标中键拖动：平移查看图像的其他区域。
* 鼠标右键：取消当前选择或中断正在进行的绘框。

按钮和快捷键
============
* “设置”：切换图片时自动保存、在下一张图沿用当前缩放和视口位置。
* “适应窗口”或按 0：恢复完整图片视图。
* 左/右方向键或上一张/下一张：切换图片；输入序号可跳转。
* Ctrl+S 保存，Ctrl+Z 撤销，Delete 删除选中框，B 隐藏/显示已有框。
* 切换图片时若有未保存修改，按设置决定自动保存或询问；关闭窗口也会遵循该设置。

两项设置默认开启，只在本次运行期间生效。视图跟随会把当前缩放倍数和图像中心的
相对位置应用到下一张图片；图片尺寸不同时也按相对位置换算。保存只会写入所选标签目录，
不会改动图片。清空本图、删除框和改类别都可以用 Ctrl+Z 撤销。
"""

from __future__ import annotations

import argparse
import colorsys
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk


ROOT = Path(__file__).resolve().parents[2]

# 可在这里填写默认目录；留空时，运行后会依次弹出三个选择窗口。
IMAGES_DIR: Optional[Path] = None
LABELS_DIR: Optional[Path] = None
CLASSES_FILE: Optional[Path] = None

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
MAX_UNDO_STEPS = 100
MIN_BOX_SIZE_PIXELS = 1.0
MIN_ZOOM = 1.0
MAX_ZOOM = 12.0
ZOOM_STEP = 1.25


@dataclass(frozen=True)
class Box:
    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float


def resolve_path(value: Optional[Path]) -> Optional[Path]:
    if value is None:
        return None
    path = value.expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def read_class_names(classes_file: Path) -> list[str]:
    """Read one class per line, also accepting lines such as '0: ace' or '0 ace'."""
    names: list[str] = []
    for line in classes_file.read_text(encoding="utf-8-sig").splitlines():
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        indexed = re.match(r"^\d+\s*:\s*(.+)$", name)
        if indexed:
            name = indexed.group(1).strip()
        else:
            indexed = re.match(r"^\d+\s+(.+)$", name)
            if indexed:
                name = indexed.group(1).strip()
        names.append(name)
    if not names:
        raise ValueError(f"类别文件为空或没有可读取的类别名称：\n{classes_file}")
    return names


def class_color(class_id: int) -> str:
    hue = (class_id * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.82, 1.0)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def read_yolo_labels(label_path: Path) -> tuple[list[Box], list[str], int]:
    """Return valid YOLO boxes, warnings, and the number of skipped bad rows."""
    boxes: list[Box] = []
    warnings: list[str] = []
    skipped_rows = 0
    if not label_path.is_file():
        return boxes, warnings, skipped_rows
    try:
        lines = label_path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as error:
        return boxes, [f"无法读取标签文件：{error}"], skipped_rows

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            warnings.append(f"第 {line_number} 行不是 YOLO 检测格式（应有 5 列）")
            skipped_rows += 1
            continue
        try:
            class_id = int(parts[0])
            center_x, center_y, box_width, box_height = map(float, parts[1:])
        except ValueError:
            warnings.append(f"第 {line_number} 行含有无法读取的数字")
            skipped_rows += 1
            continue

        coordinates = (center_x, center_y, box_width, box_height)
        if not all(math.isfinite(value) for value in coordinates):
            warnings.append(f"第 {line_number} 行含有非有限坐标")
            skipped_rows += 1
            continue
        if box_width <= 0 or box_height <= 0:
            warnings.append(f"第 {line_number} 行的框宽或框高不大于 0")
            skipped_rows += 1
            continue
        if any(value < 0 or value > 1 for value in coordinates):
            warnings.append(f"第 {line_number} 行坐标超出 0–1 范围，保存时会裁到图像边缘")
        boxes.append(Box(class_id, center_x, center_y, box_width, box_height))
    return boxes, warnings, skipped_rows


class LabelViewer:
    def __init__(self, root: tk.Tk, images_dir: Path, labels_dir: Path, classes_file: Path) -> None:
        self.root = root
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.classes_file = classes_file
        self.class_names = read_class_names(classes_file)
        self.image_paths = sorted(
            (
                path
                for path in images_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: str(path.relative_to(images_dir)).casefold(),
        )
        if not self.image_paths:
            raise ValueError(f"图片目录里没有可识别的图片：\n{images_dir}")

        self.index = 0
        self.boxes: list[Box] = []
        self.selected_box: Optional[int] = None
        self.history: list[tuple[Box, ...]] = []
        self.show_boxes = tk.BooleanVar(value=True)
        # 这两项可以在“设置”窗口里随时开关；默认符合用户的连续标注流程。
        self.auto_save_on_navigation = tk.BooleanVar(value=True)
        self.keep_view_on_navigation = tk.BooleanVar(value=True)
        self.class_var = tk.StringVar()
        self.jump_var = tk.StringVar(value="1")
        self.info_var = tk.StringVar()
        self.paths_var = tk.StringVar(
            value=f"图片：{images_dir}    标签：{labels_dir}    类别：{classes_file}"
        )
        self.hint_var = tk.StringVar()
        self.zoom_var = tk.StringVar(value="100%")

        self.image_path: Optional[Path] = None
        self.label_path: Optional[Path] = None
        self.image: Optional[Image.Image] = None
        self.photo: Optional[ImageTk.PhotoImage] = None
        self.current_warnings: list[str] = []
        self.skipped_rows = 0
        self.label_was_missing = False
        self.saved_state: tuple[Box, ...] = ()
        self.drag: Optional[dict[str, object]] = None

        self.scale = 1.0
        self.fit_scale = 1.0
        self.zoom_factor = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.display_width = 0
        self.display_height = 0
        self.last_canvas_size = (0, 0)
        self.view_initialized = False
        self.pending_view_center: Optional[tuple[float, float]] = None
        # 导航时暂存视图：zoom 倍数 + 归一化视口中心 (x, y)。
        self.navigation_view: Optional[tuple[float, float, float]] = None
        self.pan_start: Optional[tuple[float, float, float, float]] = None

        self.root.title("YOLO 标注查看与编辑")
        self.root.geometry("1380x900")
        self.root.minsize(850, 600)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.build_ui()
        self.bind_events()
        self.load_current_image()

    def build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=(10, 8, 10, 4))
        top.pack(fill=tk.X)
        ttk.Label(top, textvariable=self.info_var).pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(top, text="新框类别：").pack(side=tk.LEFT, padx=(10, 4))
        self.class_combo = ttk.Combobox(
            top,
            textvariable=self.class_var,
            values=[self.class_option(class_id) for class_id in range(len(self.class_names))],
            state="readonly",
            width=20,
        )
        self.class_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.class_combo.bind("<<ComboboxSelected>>", self.on_class_selected)
        ttk.Button(top, text="设置", command=self.open_settings).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(top, text="使用说明", command=self.show_help).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(top, text="适应窗口 (0)", command=self.reset_view).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(top, textvariable=self.zoom_var, width=6).pack(side=tk.LEFT)
        ttk.Checkbutton(
            top,
            text="显示已有框 (B)",
            variable=self.show_boxes,
            command=self.render,
        ).pack(side=tk.RIGHT)

        ttk.Label(
            self.root,
            textvariable=self.paths_var,
            padding=(10, 0, 10, 4),
            wraplength=1300,
        ).pack(fill=tk.X)

        self.canvas = tk.Canvas(self.root, background="#202124", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        bottom = ttk.Frame(self.root, padding=(10, 4, 10, 2))
        bottom.pack(fill=tk.X)
        ttk.Button(bottom, text="上一张  (←)", command=self.previous).pack(side=tk.LEFT)
        ttk.Button(bottom, text="下一张  (→)", command=self.next).pack(side=tk.LEFT, padx=(8, 12))
        ttk.Label(bottom, text="跳到图片：").pack(side=tk.LEFT)
        self.jump_entry = ttk.Entry(bottom, textvariable=self.jump_var, width=7)
        self.jump_entry.pack(side=tk.LEFT, padx=4)
        self.jump_entry.bind("<Return>", self.jump)
        ttk.Button(bottom, text="跳转", command=self.jump).pack(side=tk.LEFT, padx=(0, 16))

        ttk.Button(bottom, text="保存标注  (Ctrl+S)", command=self.save_current).pack(side=tk.LEFT)
        ttk.Button(bottom, text="撤销  (Ctrl+Z)", command=self.undo).pack(side=tk.LEFT, padx=6)
        ttk.Button(bottom, text="删除选中框  (Delete)", command=self.delete_selected).pack(side=tk.LEFT)
        ttk.Button(bottom, text="清空本图", command=self.clear_current).pack(side=tk.LEFT, padx=6)

        ttk.Label(
            self.root,
            textvariable=self.hint_var,
            padding=(10, 3, 10, 8),
            anchor=tk.W,
            wraplength=1320,
        ).pack(fill=tk.X)

    def open_settings(self) -> None:
        """Open the two requested annotation workflow switches."""
        dialog = tk.Toplevel(self.root)
        dialog.title("标注设置")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.grab_set()

        auto_save_value = tk.BooleanVar(master=dialog, value=self.auto_save_on_navigation.get())
        keep_view_value = tk.BooleanVar(master=dialog, value=self.keep_view_on_navigation.get())

        body = ttk.Frame(dialog, padding=16)
        body.pack(fill=tk.BOTH, expand=True)
        ttk.Label(body, text="切换图片", font=("Microsoft YaHei UI", 11, "bold")).pack(anchor=tk.W)
        ttk.Checkbutton(
            body,
            text="切换图片或关闭窗口时，自动保存当前修改",
            variable=auto_save_value,
        ).pack(anchor=tk.W, pady=(8, 2))
        ttk.Label(
            body,
            text="只在标注确实有改动时保存；关闭此开关后，切图时会询问保存、放弃或取消。",
            wraplength=500,
        ).pack(anchor=tk.W, padx=(24, 0), pady=(0, 14))

        ttk.Label(body, text="连续查看相同区域", font=("Microsoft YaHei UI", 11, "bold")).pack(anchor=tk.W)
        ttk.Checkbutton(
            body,
            text="切换到下一张时沿用当前放大倍数和视口位置",
            variable=keep_view_value,
        ).pack(anchor=tk.W, pady=(8, 2))
        ttk.Label(
            body,
            text="视口位置按图像宽、高的相对位置换算；关闭后，每张新图片都从完整视图开始。",
            wraplength=500,
        ).pack(anchor=tk.W, padx=(24, 0), pady=(0, 12))
        ttk.Label(body, text="设置仅在当前程序运行期间有效。", foreground="#666666").pack(anchor=tk.W)

        buttons = ttk.Frame(body)
        buttons.pack(fill=tk.X, pady=(16, 0))

        def apply_settings() -> None:
            self.auto_save_on_navigation.set(auto_save_value.get())
            self.keep_view_on_navigation.set(keep_view_value.get())
            dialog.destroy()

        ttk.Button(buttons, text="取消", command=dialog.destroy).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="确定", command=apply_settings).pack(side=tk.RIGHT, padx=(0, 8))
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        dialog.bind("<Return>", lambda _event: apply_settings())
        self.root.wait_window(dialog)

    def show_help(self) -> None:
        messagebox.showinfo(
            "使用说明",
            "左键拖动空白处新建框；点击框后可拖动移动，拖白色控制点可改大小。\n"
            "按住 Ctrl 滚动滚轮，以鼠标位置为中心缩放；按住鼠标中键拖动平移。\n"
            "顶部类别选择器决定新框类别，也可以选中框后更改类别。\n\n"
            "Ctrl+S 保存，Ctrl+Z 撤销，Delete 删除框，B 显示/隐藏框，0 适应窗口。\n"
            "“设置”中可切换自动保存和下一张图片沿用视图。",
            parent=self.root,
        )

    def bind_events(self) -> None:
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<ButtonPress-2>", self.on_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self.on_pan_end)

        self.root.bind("<Left>", self.on_previous_key)
        self.root.bind("<Right>", self.on_next_key)
        self.root.bind("<Delete>", self.on_delete_key)
        self.root.bind("<Control-s>", self.on_save_key)
        self.root.bind("<Control-S>", self.on_save_key)
        self.root.bind("<Control-z>", self.on_undo_key)
        self.root.bind("<Control-Z>", self.on_undo_key)
        self.root.bind("b", self.on_toggle_key)
        self.root.bind("B", self.on_toggle_key)
        self.root.bind("0", self.on_reset_view_key)

    def class_option(self, class_id: int) -> str:
        return f"{class_id}: {self.class_names[class_id]}"

    def class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"未定义类别 {class_id}"

    def selected_class_id(self) -> int:
        current = self.class_combo.current()
        return current if current >= 0 else 0

    def state(self) -> tuple[Box, ...]:
        return tuple(self.boxes)

    @property
    def is_dirty(self) -> bool:
        return self.state() != self.saved_state

    def paths_for_current_image(self) -> tuple[Path, Path]:
        image_path = self.image_paths[self.index]
        relative_path = image_path.relative_to(self.images_dir)
        label_path = (self.labels_dir / relative_path).with_suffix(".txt")
        return image_path, label_path

    def load_current_image(self) -> None:
        image_path, label_path = self.paths_for_current_image()
        try:
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            boxes, warnings, skipped_rows = read_yolo_labels(label_path)
        except (OSError, ValueError) as error:
            messagebox.showerror("无法打开数据", f"{image_path}\n\n{error}", parent=self.root)
            return

        self.image_path = image_path
        self.label_path = label_path
        self.image = image
        inherited_view = self.navigation_view if self.keep_view_on_navigation.get() else None
        self.navigation_view = None
        self.view_initialized = False
        if inherited_view is None:
            self.zoom_factor = 1.0
            self.pending_view_center = None
        else:
            self.zoom_factor = min(max(inherited_view[0], MIN_ZOOM), MAX_ZOOM)
            self.pending_view_center = (
                inherited_view[1] * image.width,
                inherited_view[2] * image.height,
            )
        self.boxes = boxes
        self.selected_box = None
        self.history.clear()
        self.drag = None
        self.current_warnings = warnings
        self.skipped_rows = skipped_rows
        self.label_was_missing = not label_path.is_file()
        self.saved_state = self.state()
        self.jump_var.set(str(self.index + 1))
        if self.boxes and 0 <= self.boxes[0].class_id < len(self.class_names):
            self.class_combo.current(self.boxes[0].class_id)
        elif self.class_names:
            self.class_combo.current(0)
        self.render()

    def render(self) -> None:
        if self.image is None:
            return
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        image_width, image_height = self.image.size
        self.fit_scale = min(canvas_width / image_width, canvas_height / image_height)
        self.scale = self.fit_scale * self.zoom_factor
        self.display_width = max(1, round(image_width * self.scale))
        self.display_height = max(1, round(image_height * self.scale))

        if self.pending_view_center is not None:
            center_x, center_y = self.pending_view_center
            self.offset_x = canvas_width / 2 - center_x * self.scale
            self.offset_y = canvas_height / 2 - center_y * self.scale
            self.pending_view_center = None
            self.view_initialized = True
        elif not self.view_initialized:
            self.offset_x = (canvas_width - self.display_width) / 2
            self.offset_y = (canvas_height - self.display_height) / 2
            self.view_initialized = True
        self.clamp_view(canvas_width, canvas_height)

        resampling = getattr(Image, "Resampling", Image)
        source_left = max(0, math.floor((0 - self.offset_x) / self.scale))
        source_top = max(0, math.floor((0 - self.offset_y) / self.scale))
        source_right = min(image_width, math.ceil((canvas_width - self.offset_x) / self.scale))
        source_bottom = min(image_height, math.ceil((canvas_height - self.offset_y) / self.scale))
        source_left = min(source_left, image_width - 1)
        source_top = min(source_top, image_height - 1)
        source_right = max(source_left + 1, source_right)
        source_bottom = max(source_top + 1, source_bottom)
        visible_image = self.image.crop((source_left, source_top, source_right, source_bottom))
        visible_width = max(1, round(visible_image.width * self.scale))
        visible_height = max(1, round(visible_image.height * self.scale))
        display_image = visible_image.resize((visible_width, visible_height), resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(display_image)
        self.canvas.delete("all")
        self.canvas.create_image(
            self.offset_x + source_left * self.scale,
            self.offset_y + source_top * self.scale,
            image=self.photo,
            anchor=tk.NW,
            tags=("image",),
        )

        if self.show_boxes.get():
            for index, box in enumerate(self.boxes):
                self.draw_box(index, box)
        if self.drag and self.drag.get("mode") == "draw":
            self.draw_preview()

        label_state = "缺少标签文件" if self.label_was_missing else f"{len(self.boxes)} 个框"
        changed = "  ● 未保存" if self.is_dirty else ""
        self.info_var.set(
            f"{self.index + 1}/{len(self.image_paths)} | {self.image_path.name} | "
            f"{image_width}×{image_height} | {label_state}{changed}"
        )
        self.zoom_var.set(f"{self.zoom_factor:.1f}×")
        self.jump_var.set(str(self.index + 1))
        if self.current_warnings:
            shown = "；".join(self.current_warnings[:2])
            extra = f"；另有 {len(self.current_warnings) - 2} 项" if len(self.current_warnings) > 2 else ""
            self.hint_var.set(shown + extra)
        else:
            self.hint_var.set(
                "空白处左键拖动新建框；点击已有框选中，框内拖动移动，拖动白色控制点调整大小。"
                "Ctrl+滚轮以鼠标位置为中心缩放；中键拖动平移；0 适应窗口；"
                "Delete 删除，Ctrl+S 保存，Ctrl+Z 撤销。"
            )

    def clamp_view(self, canvas_width: int, canvas_height: int) -> None:
        """Keep a zoomed region visible without breaking cursor-anchored zoom."""
        if self.display_width <= canvas_width:
            min_x, max_x = 1 - self.display_width, canvas_width - 1
        else:
            min_x, max_x = canvas_width - self.display_width, 0.0
        if self.display_height <= canvas_height:
            min_y, max_y = 1 - self.display_height, canvas_height - 1
        else:
            min_y, max_y = canvas_height - self.display_height, 0.0
        self.offset_x = min(max(self.offset_x, min_x), max_x)
        self.offset_y = min(max(self.offset_y, min_y), max_y)

    def draw_box(self, index: int, box: Box) -> None:
        left, top, right, bottom = self.box_canvas_edges(box)
        color = class_color(box.class_id)
        selected = index == self.selected_box
        outline = "#ffffff" if selected else color
        width = 3 if selected else 2
        self.canvas.create_rectangle(left, top, right, bottom, outline=outline, width=width)

        caption = self.class_name(box.class_id)
        font = ("Microsoft YaHei UI", 10, "bold")
        text_width = max(20, len(caption) * 8)
        label_height = 20
        caption_top = max(self.offset_y, top - label_height)
        caption_bottom = caption_top + label_height
        caption_right = min(self.offset_x + self.display_width, left + text_width + 10)
        self.canvas.create_rectangle(left, caption_top, caption_right, caption_bottom, fill=color, outline=color)
        self.canvas.create_text(
            left + 5,
            caption_top + label_height / 2,
            text=caption,
            fill="white",
            font=font,
            anchor=tk.W,
        )
        if selected:
            for handle_x, handle_y in self.handle_positions(box).values():
                radius = 4
                self.canvas.create_rectangle(
                    handle_x - radius,
                    handle_y - radius,
                    handle_x + radius,
                    handle_y + radius,
                    fill=color,
                    outline="white",
                    width=1,
                )

    def draw_preview(self) -> None:
        if not self.drag or self.drag.get("mode") != "draw":
            return
        start = self.drag.get("start")
        current = self.drag.get("current", start)
        if not isinstance(start, tuple) or not isinstance(current, tuple):
            return
        x1, y1 = self.image_to_canvas(start[0], start[1])
        x2, y2 = self.image_to_canvas(current[0], current[1])
        self.canvas.create_rectangle(
            x1,
            y1,
            x2,
            y2,
            outline=class_color(int(self.drag.get("class_id", 0))),
            width=2,
            dash=(5, 3),
        )

    def box_canvas_edges(self, box: Box) -> tuple[float, float, float, float]:
        if self.image is None:
            return 0.0, 0.0, 0.0, 0.0
        image_width, image_height = self.image.size
        left = self.offset_x + (box.center_x - box.width / 2) * image_width * self.scale
        top = self.offset_y + (box.center_y - box.height / 2) * image_height * self.scale
        right = self.offset_x + (box.center_x + box.width / 2) * image_width * self.scale
        bottom = self.offset_y + (box.center_y + box.height / 2) * image_height * self.scale
        return left, top, right, bottom

    def handle_positions(self, box: Box) -> dict[str, tuple[float, float]]:
        left, top, right, bottom = self.box_canvas_edges(box)
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        return {
            "nw": (left, top),
            "n": (center_x, top),
            "ne": (right, top),
            "e": (right, center_y),
            "se": (right, bottom),
            "s": (center_x, bottom),
            "sw": (left, bottom),
            "w": (left, center_y),
        }

    def canvas_to_image(self, canvas_x: float, canvas_y: float) -> Optional[tuple[float, float]]:
        if self.image is None:
            return None
        if not (
            self.offset_x <= canvas_x <= self.offset_x + self.display_width
            and self.offset_y <= canvas_y <= self.offset_y + self.display_height
        ):
            return None
        image_width, image_height = self.image.size
        image_x = min(max((canvas_x - self.offset_x) / self.scale, 0.0), image_width)
        image_y = min(max((canvas_y - self.offset_y) / self.scale, 0.0), image_height)
        return image_x, image_y

    def image_to_canvas(self, image_x: float, image_y: float) -> tuple[float, float]:
        return self.offset_x + image_x * self.scale, self.offset_y + image_y * self.scale

    def box_pixel_edges(self, box: Box) -> tuple[float, float, float, float]:
        if self.image is None:
            return 0.0, 0.0, 0.0, 0.0
        image_width, image_height = self.image.size
        x1 = (box.center_x - box.width / 2) * image_width
        y1 = (box.center_y - box.height / 2) * image_height
        x2 = (box.center_x + box.width / 2) * image_width
        y2 = (box.center_y + box.height / 2) * image_height
        return (
            min(max(x1, 0.0), image_width),
            min(max(y1, 0.0), image_height),
            min(max(x2, 0.0), image_width),
            min(max(y2, 0.0), image_height),
        )

    def box_from_pixel_edges(self, class_id: int, x1: float, y1: float, x2: float, y2: float) -> Box:
        if self.image is None:
            return Box(class_id, 0.5, 0.5, 0.01, 0.01)
        image_width, image_height = self.image.size
        x1 = min(max(x1, 0.0), image_width - MIN_BOX_SIZE_PIXELS)
        y1 = min(max(y1, 0.0), image_height - MIN_BOX_SIZE_PIXELS)
        x2 = min(max(x2, x1 + MIN_BOX_SIZE_PIXELS), float(image_width))
        y2 = min(max(y2, y1 + MIN_BOX_SIZE_PIXELS), float(image_height))
        return Box(
            class_id,
            ((x1 + x2) / 2) / image_width,
            ((y1 + y2) / 2) / image_height,
            (x2 - x1) / image_width,
            (y2 - y1) / image_height,
        )

    def hit_test(self, canvas_x: float, canvas_y: float) -> Optional[tuple[int, str]]:
        if not self.show_boxes.get():
            return None
        if self.selected_box is not None and 0 <= self.selected_box < len(self.boxes):
            box = self.boxes[self.selected_box]
            for handle, (handle_x, handle_y) in self.handle_positions(box).items():
                if math.hypot(canvas_x - handle_x, canvas_y - handle_y) <= 8:
                    return self.selected_box, handle

        for index in range(len(self.boxes) - 1, -1, -1):
            left, top, right, bottom = self.box_canvas_edges(self.boxes[index])
            if left <= canvas_x <= right and top <= canvas_y <= bottom:
                return index, "move"
        return None

    def on_mouse_down(self, event: tk.Event) -> None:
        point = self.canvas_to_image(event.x, event.y)
        if point is None:
            return
        self.canvas.focus_set()
        hit = self.hit_test(event.x, event.y)
        before = self.state()
        if hit is not None:
            index, mode = hit
            self.selected_box = index
            box = self.boxes[index]
            if 0 <= box.class_id < len(self.class_names):
                self.class_combo.current(box.class_id)
            self.drag = {
                "mode": mode,
                "index": index,
                "start": point,
                "original": box,
                "before": before,
            }
        else:
            self.selected_box = None
            self.drag = {
                "mode": "draw",
                "start": point,
                "current": point,
                "class_id": self.selected_class_id(),
                "before": before,
            }
        self.render()

    def on_mouse_drag(self, event: tk.Event) -> None:
        if not self.drag:
            return
        point = self.canvas_to_image(event.x, event.y)
        if point is None:
            image_width, image_height = self.image.size if self.image else (0, 0)
            point = (
                min(max((event.x - self.offset_x) / self.scale, 0.0), image_width),
                min(max((event.y - self.offset_y) / self.scale, 0.0), image_height),
            )
        mode = str(self.drag.get("mode"))
        start = self.drag.get("start")
        if not isinstance(start, tuple):
            return

        if mode == "draw":
            self.drag["current"] = point
            self.render()
            return

        index = int(self.drag["index"])
        original = self.drag.get("original")
        if not isinstance(original, Box) or self.image is None:
            return
        image_width, image_height = self.image.size
        x1, y1, x2, y2 = self.box_pixel_edges(original)
        if mode == "move":
            delta_x, delta_y = point[0] - start[0], point[1] - start[1]
            box_width, box_height = x2 - x1, y2 - y1
            new_x1 = min(max(x1 + delta_x, 0.0), image_width - box_width)
            new_y1 = min(max(y1 + delta_y, 0.0), image_height - box_height)
            x1, y1 = new_x1, new_y1
            x2, y2 = x1 + box_width, y1 + box_height
        else:
            handle = mode
            if "w" in handle:
                x1 = min(max(point[0], 0.0), x2 - MIN_BOX_SIZE_PIXELS)
            if "e" in handle:
                x2 = min(max(point[0], x1 + MIN_BOX_SIZE_PIXELS), float(image_width))
            if "n" in handle:
                y1 = min(max(point[1], 0.0), y2 - MIN_BOX_SIZE_PIXELS)
            if "s" in handle:
                y2 = min(max(point[1], y1 + MIN_BOX_SIZE_PIXELS), float(image_height))

        self.boxes[index] = self.box_from_pixel_edges(original.class_id, x1, y1, x2, y2)
        self.render()

    def on_mouse_up(self, _event: tk.Event) -> None:
        if not self.drag:
            return
        mode = str(self.drag.get("mode"))
        before = self.drag.get("before", self.state())
        if mode == "draw" and self.image is not None:
            start = self.drag.get("start")
            current = self.drag.get("current", start)
            if isinstance(start, tuple) and isinstance(current, tuple):
                x1, x2 = sorted((start[0], current[0]))
                y1, y2 = sorted((start[1], current[1]))
                if x2 - x1 >= MIN_BOX_SIZE_PIXELS and y2 - y1 >= MIN_BOX_SIZE_PIXELS:
                    new_box = self.box_from_pixel_edges(int(self.drag.get("class_id", 0)), x1, y1, x2, y2)
                    self.boxes.append(new_box)
                    self.selected_box = len(self.boxes) - 1
                    self.class_combo.current(new_box.class_id)
        self.drag = None
        if isinstance(before, tuple):
            self.remember(before)
        self.render()

    def on_right_click(self, _event: tk.Event) -> None:
        self.drag = None
        self.selected_box = None
        self.render()

    def remember(self, before: tuple[Box, ...]) -> None:
        if before == self.state():
            return
        self.history.append(before)
        if len(self.history) > MAX_UNDO_STEPS:
            del self.history[0]

    def undo(self) -> None:
        if not self.history:
            self.hint_var.set("没有可撤销的操作")
            return
        self.boxes = list(self.history.pop())
        self.selected_box = None
        self.render()

    def delete_selected(self) -> None:
        if self.selected_box is None or not 0 <= self.selected_box < len(self.boxes):
            self.hint_var.set("请先点击要删除的框")
            return
        before = self.state()
        del self.boxes[self.selected_box]
        self.selected_box = None
        self.remember(before)
        self.render()

    def clear_current(self) -> None:
        if not self.boxes:
            self.hint_var.set("这张图片没有可清空的框")
            return
        if not messagebox.askyesno("清空本图标注", "确定删除当前图片的全部标注框吗？", parent=self.root):
            return
        before = self.state()
        self.boxes.clear()
        self.selected_box = None
        self.remember(before)
        self.render()

    def on_class_selected(self, _event: Optional[tk.Event] = None) -> None:
        if self.selected_box is None or not 0 <= self.selected_box < len(self.boxes):
            return
        class_id = self.selected_class_id()
        old_box = self.boxes[self.selected_box]
        if old_box.class_id == class_id:
            return
        before = self.state()
        self.boxes[self.selected_box] = Box(class_id, old_box.center_x, old_box.center_y, old_box.width, old_box.height)
        self.remember(before)
        self.render()

    def save_current(self) -> bool:
        if self.label_path is None:
            return False
        if self.skipped_rows and not messagebox.askyesno(
            "标签文件含有无效行",
            f"读取时跳过了 {self.skipped_rows} 行无法识别的标注。继续保存会移除这些无效行，是否继续？",
            parent=self.root,
        ):
            return False
        invalid_classes = sorted(
            {box.class_id for box in self.boxes if not 0 <= box.class_id < len(self.class_names)}
        )
        if invalid_classes:
            messagebox.showwarning(
                "类别编号无效",
                f"当前有类别编号不在 class.txt 范围内：{invalid_classes}。请选中对应框并重新选择类别，或删除它们后再保存。",
                parent=self.root,
            )
            return False

        try:
            self.label_path.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                f"{box.class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"
                for box in self.boxes
                for cx, cy, width, height in [self.clipped_yolo_box(box)]
                if width > 0 and height > 0
            ]
            self.label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        except OSError as error:
            messagebox.showerror("保存失败", f"无法保存标签：\n{self.label_path}\n\n{error}", parent=self.root)
            return False

        self.saved_state = self.state()
        self.label_was_missing = False
        self.current_warnings = []
        self.skipped_rows = 0
        self.render()
        self.hint_var.set(f"已保存：{self.label_path}")
        return True

    def clipped_yolo_box(self, box: Box) -> tuple[float, float, float, float]:
        if self.image is None:
            return box.center_x, box.center_y, box.width, box.height
        image_width, image_height = self.image.size
        x1 = min(max((box.center_x - box.width / 2) * image_width, 0.0), image_width)
        y1 = min(max((box.center_y - box.height / 2) * image_height, 0.0), image_height)
        x2 = min(max((box.center_x + box.width / 2) * image_width, 0.0), image_width)
        y2 = min(max((box.center_y + box.height / 2) * image_height, 0.0), image_height)
        return (
            ((x1 + x2) / 2) / image_width,
            ((y1 + y2) / 2) / image_height,
            (x2 - x1) / image_width,
            (y2 - y1) / image_height,
        )

    def confirm_leave(self) -> bool:
        if not self.is_dirty:
            return True
        if self.auto_save_on_navigation.get():
            return self.save_current()
        answer = messagebox.askyesnocancel(
            "标注尚未保存",
            "当前图片的标注有修改。\n\n选择“是”保存后继续；“否”放弃修改；“取消”留在当前图片。",
            parent=self.root,
        )
        if answer is None:
            return False
        if answer:
            return self.save_current()
        return True

    def remember_navigation_view(self) -> None:
        """Store zoom plus normalized viewport center for the next image, if enabled."""
        if not self.keep_view_on_navigation.get() or self.image is None or self.scale <= 0:
            self.navigation_view = None
            return
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        image_width, image_height = self.image.size
        center_x = min(max((canvas_width / 2 - self.offset_x) / self.scale, 0.0), image_width)
        center_y = min(max((canvas_height / 2 - self.offset_y) / self.scale, 0.0), image_height)
        self.navigation_view = (
            self.zoom_factor,
            center_x / image_width,
            center_y / image_height,
        )

    def previous(self) -> None:
        if not self.confirm_leave():
            return
        self.remember_navigation_view()
        self.index = (self.index - 1) % len(self.image_paths)
        self.load_current_image()

    def next(self) -> None:
        if not self.confirm_leave():
            return
        self.remember_navigation_view()
        self.index = (self.index + 1) % len(self.image_paths)
        self.load_current_image()

    def jump(self, _event: Optional[tk.Event] = None) -> None:
        try:
            requested = int(self.jump_var.get())
        except ValueError:
            self.hint_var.set("请输入图片序号")
            return
        if not 1 <= requested <= len(self.image_paths):
            self.hint_var.set(f"序号范围是 1 到 {len(self.image_paths)}")
            return
        if requested == self.index + 1 or not self.confirm_leave():
            return
        self.remember_navigation_view()
        self.index = requested - 1
        self.load_current_image()

    def on_canvas_resize(self, event: tk.Event) -> None:
        size = (event.width, event.height)
        if size != self.last_canvas_size:
            old_width, old_height = self.last_canvas_size
            if self.image is not None and self.view_initialized and old_width > 0 and old_height > 0:
                image_width, image_height = self.image.size
                center_x = min(max((old_width / 2 - self.offset_x) / self.scale, 0.0), image_width)
                center_y = min(max((old_height / 2 - self.offset_y) / self.scale, 0.0), image_height)
                self.pending_view_center = (center_x, center_y)
            self.last_canvas_size = size
            self.render()

    def on_mouse_wheel(self, event: tk.Event) -> Optional[str]:
        # On Windows, Control is bit 0x0004 in Tk's event state mask.
        if not (event.state & 0x0004):
            return None
        if event.delta == 0:
            return "break"
        factor = ZOOM_STEP if event.delta > 0 else 1 / ZOOM_STEP
        self.zoom_at(event.x, event.y, factor)
        return "break"

    def zoom_at(self, canvas_x: float, canvas_y: float, factor: float) -> None:
        if self.image is None:
            return
        image_width, image_height = self.image.size
        anchor_x = min(max((canvas_x - self.offset_x) / self.scale, 0.0), image_width)
        anchor_y = min(max((canvas_y - self.offset_y) / self.scale, 0.0), image_height)
        self.zoom_factor = min(max(self.zoom_factor * factor, MIN_ZOOM), MAX_ZOOM)
        self.scale = self.fit_scale * self.zoom_factor
        self.display_width = max(1, round(image_width * self.scale))
        self.display_height = max(1, round(image_height * self.scale))
        self.offset_x = canvas_x - anchor_x * self.scale
        self.offset_y = canvas_y - anchor_y * self.scale
        self.view_initialized = True
        self.clamp_view(max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height()))
        self.render()

    def on_pan_start(self, event: tk.Event) -> None:
        self.pan_start = (event.x, event.y, self.offset_x, self.offset_y)
        self.canvas.configure(cursor="fleur")

    def on_pan_drag(self, event: tk.Event) -> None:
        if self.pan_start is None:
            return
        start_x, start_y, offset_x, offset_y = self.pan_start
        self.offset_x = offset_x + event.x - start_x
        self.offset_y = offset_y + event.y - start_y
        self.clamp_view(max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height()))
        self.render()

    def on_pan_end(self, _event: tk.Event) -> None:
        self.pan_start = None
        self.canvas.configure(cursor="crosshair")

    def reset_view(self) -> None:
        self.zoom_factor = 1.0
        self.view_initialized = False
        self.pending_view_center = None
        self.render()

    def on_reset_view_key(self, _event: tk.Event) -> str:
        self.reset_view()
        return "break"

    def on_previous_key(self, event: tk.Event) -> Optional[str]:
        if isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Combobox)):
            return None
        self.previous()
        return "break"

    def on_next_key(self, event: tk.Event) -> Optional[str]:
        if isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Combobox)):
            return None
        self.next()
        return "break"

    def on_delete_key(self, _event: tk.Event) -> str:
        self.delete_selected()
        return "break"

    def on_save_key(self, _event: tk.Event) -> str:
        self.save_current()
        return "break"

    def on_undo_key(self, _event: tk.Event) -> str:
        self.undo()
        return "break"

    def on_toggle_key(self, _event: tk.Event) -> str:
        self.show_boxes.set(not self.show_boxes.get())
        self.render()
        return "break"

    def on_close(self) -> None:
        if self.confirm_leave():
            self.root.destroy()


def choose_path(
    picker_root: tk.Tk,
    configured_path: Optional[Path],
    kind: str,
    title: str,
    initial_dir: Path,
) -> Optional[Path]:
    candidate = resolve_path(configured_path)
    if candidate is not None:
        exists = candidate.is_dir() if kind == "directory" else candidate.is_file()
        if exists:
            return candidate

    start_dir = initial_dir if initial_dir.is_dir() else ROOT
    if kind == "directory":
        selected = filedialog.askdirectory(
            parent=picker_root,
            title=title,
            initialdir=str(start_dir),
            mustexist=True,
        )
    else:
        selected = filedialog.askopenfilename(
            parent=picker_root,
            title=title,
            initialdir=str(start_dir),
            filetypes=[("类别文本文件", "*.txt"), ("所有文件", "*.*")],
        )
    return Path(selected).resolve() if selected else None


def choose_input_paths(args: argparse.Namespace) -> Optional[tuple[Path, Path, Path]]:
    configured = (
        args.images if args.images is not None else IMAGES_DIR,
        args.labels if args.labels is not None else LABELS_DIR,
        args.classes if args.classes is not None else CLASSES_FILE,
    )
    picker_root = tk.Tk()
    picker_root.withdraw()
    images_dir = choose_path(picker_root, configured[0], "directory", "选择图片文件夹", ROOT)
    if images_dir is None:
        picker_root.destroy()
        return None
    labels_dir = choose_path(picker_root, configured[1], "directory", "选择标签文件夹", images_dir.parent)
    if labels_dir is None:
        picker_root.destroy()
        return None
    classes_file = choose_path(picker_root, configured[2], "file", "选择类别 txt 文件", labels_dir.parent)
    picker_root.destroy()
    if classes_file is None:
        return None
    return images_dir, labels_dir, classes_file


def main() -> None:
    parser = argparse.ArgumentParser(description="查看、绘制和修改 YOLO 矩形框标注")
    parser.add_argument("--images", type=Path, help="图片文件夹路径")
    parser.add_argument("--labels", type=Path, help="YOLO 标签文件夹路径")
    parser.add_argument("--classes", type=Path, help="类别 txt 文件路径，每行一个类别名")
    args = parser.parse_args()

    selected = choose_input_paths(args)
    if selected is None:
        return
    images_dir, labels_dir, classes_file = selected
    if not images_dir.is_dir() or not labels_dir.is_dir() or not classes_file.is_file():
        messagebox.showerror("路径无效", "请分别指定存在的图片文件夹、标签文件夹和类别 txt 文件。")
        return

    root = tk.Tk()
    try:
        LabelViewer(root, images_dir, labels_dir, classes_file)
    except (OSError, ValueError) as error:
        messagebox.showerror("无法打开数据", str(error), parent=root)
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
