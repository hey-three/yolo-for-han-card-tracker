# 斗地主牌面识别

基于 Ultralytics YOLO 的扑克牌牌面目标检测项目，主要为 [Han Card Tracker 斗地主记牌器](https://github.com/hey-three/han-card-tracker) 训练模型。记牌器使用模型实时检测游戏窗口中的牌面及位置，再结合游戏区域统计剩余牌并记录出牌历史。

本仓库负责数据准备、标注检查、模型训练和预测评估；记牌器的桌面界面、窗口识别和记牌逻辑在另一个仓库中。模型按点数区分、不区分花色，共 15 类。训练入口为 `scripts/train.py`，模型和输入尺寸可在脚本顶部设置。模型接入记牌器时，请参考其[模型放置与尺寸规范](https://github.com/hey-three/han-card-tracker/blob/main/docs/模型训练与拼图约定.md)，并在目标游戏画面中实际验证识别效果。

记牌器项目地址：<https://github.com/hey-three/han-card-tracker>

## 下载数据和权重

- [训练数据集 Release](https://github.com/hey-three/yolo-for-han-card-tracker/releases/tag/training-dataset-2026-10-29)：下载 `data.rar`，解压后可查看 `data/` 下的原始图片和标签。该压缩包不是已经划分好的 `dataset/`，训练前需要自行准备训练集和验证集。
- [预训练模型权重 Release](https://github.com/hey-three/yolo-for-han-card-tracker/releases/tag/pretrained-weights-2026-10-30)：下载 `weights.rar`，其中包含作者训练的 `yolov11n768`、`yolov11n960`、`yolov11s768`、`yolov11s960` 四组权重和对应的 `model_config.yaml`。

下载后解压到合适的位置，并在训练或预测脚本顶部填写实际的 `.pt` 文件路径。脚本里当前使用的 `runs/detect/...` 路径属于本机训练结果，克隆仓库后需要自行修改。

## 项目结构

~~~text
yolo-for-han-card-tracker/
├─ README.md                         项目说明、数据准备和运行方法
├─ AGENTS.md                         项目维护约定
├─ .gitignore                        忽略本地数据、训练输出和 IDE 文件
├─ requirements.txt                  Python 和 CUDA 依赖列表
├─ data.yaml                         YOLO 数据集配置
├─ data/                             收集或下载的图片和标签，已同步到仓库
│  ├─ jj_and_other_data/             JJ 斗地主等图片及标签
│  ├─ huanleddz/                     欢乐斗地主图片及标签
│  ├─ external_data/                 外部数据
│  └─ poker/                         从训练数据中排除的图片及标签
├─ dataset/                          本地数据目录，不上传到 GitHub
│  ├─ images/{train,val}/            训练和验证图片
│  ├─ labels/{train,val}/            YOLO 格式标签
│  └─ classes.txt                    类别列表
├─ model_test/
│  ├─ images/                        来自 jj_and_other_data 的测试图片
│  ├─ labels/                        对应的 YOLO 标签
│  ├─ classes.txt                    类别列表
│  ├─ predictions/                   预测可视化结果，本地生成且不上传
│  ├─ predict_folder.py               将图片文件夹中的检测结果画到原图上
│  └─ predict_images.py              批量预测并对照标签生成报告
├─ scripts/
│  ├─ train.py                       YOLO 模型训练
│  ├─ predict_one.py                 单张图片预测
│  ├─ prelabel.py                    图片或目录的批量预标注
│  ├─ benchmark_cpu.py               CPU 单图推理测速
│  └─ data_tools/                    标注检查、标签调整、裁剪和类别转换工具
├─ weights/pretrained/               项目使用的预训练权重
├─ runs/detect/                      训练和预测结果，本地生成且不上传
├─ docs/development/                 Python 开发规范
└─ yolo26n.pt                        根目录保留的模型权重文件
~~~

## 环境准备

当前 `requirements.txt` 对齐项目的 Windows、CUDA 12.1 训练环境，包含 PyTorch 2.5.1、TorchVision 0.20.1、Ultralytics 8.4.6，以及数据工具所需的 OpenCV 和 Pillow。激活 Python 环境后安装：

~~~powershell
conda activate yolov11
python -m pip install -r requirements.txt
~~~

如果使用 CPU 或其他 CUDA 版本，请先按本机环境调整 `requirements.txt` 中 PyTorch 的下载源和版本。Windows 下训练脚本使用 `workers=0`。

`dataset/` 训练数据不随仓库上传。克隆项目后，请在根目录准备以下结构，并确认标签与图片文件名对应：

~~~text
dataset/
├─ images/
│  ├─ train/
│  └─ val/
├─ labels/
│  ├─ train/
│  └─ val/
└─ classes.txt
~~~

`data.yaml` 将数据根目录设为 `dataset`，训练与验证路径分别为 `images/train`、`images/val`。类别顺序应与 `dataset/classes.txt` 一致。

## 训练模型

在项目根目录打开终端，激活环境后运行：

~~~powershell
conda activate yolov11
python .\scripts\train.py
~~~

当前配置：

| 项目 | 设置 |
|---|---|
| 起始权重 | 当前 `LOAD_WEIGHTS=True`，加载 `runs/detect/train4/weights/best.pt` |
| 随机初始化 | `LOAD_WEIGHTS=False` 时使用 `yolo11n.yaml`，不加载权重 |
| 输入尺寸 | 宽 960 × 高 608，以脚本顶部设置为准 |
| 训练轮数 | 最多 100，可能因提前停止而减少 |
| Batch | 16 |
| 设备 | GPU 0 |
| Windows 数据加载 | `workers=0` |
| 结果图 | 已启用 `plots=True` |

在 `scripts/train.py` 顶部设置 `LOAD_WEIGHTS`：`True` 加载 `MODEL_PATH` 指定的预训练权重或此前训练的 `.pt` 权重；`False` 不加载权重，按 `MODEL_CONFIG` 指定的模型结构随机初始化。加载旧权重会创建新的训练运行，轮数、优化器和学习率重新开始，不是从中断处恢复训练。

当前 `MODEL_PATH` 指向的是本机 `runs/` 下的权重，该目录不会随仓库上传。首次使用时请将它改为实际存在的权重路径；`LOAD_WEIGHTS=True` 时模型结构由所选权重决定，`MODEL_CONFIG` 仅用于 `LOAD_WEIGHTS=False`。

增强配置采用轻微色调、饱和度和亮度变化；旋转、平移、缩放、倾斜、透视、翻转、Mosaic、MixUp、CutMix 和 BGR 通道交换均关闭。实际输入画布由脚本顶部的 `IMAGE_WIDTH` 和 `IMAGE_HEIGHT` 固定，宽高应为 YOLO11 的 32 倍数步长整除。每张图片会按原比例缩放并补边到画布，不会被拉伸。训练曲线、验证图和权重保存在 `runs/detect/train/`（如目录已存在，Ultralytics 会自动编号）。

训练准备阶段，脚本会在该次运行的 `weights/` 文件夹中生成 `model_config.yaml`，记录输入宽高和预测补边方式；训练正常结束后会更新实际保存的权重列表。Ultralytics 自带的 `args.yaml` 仍只记录单值 `imgsz`。当前 `model_test/predict_images.py` 仍使用脚本顶部设置的宽高，尚不会自动读取此配置文件。

每次训练会在 `runs/detect/` 下建立单独实验目录，例如 `train`、`train2`。每个目录保存模型权重、参数和训练记录。正常完成训练后，Ultralytics 会保存 `results.csv`、`results.png`，以及混淆矩阵、PR 等验证图。CSV 按轮记录训练/验证 loss、Precision、Recall、mAP 和学习率。目标检测通常使用这些指标评估，不以单一分类准确率表示。

`runs/` 是可重新生成的训练输出，已加入 Git 忽略规则；训练结果只保存在运行电脑上，需自行备份重要的 `best.pt`。

## 预测

### 仅生成带检测框的图片

在 `model_test/predict_folder.py` 顶部设置 `MODEL_DIR`（包含 `best.pt` 的模型目录，也可直接填写 `.pt` 文件）和 `IMAGE_DIR`（原始图片文件夹），然后在项目根目录运行：

~~~powershell
python .\model_test\predict_folder.py
~~~

脚本优先读取权重同目录下 `model_config.yaml` 中的输入宽高；如果没有配置文件，使用脚本顶部的 `IMAGE_WIDTH` 和 `IMAGE_HEIGHT`。`CONFIDENCE` 可调整置信度阈值。结果按原图尺寸保存到 `model_test/predictions/folder/` 下新建的运行目录，不需要标签或类别文件。

### 批量预测并检查标注

`model_test/` 当前使用从 `data/jj_and_other_data` 复制的 208 张图片和同名标签。将其他数据用于检查时，在 `model_test/images/` 放入图片，在 `model_test/labels/` 放入与图片同名的 YOLO 检测标签，并在 `model_test/classes.txt` 中按标签类别 ID 顺序填写类别名。没有目标的图片也应有一个空标签文件；缺失标签会停止检查，避免误判。

在 `model_test/predict_images.py` 顶部修改 `MODEL_PATH` 为要测试的权重，使用 `IMAGE_WIDTH` 和 `IMAGE_HEIGHT` 设置预测画布的宽和高（默认 960×608），用 `CONFIDENCE` 和 `MATCH_IOU` 分别设置预测置信度和框匹配阈值，然后运行：

当前 `MODEL_PATH` 指向本机的 `runs/detect/train15/weights/best.pt`，克隆项目后请改成下载或训练得到的实际权重路径。`CONFIDENCE=0.6` 表示低于 0.6 的预测不参与检查；`MATCH_IOU=0.5` 用于判断预测框与标注框的位置是否足够接近。

图片会按原比例缩放并补边到设置的画布，不会拉伸。宽、高建议使用 32 的倍数，否则 Ultralytics 会向上调整尺寸。

~~~powershell
python .\model_test\predict_images.py
~~~

结果保存在 `model_test/predictions/` 下新建的运行目录：`images/` 是所有预测图片，`errors/` 是有问题的图片，`report.txt` 是汇总及逐张问题说明，`errors.csv`、`image_summary.csv`、`class_summary.csv` 可用 Excel 查看明细。脚本按框的位置一一匹配标注和预测，再区分正确、分类错误、漏检和多检；低于置信度阈值的预测不会参与比较。错误图中绿色为真实框、红色为分类错误的预测框、橙色为漏检、蓝色为多检。报告统计查准率和查全率，不是 mAP；结果需要结合图片和标签人工核对。类别名称须与模型一致，且区分大小写。结果属于本地生成文件，不上传到 GitHub。

### 单张图片预测

将模型路径和图片路径替换为实际文件：

~~~powershell
python .\scripts\predict_one.py --model runs/detect/train/weights/best.pt --image model_test/images/pic_000.png
~~~

### 批量预标注

~~~powershell
python .\scripts\prelabel.py --model runs/detect/train/weights/best.pt --source "图片文件夹路径"
~~~

预标注结果写入 `runs/detect/prelabel/`。生成的标签是草稿，使用前需要人工检查。

### 逐张检查图片标注

标注检查器分别读取图片文件夹、标签文件夹和类别 txt。类别 txt 每行写一个类别名，行号从 0 开始对应 YOLO 标签里的类别 ID；框上显示类别名称，而不是 `class 0` 这样的编号。

~~~powershell
python .\scripts\data_tools\review_labels.py
~~~

不填写路径时，运行后会依次选择图片文件夹、标签文件夹和类别 txt。也可以直接在 `review_labels.py` 顶部填写 `IMAGES_DIR`、`LABELS_DIR`、`CLASSES_FILE`，或者用命令行指定：

~~~powershell
python .\scripts\data_tools\review_labels.py --images "E:\path\to\images" --labels "E:\path\to\labels" --classes "E:\path\to\classes.txt"
~~~

例如检查刚转换的数据集训练划分：

~~~powershell
python .\scripts\data_tools\review_labels.py --images .\tmp\images\train --labels .\tmp\labels\train --classes .\tmp\classes.txt
~~~

窗口支持查看和编辑标注：先选择类别，在空白位置按住鼠标左键拖动可画框；选中已有框后可拖动或调整大小，按 Delete 删除。按住 Ctrl 滚动鼠标滚轮可围绕鼠标位置缩放，中键拖动可平移，按 0 恢复适应窗口。左右方向键翻图，`B` 显示或隐藏标注框，Ctrl+S 保存，Ctrl+Z 撤销。

“设置”中可以开关切换图片时自动保存，以及切图时沿用缩放比例和视口位置；两项默认开启，仅在本次运行期间生效。图片与标签按相同相对路径和文件名对应，支持子目录，保存只修改所选标签目录。

## 数据和验证说明

当前训练目录有 342 张图片，验证目录有 137 张图片；验证目录的图片也都存在于训练目录中。`data/poker` 中的 52 张图片已从主数据集中移除。此划分下的验证分数不能作为未参与训练图片的独立评估成绩。

`model_test/images/` 目前包含来自 `data/jj_and_other_data` 的 208 张图片，配有对应标签。这些图片全部存在于当前训练集中，其中 83 张也存在于验证集中，因此用它们检查模型时得到的结果不是独立测试成绩。

## 类别

类别 ID 顺序由 `data.yaml` 和 `dataset/classes.txt` 定义：

~~~text
0 two, 1 three, 2 four, 3 five, 4 six, 5 seven, 6 eight, 7 nine,
8 ten, 9 J, 10 Q, 11 K, 12 A, 13 joker, 14 JOKER
~~~

修改类别时要同步更新两个文件以及对应标签。辅助类别转换工具可能使用不同的大小王命名，不要未经确认就把转换后的标签混入主数据集。

## GitHub 文件范围

仓库忽略本地的 `dataset/`、`data.rar`、`runs/`、`.idea/` 和 `model_test/predictions/`。`data/`、`model_test/images/` 和 `model_test/labels/` 已同步到仓库；数据压缩包和作者训练的权重另见上方 Release。

