# 斗地主牌面识别

基于 Ultralytics YOLO 的扑克牌牌面目标检测项目。类别按点数区分，不区分花色，共 15 类。当前推荐的训练入口是 YOLO11n、416 输入的 `scripts/train_416.py`；旧的 960 输入训练入口 `scripts/train.py` 仍保留。

## 项目结构

~~~text
yolo-for-han-card-tracker/
├─ README.md                         项目说明、数据准备和运行方法
├─ AGENTS.md                         项目维护约定
├─ .gitignore                        忽略本地数据、训练输出和 IDE 文件
├─ requirements.txt                  Python 和 CUDA 依赖列表
├─ data.yaml                         YOLO 数据集配置
├─ dataset/                          本地数据目录，不上传到 GitHub
│  ├─ images/{train,val}/            训练和验证图片
│  ├─ labels/{train,val}/            YOLO 格式标签
│  └─ classes.txt                    类别列表
├─ model_test/
│  ├─ images/                        20 张预测示例图
│  ├─ predictions/                   预测可视化结果，本地生成且不上传
│  └─ predict_images.py              修改模型路径后批量预测示例图
├─ scripts/
│  ├─ train_416.py                   YOLO11n、416 输入训练
│  ├─ train.py                       保留的 960 输入训练
│  ├─ predict_one.py                 单张图片预测
│  ├─ prelabel.py                    图片或目录的批量预标注
│  ├─ benchmark_cpu.py               CPU 单图推理测速
│  └─ data_tools/                    标签可视化、调整、裁剪和类别转换工具
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

数据集不随仓库上传。克隆项目后，请在根目录准备以下结构，并确认标签与图片文件名对应：

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

## 416 训练

在项目根目录打开终端，激活环境后运行：

~~~powershell
conda activate yolov11
python .\scripts\train_416.py
~~~

当前配置：

| 项目 | 设置 |
|---|---|
| 模型 | `weights/pretrained/yolo11n.pt` |
| 输入尺寸 | 416 |
| 训练轮数 | 最多 100 |
| Batch | 32 |
| 设备 | GPU 0 |
| Windows 数据加载 | `workers=0` |
| 结果图 | 已启用 `plots=True` |

增强配置采用轻微色调、饱和度和亮度变化；旋转、平移、缩放、倾斜、透视、翻转、Mosaic、MixUp、CutMix 和 BGR 通道交换均关闭。

旧的 960 输入训练脚本可运行：

~~~powershell
python .\scripts\train.py
~~~

每次训练会在 `runs/detect/` 下建立单独实验目录，例如 `train_416`、`train_4162`。每个目录保存模型权重、参数和训练记录。正常完成训练后，Ultralytics 会保存 `results.csv`、`results.png`，以及混淆矩阵、PR 等验证图。CSV 按轮记录训练/验证 loss、Precision、Recall、mAP 和学习率。目标检测通常使用这些指标评估，不以单一分类准确率表示。

`runs/` 是可重新生成的训练输出，已加入 Git 忽略规则；训练结果只保存在运行电脑上，需自行备份重要的 `best.pt`。

## 预测

### 批量预测示例图片

在 `model_test/predict_images.py` 顶部修改 `MODEL_PATH` 为要测试的权重，然后运行：

~~~powershell
python .\model_test\predict_images.py
~~~

预测框图片保存在 `model_test/predictions/`。该目录是本地生成输出，不上传到 GitHub。

### 单张图片预测

将模型路径和图片路径替换为实际文件：

~~~powershell
python .\scripts\predict_one.py --model runs/detect/train_416/weights/best.pt --image model_test/images/pic_000.png
~~~

### 批量预标注

~~~powershell
python .\scripts\prelabel.py --model runs/detect/train_416/weights/best.pt --source "图片文件夹路径"
~~~

预标注结果写入 `runs/detect/prelabel/`。生成的标签是草稿，使用前需要人工检查。

## 数据和验证说明

当前本地数据有 260 张唯一图片；`train` 和 `val` 中的图片与标签逐项相同。训练集用于更新模型，验证集用于训练过程中的评分和保存最佳权重。因此验证分数不是对未见新图片的独立泛化评估。对于自用训练可以按此流程运行，但解读分数时应保留这一前提。

`model_test/images/` 中的示例图来自训练图片，也不是独立测试集。

## 类别

类别 ID 顺序由 `data.yaml` 和 `dataset/classes.txt` 定义：

~~~text
0 two, 1 three, 2 four, 3 five, 4 six, 5 seven, 6 eight, 7 nine,
8 ten, 9 J, 10 Q, 11 K, 12 A, 13 joker, 14 JOKER
~~~

修改类别时要同步更新两个文件以及对应标签。辅助类别转换工具可能使用不同的大小王命名，不要未经确认就把转换后的标签混入主数据集。

## GitHub 文件范围

仓库忽略本地的 `dataset/`、`runs/`、`.idea/` 和 `model_test/predictions/`。这可以避免上传训练数据、训练产物、IDE 设置和生成的预测图。预训练权重和 20 张示例图片保留在仓库中。

