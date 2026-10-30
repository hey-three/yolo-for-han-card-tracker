# 项目维护约定

## 项目概述

这是一个基于 Ultralytics YOLO 的斗地主牌面目标检测项目。主数据集使用 15 个不区分花色的点数类别。保持现有独立脚本结构，不引入不必要的应用层或包结构。

## 目录与数据

- `data.yaml`：数据集配置，根目录为 `dataset/`。
- `dataset/`：本地训练数据，图片、标签和类别表；GitHub 忽略该目录，不要将数据文件加入提交。
- `model_test/`：来自 `jj_and_other_data` 的 208 张测试图片、对应标签和类别表，以及手动指定模型的批量预测脚本；预测输出位于被忽略的 `predictions/`。这些图片同时存在于当前训练集中，测试结果不是独立测试成绩。
- `scripts/`：训练、预测、预标注和数据处理入口。
- `weights/pretrained/`：训练入口使用的预训练权重。
- `runs/`：Ultralytics 训练、预测输出；属于本地生成文件，不上传。
- `docs/development/`：通用 Python 开发规范。

## 类别与数据划分

主类别按以下顺序定义：two、three、four、five、six、seven、eight、nine、ten、J、Q、K、A、joker、JOKER。修改类别时同步维护 `data.yaml`、`dataset/classes.txt` 和标签中的类别 ID。

当前 train 有 342 张图片、val 有 137 张图片；val 图片及标签也存在于 train 中。训练使用 train 更新参数，val 用于每轮评估和保存 best.pt；因此 val 分数不是独立测试成绩。不要把示例图片或验证结果描述为独立测试集结果。

## 训练和结果

- 唯一训练入口：YOLO11s、固定宽 960 高 608 输入，运行 `python scripts/train.py`；宽高均为 32 的倍数，固定矩形 batch shape，batch 32、GPU 0、200 轮、workers 0，启用 plots。
- Windows 下保留 `workers=0`。
- 训练启用 `plots=True`。运行目录下会保存每轮指标 CSV 和训练/验证图；只有正常完成训练才会生成完整最终图。
- 所有训练结果保存在 `runs/detect/`。该目录被 Git 忽略，不要提交模型检查点或运行产物，除非用户明确要求。

## 路径和脚本修改

- 使用 `pathlib.Path` 以脚本位置定位项目根目录，避免依赖 IDE 当前工作目录。
- 训练和预测命令示例以项目根目录为起点。
- 预标注标签需要人工检查后再用于训练。
- 辅助标签转换可能使用不同的大小王名称；不要未经明确转换就与主数据集混用。
- 不删除数据、权重或历史运行记录，除非用户明确要求。
