# 项目维护约定

## 项目概述

这是一个基于 Ultralytics YOLO 的斗地主牌面目标检测项目。主数据集使用 15 个不区分花色的点数类别。保持现有独立脚本结构，不引入不必要的应用层或包结构。

## 目录与数据

- `data.yaml`：数据集配置，根目录为 `dataset/`。
- `dataset/`：本地训练数据，图片、标签和类别表；GitHub 忽略该目录，不要将数据文件加入提交。
- `model_test/`：20 张示例图、手动指定模型的批量预测脚本；预测输出位于被忽略的 `predictions/`。
- `scripts/`：训练、预测、预标注和数据处理入口。
- `weights/pretrained/`：训练入口使用的预训练权重。
- `runs/`：Ultralytics 训练、预测输出；属于本地生成文件，不上传。
- `docs/development/`：通用 Python 开发规范。

## 类别与数据划分

主类别按以下顺序定义：two、three、four、five、six、seven、eight、nine、ten、J、Q、K、A、joker、JOKER。修改类别时同步维护 `data.yaml`、`dataset/classes.txt` 和标签中的类别 ID。

当前 train 和 val 各有 260 张图片，图片及标签逐项重复。训练使用 train 更新参数，val 用于每轮评估和保存 best.pt；因此 val 分数不是独立泛化成绩。不要把示例图片或验证结果描述为独立测试集结果。

## 训练和结果

- YOLO11n、416 输入：`python scripts/train_416.py`，batch 32、GPU 0、100 轮、workers 0。
- 旧的 960 输入训练：`python scripts/train.py`。
- Windows 下保留 `workers=0`。
- 416 训练启用 `plots=True`。运行目录下会保存每轮指标 CSV 和训练/验证图；只有正常完成训练才会生成完整最终图。
- 所有训练结果保存在 `runs/detect/`。该目录被 Git 忽略，不要提交模型检查点或运行产物，除非用户明确要求。

## 路径和脚本修改

- 使用 `pathlib.Path` 以脚本位置定位项目根目录，避免依赖 IDE 当前工作目录。
- 训练和预测命令示例以项目根目录为起点。
- 预标注标签需要人工检查后再用于训练。
- 辅助标签转换可能使用不同的大小王名称；不要未经明确转换就与主数据集混用。
- 不删除数据、权重或历史运行记录，除非用户明确要求。
