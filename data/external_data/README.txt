未参与训练, 未处理, 需要处理, 大小joker尺寸问题
外部斗地主数据集转换说明

来源：https://universe.roboflow.com/yuan-t01pu/doudizhu-v2-ujufj/dataset/9
原始发布者：Roboflow 用户；源数据标明许可：CC BY 4.0。原始导出说明见 README.roboflow.txt。

保留原图像和原始标注框坐标，仅转换类别 ID。原 train、valid、test 图片和标签已分别合并到 images、labels，不再划分数据集。
原类别 1=A，2-10 对应点数，11=J，12=Q，13=K。
原类别 0 同时包含大小王：逐个核对牌面字色，黑色为 joker（类别 13），红色为 JOKER（类别 14）。详情见 joker_review.csv。
数据集尚未加入项目主训练集。请使用前检查标注与原数据来源。
