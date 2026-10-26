"""
图片裁剪脚本 - 宽度不变，高度按比例裁剪
适用于YOLO标注的数据集

功能：
1. 裁剪图片（宽度不变，高度按指定比例裁剪）
2. 自动调整YOLO标注文件
3. 支持批量处理

使用方法：
    python crop_images.py --input_dir dataset/images/train --label_dir dataset/labels/train --output_dir dataset/cropped --ratio 0.4
"""

import os
import argparse
from pathlib import Path
from PIL import Image
import shutil


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='裁剪图片并调整YOLO标注')
    
    parser.add_argument('--input_dir', type=str, required=True,
                        help='输入图片目录')
    parser.add_argument('--label_dir', type=str, required=True,
                        help='输入标注目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--ratio', type=float, default=0.4,
                        help='高度裁剪比例 (默认: 0.4 即 2/5)')
    parser.add_argument('--crop_position', type=str, default='top',
                        choices=['top', 'center', 'bottom'],
                        help='裁剪位置: top(从顶部), center(中心), bottom(从底部) (默认: top)')
    parser.add_argument('--image_ext', type=str, default='.png',
                        help='图片扩展名 (默认: .png)')
    
    return parser.parse_args()


def crop_image_and_adjust_labels(input_img_path, input_label_path, 
                                  output_img_path, output_label_path, 
                                  ratio, crop_position='top'):
    """
    裁剪图片并调整YOLO标注
    
    Args:
        input_img_path: 输入图片路径
        input_label_path: 输入标注路径
        output_img_path: 输出图片路径
        output_label_path: 输出标注路径
        ratio: 高度裁剪比例
        crop_position: 裁剪位置 (top/center/bottom)
    """
    
    # 读取图片
    img = Image.open(input_img_path)
    img_width, img_height = img.size
    
    # 计算裁剪区域
    new_height = int(img_height * ratio)
    
    if crop_position == 'top':
        # 从顶部开始裁剪
        crop_box = (0, 0, img_width, new_height)
        y_offset = 0
    elif crop_position == 'center':
        # 从中心裁剪
        y_start = (img_height - new_height) // 2
        crop_box = (0, y_start, img_width, y_start + new_height)
        y_offset = -y_start
    else:  # bottom
        # 从底部开始裁剪
        crop_box = (0, img_height - new_height, img_width, img_height)
        y_offset = -(img_height - new_height)
    
    # 裁剪图片
    cropped_img = img.crop(crop_box)
    cropped_img.save(output_img_path)
    
    # 调整标注
    if os.path.exists(input_label_path):
        adjusted_labels = []
        
        with open(input_label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            class_id = parts[0]
            center_x = float(parts[1])
            center_y = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # 转换为绝对坐标
            abs_center_x = center_x * img_width
            abs_center_y = center_y * img_height
            abs_width = width * img_width
            abs_height = height * img_height
            
            # 计算新的绝对坐标
            if crop_position == 'top':
                new_abs_center_y = abs_center_y
            elif crop_position == 'center':
                new_abs_center_y = abs_center_y - (img_height - new_height) // 2
            else:  # bottom
                new_abs_center_y = abs_center_y - (img_height - new_height)
            
            # 检查目标是否在新图片范围内
            # 目标的边界
            y_min = new_abs_center_y - abs_height / 2
            y_max = new_abs_center_y + abs_height / 2
            
            # 如果目标完全在新图片外，跳过
            if y_max < 0 or y_min > new_height:
                continue
            
            # 裁剪目标（如果部分在图片外）
            y_min = max(0, y_min)
            y_max = min(new_height, y_max)
            new_abs_height = y_max - y_min
            new_abs_center_y = (y_min + y_max) / 2
            
            # 转换回归一化坐标
            new_center_x = abs_center_x / img_width
            new_center_y = new_abs_center_y / new_height
            new_width = abs_width / img_width
            new_height_norm = new_abs_height / new_height
            
            # 保存调整后的标注
            adjusted_labels.append(
                f"{class_id} {new_center_x:.6f} {new_center_y:.6f} {new_width:.6f} {new_height_norm:.6f}\n"
            )
        
        # 保存调整后的标注
        if adjusted_labels:
            with open(output_label_path, 'w') as f:
                f.writelines(adjusted_labels)
        else:
            # 如果没有有效标注，创建空文件
            open(output_label_path, 'w').close()
    else:
        # 如果没有标注文件，创建空文件
        open(output_label_path, 'w').close()
    
    if not os.path.exists(output_label_path):
        return 0
    with open(output_label_path, 'r', encoding='utf-8') as label_file:
        return sum(1 for line in label_file if line.strip())


def process_dataset(input_dir, label_dir, output_dir, ratio, crop_position, image_ext):
    """
    批量处理数据集
    
    Args:
        input_dir: 输入图片目录
        label_dir: 输入标注目录
        output_dir: 输出目录
        ratio: 高度裁剪比例
        crop_position: 裁剪位置
        image_ext: 图片扩展名
    """
    
    # 创建输出目录
    output_img_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels')
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # 获取所有图片文件
    input_path = Path(input_dir)
    image_files = list(input_path.glob(f'*{image_ext}'))
    
    print(f"找到 {len(image_files)} 张图片")
    print(f"裁剪比例: {ratio} (高度变为原来的 {ratio})")
    print(f"裁剪位置: {crop_position}")
    print("-" * 50)
    
    success_count = 0
    skip_count = 0
    
    for i, img_file in enumerate(image_files, 1):
        # 构建路径
        label_file = Path(label_dir) / (img_file.stem + '.txt')
        output_img = Path(output_img_dir) / img_file.name
        output_label = Path(output_label_dir) / (img_file.stem + '.txt')
        
        try:
            # 处理图片和标注
            crop_image_and_adjust_labels(
                str(img_file),
                str(label_file),
                str(output_img),
                str(output_label),
                ratio,
                crop_position
            )
            
            success_count += 1
            if i % 50 == 0:
                print(f"处理进度: {i}/{len(image_files)}")
                
        except Exception as e:
            print(f"处理失败: {img_file.name} - {str(e)}")
            skip_count += 1
    
    print("-" * 50)
    print(f"处理完成!")
    print(f"成功: {success_count} 张")
    print(f"失败: {skip_count} 张")
    print(f"输出目录: {output_dir}")


def main():
    """主函数"""
    args = parse_arguments()
    
    print("=" * 50)
    print("YOLO标注图片裁剪工具")
    print("=" * 50)
    print(f"输入图片目录: {args.input_dir}")
    print(f"输入标注目录: {args.label_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"高度裁剪比例: {args.ratio}")
    print(f"裁剪位置: {args.crop_position}")
    print("=" * 50)
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入图片目录不存在: {args.input_dir}")
        return
    
    if not os.path.exists(args.label_dir):
        print(f"警告: 输入标注目录不存在: {args.label_dir}")
        print("将继续处理，但标注文件可能缺失")
    
    # 处理数据集
    process_dataset(
        args.input_dir,
        args.label_dir,
        args.output_dir,
        args.ratio,
        args.crop_position,
        args.image_ext
    )


if __name__ == '__main__':
    main()

