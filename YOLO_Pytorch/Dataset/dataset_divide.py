#### 此部分用于将数据集按照一定的比例划分为训练集、验证集和测试集 ####
import os
import random

def split_dataset(xml_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """按照一定的比例划分数据集
    Args:
        xml_dir: 标签文件路径; 使用标签路径是防止图像无标签也被读取
        train_ratio (float, optional): 训练集占比, 默认 0.8
        val_ratio (float, optional): 验证集占比, 默认 0.1
        test_ratio (float, optional): 测试集占比, 默认 0.1
        seed (int, optional): 随机数种子, 默认42
    """
    # 设置随机种子，确保结果可复现
    random.seed(seed)
    # 获取文件夹中的所有xml文件
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    # 打乱文件顺序
    random.shuffle(xml_files)
    
    # 计算各个子集的数量
    total_files = len(xml_files)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    test_count = total_files - train_count - val_count  # 剩余的用于测试集

    # 划分数据集
    train_files = xml_files[:train_count]
    val_files = xml_files[train_count:train_count + val_count]
    test_files = xml_files[train_count + val_count:]

    # 保存文件名到相应的txt文件中
    with open('.//ImageSets//Main//train.txt', 'w') as train_file:
        for file in train_files:
            train_file.write(f"{file[:-4]}\n")  # 去掉后缀

    with open('.//ImageSets//Main//val.txt', 'w') as val_file:
        for file in val_files:
            val_file.write(f"{file[:-4]}\n")

    with open('.//ImageSets//Main//test.txt', 'w') as test_file:
        for file in test_files:
            test_file.write(f"{file[:-4]}\n")

    print(f"数据集划分完成：训练集 {train_count} 个，验证集 {val_count} 个，测试集 {test_count} 个。")

# 调用函数进行划分，假设xml文件位于 'annotations' 目录下
split_dataset(xml_dir='.//Annotations')