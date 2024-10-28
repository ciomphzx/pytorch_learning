#### 此文件是加载本地VOC格式的数据集 ####
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import xml.etree.ElementTree as ET
from torchvision import transforms

class Dataset_VOC(Dataset):
    def __init__(self, root_dir, image_set='train', transform=None, categroies_file='.//categroies.txt', resize=0):
        """ 类初始化函数

        Args:
            root_dir (_type_): 数据集根目录
            image_set (str, optional): 加载数据集类型, 默认值是'train', 加载训练集.
            transform (_type_, optional): 对图像所做的操作
            categroies_file (str, optional): 数据集类别配置文件路径, 默认'.//categroies.txt'.
            resize (int, optional): 图像缩放后的size, 这里是为了对标签中的坐标进行变换的缩放因子.
        """
        self.root_dir = root_dir
        self.image_set = image_set
        self.transform = transform
        self.categroies_map = self.load_categroies_map(categroies_file)
        self.resize = resize
        
        image_set_path = os.path.join(self.root_dir, 'ImageSets', 'Main', f'{self.image_set}.txt')
        
        with open(image_set_path) as f:
            self.image_ids = f.readlines()
        self.image_ids = [x.strip() for x in self.image_ids]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # 支持的图像格式列表
        supported_formats = ['jpg', 'jpeg', 'png', 'bmp']
        
        img_path = None
        for format in supported_formats:
            possible_path = os.path.join(self.root_dir, 'Images', f'{image_id}.{format}')
            if os.path.exists(possible_path):
                img_path = possible_path
                break
        if img_path is None:
            raise FileNotFoundError(f"No image found for {image_id} in supported formats.")

        xml_path = os.path.join(self.root_dir, 'Annotations', f'{image_id}.xml')

        # 加载图像
        img = Image.open(img_path).convert('RGB')
        # 加载标注信息(解析XML文件), 如果图像resize, 读取的标签也对应进行resize
        if self.resize: # 如果resize, 标签对应reszie
            scaler = [self.resize / img.width, self.resize / img.height]
            boxes, categories = self.parse_voc_xml(xml_path, scaler)
        else:   #没有resize, scaler取默认值[1, 1], 标签不变化
            boxes, categories = self.parse_voc_xml(xml_path)
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = torch.cat((torch.as_tensor(boxes, dtype=torch.float32), 
                        torch.as_tensor(categories, dtype=torch.float32).unsqueeze(1)), dim=1)
        return img, label

    ## 解析VOC格式的XML文件 ##
    def parse_voc_xml(self, xml_path, scaler=[1, 1]):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        categroies = []
        for obj in root.findall('object'):
            categroy = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text) * scaler[0]
            ymin = int(bbox.find('ymin').text) * scaler[1]
            xmax = int(bbox.find('xmax').text) * scaler[0]
            ymax = int(bbox.find('ymax').text) * scaler[1]
            boxes.append([xmin, ymin, xmax, ymax])
            categroies.append(self.categroy_to_int(categroy))
        return boxes, categroies

    def load_categroies_map(self, categroies_file):
        """ 从配置文件中读取类别并构建映射 """
        categroies_map = {}
        with open(categroies_file, 'r') as f:
            for line in f:
                index, categroy = line.strip().split(' ', 1)
                categroies_map[categroy] = int(index)
        return categroies_map

    def categroy_to_int(self, categroy):
        """ 根据类别名称返回类别的整数编码 """
        return self.categroies_map.get(categroy, -1)  # 返回 -1 代表找不到该类别

## 不同图像的目标数量不同, 标签就不同, pytorch无法将不同维度的标签堆叠为一个批次 ##
## 此函数通过自定义的批处理方法, 允许不同样本有不同数量的边界框 ##
def custom_collate_fn(batch):
    """ 允许不同样本有不同边界框
    Args:
        batch: 批次
    Returns:
        images(Tensor): 一个批次的图像数据 
        targets(List): 一个批次的标签
    """
    images = []
    labels = []
    for sample in batch:
        images.append(sample[0])
        labels.append(sample[1])
    # 不使用 torch.stack，因为目标数据可能大小不同
    images = torch.stack(images, dim=0)
    return images, labels

def data_loader_voc(dataset_dir, batch_size, resize=0, image_set='train'):
    """构建数据集加载器
    Args:
        dataset_dir : 数据集目录
        batch_size : 小批量读取的数据量
        resize (int, optional): 图像后缩放的尺寸, 默认值为0, 此时不对图像进行缩放
        image_set (str, optional): 构建加载器的种类, 默认'train'构建训练集的数据加载器
    Returns:
        DataLoader实例: 数据集加载器
    """
    trans = [transforms.ToTensor()]
    if resize:  #这里定义的是对图像进行的缩放操作
        trans.insert(0, transforms.Resize(size=(resize, resize)))
    trans = transforms.Compose(trans)
    # 这里传入resize是为了对应地缩放标签中的边界框坐标
    dataset = Dataset_VOC(root_dir=dataset_dir, image_set=image_set, transform=trans, resize=resize)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    return dataloader

if __name__ == '__main__':
    # 测试VOC格式数据集数据加载
    train_dataloader = data_loader_voc(dataset_dir='./', batch_size=2, resize=256, image_set='train')
    for batch_idx, (imgs, labels) in enumerate(train_dataloader):
        print(f"batch: {batch_idx + 1}, image shape: {imgs.shape}, labels: {labels[0], labels[1]}")
        break