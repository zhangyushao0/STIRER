import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import unicodedata
from torchvision import transforms as T
from torch.utils.data import DataLoader
from utils.augment import rand_augment_transform
from utils.label_converter import get_charset, strLabelConverter, str_filt
import random
import numpy as np

class resizeNormalize(object):
    def __init__(self, size, mask=False, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = T.ToTensor()
        self.mask = mask

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img_tensor = self.toTensor(img)
        if self.mask:
            mask = img.convert("L")
            thres = np.array(mask).mean()
            mask = mask.point(lambda x: 0 if x > thres else 255)
            mask = self.toTensor(mask)
            img_tensor = torch.cat((img_tensor, mask), 0)
        return img_tensor

class TextPairDataset(Dataset):
    def __init__(self, root_dir, max_len=20, train=False, args=None):
        super(TextPairDataset, self).__init__()
        self.root_dir = root_dir
        self.max_len = max_len
        self.train = train
        self.charset = args["charset"]
        
        # 读取标签文件
        self.labels = []
        self.image_names = []
        label_file = os.path.join(root_dir, 'label.txt')
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                # 假设每行格式为: image_name label
                parts = line.strip().split()
                if len(parts) >= 2:
                    image_name = parts[0]
                    label = parts[1]
                    self.image_names.append(image_name)
                    self.labels.append(label)

        print(f"Found {len(self.labels)} samples in {root_dir}")

        # 设置数据增强
        basic_transforms = rand_augment_transform()
        num_da = len(basic_transforms.ops)
        basic_transforms.ops = basic_transforms.ops[: num_da - 3]
        basic_transforms.choice_weights = [1.0 / (num_da - 3)] * (num_da - 3)
        quality_aware_transforms = rand_augment_transform()
        num_da = len(quality_aware_transforms.ops)
        quality_aware_transforms.ops = quality_aware_transforms.ops[num_da - 3 :]
        quality_aware_transforms.num_layers = 2
        quality_aware_transforms.choice_weights = [1.0 / (3)] * 3
        self.basic_transforms = basic_transforms
        self.quality_transforms = quality_aware_transforms
        self.totensor = T.ToTensor()
        self.p_da = 0.5

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        original_index = index
        while True:
            try:
                # 获取图像路径
                image_name = self.image_names[index]
                hr_path = os.path.join(self.root_dir, 'Intact_Image', image_name)
                lr_path = os.path.join(self.root_dir, 'Corrupted_Image', image_name)
                
                # 读取图像
                img_hr = Image.open(hr_path).convert('RGB')
                img_lr = Image.open(lr_path).convert('RGB')
                
                # 获取标签
                label = self.labels[index]
                
                # 过滤标签
                if self.charset == 37:
                    label = str_filt(label, "lower")
                elif self.charset == 95:
                    label = str_filt(label, "all")
                    
                # 检查标签长度
                if len(label) == 0 or (self.train and len(label) > self.max_len):
                    index = (index + 1) % len(self)
                    if index == original_index:  # 避免无限循环
                        raise RuntimeError("No valid sample found in dataset")
                    continue
                    
                # 标准化标签
                label = (
                    unicodedata.normalize("NFKD", label)
                    .encode("ascii", "ignore")
                    .decode()
                    .replace(" ", "")
                )

                # 应用数据增强
                if self.train and random.random() < 0.0:  # 与old版本保持一致，设为0.0
                    img_hr = self.basic_transforms(img_hr)
                    img_lr = self.quality_transforms(img_hr)
                    cls_gt = 0
                else:
                    cls_gt = 1
                
                return img_hr, img_lr, label, cls_gt
                
            except Exception as e:
                print(f"Error loading index {index}: {e}")
                index = (index + 1) % len(self)
                if index == original_index:
                    raise RuntimeError("No valid sample found in dataset")

class LRSTR_collect_fn(object):
    def __init__(
        self,
        imgH=32,
        imgW=128,
        down_sample_scale=2,
        keep_ratio=False,
        min_ratio=1,
        mask=True,
        train=True,
        upsample=False,
        stacklast=False,
        args=None,
    ):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.down_sample_scale = down_sample_scale
        self.mask = mask
        self.train = train
        # use for encode & decode
        self.charset = args["charset"]
        self.label_converter = strLabelConverter(get_charset(args["charset"]))
        self.transform = resizeNormalize((imgW, imgH), self.mask)
        self.transformd2 = resizeNormalize(
            (imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask
        )
        self.bicubic_factor = [2, 3, 4, 5, 6]
        self.upsample = upsample
        self.stacklast = stacklast
        self.args = args

    def __call__(self, batch):
        images_hr, raw_images_lr, label_strs, cls_gt = zip(*batch)
        label_ce = []
        images_hr = [self.transform(image) for image in images_hr]
        images_hr = torch.cat([t.unsqueeze(0) for t in images_hr], 0)

        images_lr = []
        for image in raw_images_lr:
            # if is evaluating, real datas are used, direct eval them without bicubic
            if min(image.size[0], image.size[1]) <= 10 or not self.train:
                image = image.resize((64, 16), Image.BICUBIC)
                images_lr.append(image)
                continue
            if min(image.size[0], image.size[1]) <= 28:
                d_factor = random.choice([1, 2, 3])
                images_lr.append(
                    image.resize(
                        (image.size[0] // d_factor, image.size[1] // d_factor),
                        Image.BICUBIC,
                    )
                )
                continue
            d_factor = random.choice([1, 2, 3, 4])
            images_lr.append(
                image.resize(
                    (image.size[0] // d_factor, image.size[1] // d_factor),
                    Image.BICUBIC,
                )
            )
        if self.upsample:
            images_lr = [self.transform(image) for image in images_lr]
        else:
            images_lr = [self.transformd2(image) for image in images_lr]
        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        label_tensors, length_tensors = self.label_converter.encode(label_strs)
        if self.stacklast:
            images_hr = torch.cat((images_hr, images_lr), 0)
            label_tensors = torch.cat((label_tensors, label_tensors), 0)
            length_tensors = torch.cat((length_tensors, length_tensors), 0)
            label_strs = label_strs + label_strs

        # 30 + 1 plus eos
        label_ce = torch.zeros((images_hr.size(0), 31))
        sum_v = 0
        if self.train:
            for idx, v in enumerate(length_tensors.tolist()):
                try:
                    label_ce[idx, 0:v] = label_tensors[sum_v : (sum_v + v)]
                except:
                    print("v=", v, label_strs[idx])
                    os._exit(233)
                sum_v += v
                # add eos
                label_ce[idx, v] = self.args["charset"]
                for j in range(15 - v):
                    label_ce[idx, v + j] = 0
        label_ce = label_ce.long()
        cls_gt = torch.LongTensor(cls_gt)

        return (
            images_hr,
            images_lr,
            label_tensors,
            length_tensors,
            label_strs,
            label_ce,
        )
