import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import imgaug as ia
import imgaug.augmenters as iaa  # 导入iaa
from torch.utils.data import DataLoader
from torch.nn import functional as F
import cv2


def colour(label, ra, ga, ba):
    size = np.shape(label)
    for i in range(size[0]):
        for j in range(size[1]):
            if label[i][j] == 1.0:
                ra[i][j] = 255
                ga[i][j] = 0
                ba[i][j] = 0
            elif label[i][j] == 2.0:
                ra[i][j] = 0
                ga[i][j] = 255
                ba[i][j] = 0
            elif label[i][j] == 3.0:
                ra[i][j] = 0
                ga[i][j] = 0
                ba[i][j] = 255
            elif label[i][j] == 4.0:
                ra[i][j] = 255
                ga[i][j] = 255
                ba[i][j] = 0
            elif label[i][j] == 5.0:
                ra[i][j] = 255
                ga[i][j] = 0
                ba[i][j] = 255
            elif label[i][j] == 6.0:
                ra[i][j] = 0
                ga[i][j] = 255
                ba[i][j] = 255
            elif label[i][j] == 7.0:
                ra[i][j] = 112
                ga[i][j] = 48
                ba[i][j] = 160
            elif label[i][j] == 8.0:
                ra[i][j] = 255
                ga[i][j] = 192
                ba[i][j] = 0
    return ra, ga, ba
# ----------------------------------------------------------------------------------------------------------
# 将 [4] 转换为 [0,0,0,1,0] 类似的分类
def mask_to_onehot(mask, ):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    mask = np.expand_dims(mask, -1)
    for colour in range(9):
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.int32)
    return semantic_map


# img_aug 为增强序列，让img和seg得到相同的数据增强
def augment_seg(img_aug, img, seg, edge=None):
    seg = mask_to_onehot(seg)
    aug_det = img_aug.to_deterministic()
    image_aug = aug_det.augment_image(img)

    segmap = ia.SegmentationMapOnImage(seg, nb_classes=np.max(seg) + 1, shape=img.shape)
    segmap_aug = aug_det.augment_segmentation_maps(segmap)
    segmap_aug = segmap_aug.get_arr_int()
    segmap_aug = np.argmax(segmap_aug, axis=-1).astype(np.float32)
    if edge is not None:
        edge_ = mask_to_onehot(edge)
        edgemap = ia.SegmentationMapOnImage(edge_, nb_classes=np.max(edge_) + 1, shape=img.shape)
        edgemap_aug = aug_det.augment_segmentation_maps(edgemap)
        edgemap_aug = edgemap_aug.get_arr_int()
        edgemap_aug = np.argmax(edgemap_aug, axis=-1).astype(np.float32)
        return image_aug, segmap_aug, edgemap_aug
    else:
        return image_aug, segmap_aug


# 90度旋转和上下或者左右翻转
def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


# 随机-20度到20度的旋转
def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)  # order 样条插值的阶数
    return image, label


# image order=3 , label order=0 , 貌似没用上这个
class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


# data集名在list内
# sample = {'image': image, 'label': label, 'case_name': case_name}
# ps : 对于test返回的是多张切片图片
class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, img_size, norm_x_transform=None, norm_y_transform=None):
        self.norm_x_transform = norm_x_transform
        self.norm_y_transform = norm_y_transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir
        self.img_size = img_size

        # 每次使用 0~4 个Augmenter来处理图片,每个batch中的Augmenters顺序不一样
        self.img_aug = iaa.SomeOf((0, 4), [
            iaa.Flipud(0.5, name="Flipud"),
            iaa.Fliplr(0.5, name="Fliplr"),
            iaa.AdditiveGaussianNoise(scale=0.005 * 255),
            iaa.GaussianBlur(sigma=(1.0)),
            iaa.LinearContrast((0.5, 1.5), per_channel=0.5),
            iaa.Affine(scale={"x": (0.5, 2), "y": (0.5, 2)}),
            iaa.Affine(rotate=(-40, 40)),
            iaa.Affine(shear=(-16, 16)),
            iaa.PiecewiseAffine(scale=(0.008, 0.03)),
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})
        ], random_order=True)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name + '.npz')
            data = np.load(data_path)
            image, label= data['image'], data['label']
            
            edge = data['edge']
            image, label, edge = augment_seg(self.img_aug, image, label, edge)  # 数据增强
            
            # image, label = augment_seg(self.img_aug, image, label)  # 不做边缘

            # --------------------------------------------------------------------------------------
            # print(label.shape) 512 512  0-8  numpy.ndarray

            x, y = image.shape  # 512 512
            if x != self.img_size or y != self.img_size:
                image = zoom(image, (self.img_size / x, self.img_size / y), order=3)  # why not 3?
                label = zoom(label, (self.img_size / x, self.img_size / y), order=0)  # order是样条插值的阶数  0-5
                edge = zoom(edge, (self.img_size / x, self.img_size / y), order=0)

        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]
            edge = []

        sample = {'image': image, 'label': label,'edge': edge} #,'edge': edge
        if self.norm_x_transform is not None:
            sample['image'] = self.norm_x_transform(sample['image'].copy())  # totensor
        if self.norm_y_transform is not None:
            sample['label'] = self.norm_y_transform(sample['label'].copy())  # totensor
            sample['edge'] = self.norm_y_transform(sample['edge'].copy())  # totensor
        sample['case_name'] = self.sample_list[idx].strip('\n')
        # strip() 方法用于移除字符串头尾指定的字符
        return sample
