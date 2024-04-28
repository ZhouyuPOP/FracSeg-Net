import random
import torch
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from typing import List
from scipy.ndimage import distance_transform_edt as distance
from scipy.ndimage.filters import gaussian_filter


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.shape[1:] == mask.shape[0:]

        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomCrop(object):
    def __init__(self, CropSize, dist_flag=False):
        self.CropSize = CropSize
        self.dist_flag = dist_flag

    def __call__(self, volume, seg):
        num_cls = len(np.unique(seg))
        # classify the voxel by class
        propose_center = [np.argwhere(seg == i) for i in range(num_cls)]
        # choose the center point in different probability 'p'
        class_id = np.random.choice(num_cls, size=1, replace=True, p=[0.1, 0.15, 0.3, 0.15, 0.3])
        center = propose_center[int(class_id)][random.randrange(len(propose_center[int(class_id)]))]
        # cal the upper left point
        d1 = center[0] - int(self.CropSize[0] / 2 - 1)
        w1 = center[1] - int(self.CropSize[1] / 2 - 1)
        h1 = center[2] - int(self.CropSize[2] / 2 - 1)
        # the region that can be chosen as the upper left point (prevent from out of range)
        propose_region_d = (int(self.CropSize[0] / 2 - 1), int(seg.shape[0] - self.CropSize[0] / 2 - 1))
        propose_region_w = (int(self.CropSize[1] / 2 - 1), int(seg.shape[1] - self.CropSize[1] / 2 - 1))
        propose_region_h = (int(self.CropSize[2] / 2 - 1), int(seg.shape[2] - self.CropSize[2] / 2 - 1))
        # get the final upper left point
        d1 = np.clip(d1, propose_region_d[0], propose_region_d[1])
        w1 = np.clip(w1, propose_region_w[0], propose_region_w[1])
        h1 = np.clip(h1, propose_region_h[0], propose_region_h[1])
        # crop the dist_map if needed, and then unpack the volume
        if self.dist_flag:
            dist_map_onehot = volume[1:, ...]
            volume = volume[0:1, ...]

        # crop the volume and the segmentation
        seg = seg[d1:d1 + self.CropSize[0], w1:w1 + self.CropSize[1], h1:h1 + self.CropSize[2]]
        volume = volume[:, d1:d1 + self.CropSize[0], w1:w1 + self.CropSize[1], h1:h1 + self.CropSize[2]]

        # pack the cropped dist_map if needed
        if self.dist_flag:
            dist_map_onehot = dist_map_onehot[:, d1:d1 + self.CropSize[0], w1:w1 + self.CropSize[1],
                              h1:h1 + self.CropSize[2]]
            volume_and_dist_map = np.concatenate((volume, dist_map_onehot), axis=0)
            return volume_and_dist_map, seg
        return volume, seg


class Std(object):
    def __init__(self, dist_flag=False):
        super().__init__()
        self.dist_flag = dist_flag

    def __call__(self, volume, seg):
        if self.dist_flag:
            dist = volume[1:, ...]
            volume = volume[0:1, ...]
        volume = volume.astype(np.float64)
        mu = np.mean(volume)
        std = np.std(volume)
        volume = (volume - mu) / (std + 1e-15)
        seg = seg.astype(np.uint8)
        if self.dist_flag:
            volume = np.concatenate((volume, dist), axis=0)
        return volume, seg


class ToTensor(object):
    def __init__(self):
        super().__init__()

    def __call__(self, volume, seg):
        volume = torch.Tensor(volume.copy())
        seg = torch.LongTensor(seg.copy())
        return volume, seg


def mask_to_onehot(mask, palette, axis=0):
    """
    Converts a segmentation mask (C, H, W, D) to (K, H, W, D) where the last dim is a one
    hot encoding vector, C is usually, and K is the number of class.
    """
    semantic_map = []
    for color in palette:
        equality = np.equal(mask, color)
        class_map = np.all(equality, axis=axis)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=axis).astype(np.int8)
    return semantic_map


def one_hot2dist(seg: np.ndarray, C:int=None) -> np.ndarray:
    if C == None:
        C: int = seg.shape[0]
        res = np.zeros_like(seg)
    else:
        res = np.zeros([C, seg.shape[1], seg.shape[2], seg.shape[3]])
    for c in range(C):
        posmask = seg[c].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res


def binary(image):
    image[image < 0.5] = 0
    image[image > 0.5] = 1
    return image


def get_gaussian(size, sigma=1.0/8):
    temp = np.zeros(size)
    coords = [i // 2 for i in size]
    sigmas = [i * sigma for i in size]
    temp[tuple(coords)] = 1
    gaussian_map = gaussian_filter(temp, sigmas, 0, mode='constant', cval=0)
    gaussian_map /= np.max(gaussian_map)
    return gaussian_map


def fit_ndimage_param(image_size, crop_size, min_overlap_rate=0.33):
    """
    Args:
    -----
        image_shape: a tuple of 3d volume (depth, height, width)
                     or 2d image shape (height, width)
        crop_shape: a list or tuple of crop image shape
        min_overlap_rate:
    Returns:
    --------
        fold: patches to crop
        overlap: overlap voxel num between two patches
    """
    assert min_overlap_rate <= 0.5, "overlap can not be bigger than 0.5"
    image_size = np.asarray(image_size)
    crop_size = np.asarray(crop_size)

    min_overlap = min_overlap_rate * crop_size
    dim = image_size - min_overlap
    fold = np.ceil(dim / (crop_size - min_overlap))
    fold = fold.astype('int')
    overlap = np.true_divide((fold * crop_size - image_size), (fold - 1))
    return fold, overlap


def decompose_ndimage(ndimage1, crop_size, min_overlap_rate=0.33):
    """
    decompose ndimage into list of cubes
    Args:
    -----
        ndimage: array, ndimage
                channel is optional, but if data has channel,
                must be sure channel last
                when 3d, size is (depth, height, width, channel)
                when 2d, size is (height, width, channel)
        crop_size: cube size tuple
        min_overlap_rate: float, minmum overlap between two neighbor, cubes, range is [0, 1]
    Returns:
    --------
        ndcubes:
    """
    # get parameters for decompose
    fold, overlap = fit_ndimage_param(ndimage1.shape, crop_size, min_overlap_rate)
    start_point_list = []

    for dim_fold, dim_overlap, dim_len in zip(fold, overlap, crop_size):
        start_point = np.asarray(range(dim_fold)) * (dim_len - dim_overlap)
        start_point = np.floor(start_point)
        start_point = start_point.astype('int')
        start_point_list.append(start_point)

    ndcube_list1 = []
    for i in range(fold[0]):
        for j in range(fold[1]):
            for k in range(fold[2]):
                crop_slice = (slice(start_point_list[0][i], start_point_list[0][i] + crop_size[0]),
                              slice(start_point_list[1][j], start_point_list[1][j] + crop_size[1]),
                              slice(start_point_list[2][k], start_point_list[2][k] + crop_size[2]))
                ndcube_list1.append(ndimage1[crop_slice])
    return ndcube_list1


def compose_ndcube_gaussian(ndcube_list, ndimage_shape, min_overlap_rate=0.33):
    """
    Args:
        ndcube_list: ndarray, cubes deposed from a volume,
            if 3d volume, size is (depth, height, width, channel)
            if 2d image, size is (height, width, channel)
        ndimage_shape: list or tuple, orignal volume shape
            if 3d volume, is (depth, height, width, channel)
            if 2d image, is (height, width, channel)
        min_overlap_rate: float, minmum overlap between two neighbor, cubes, range is [0, 1]
    Returns:
        ndarray, a composed volume
    """
    mask = np.zeros(ndimage_shape)
    ndimage = np.zeros(ndimage_shape)
    crop_size = ndcube_list[0].shape
    gaussian_map = get_gaussian(crop_size)

    fold, overlap = fit_ndimage_param(ndimage_shape, crop_size, min_overlap_rate)
    start_point_list = []

    for dim_fold, dim_overlap, dim_len in zip(fold, overlap, crop_size):
        start_point = np.asarray(range(dim_fold)) * (dim_len - dim_overlap)
        start_point = np.floor(start_point)
        start_point = start_point.astype('int')
        start_point_list.append(start_point)
    cnt = 0
    for i in range(fold[0]):
        for j in range(fold[1]):
            for k in range(fold[2]):
                crop_slice = (slice(start_point_list[0][i], start_point_list[0][i] + crop_size[0]),
                              slice(start_point_list[1][j], start_point_list[1][j] + crop_size[1]),
                              slice(start_point_list[2][k], start_point_list[2][k] + crop_size[2]))
                mask[crop_slice] = mask[crop_slice] + 1.0
                ndimage[crop_slice] = ndimage[crop_slice] + ndcube_list[cnt] * gaussian_map
                cnt = cnt + 1
    ndimage[mask > 0] = ndimage[mask > 0] / mask[mask > 0]
    return ndimage


def compose_ndcube(ndcube_list, ndimage_shape, min_overlap_rate=0.33):
    """
    Args:
        ndcube_list: ndarray, cubes deposed from a volume,
            if 3d volume, size is (depth, height, width, channel)
            if 2d image, size is (height, width, channel)
        ndimage_shape: list or tuple, orignal volume shape
            if 3d volume, is (depth, height, width, channel)
            if 2d image, is (height, width, channel)
        min_overlap_rate: float, minmum overlap between two neighbor, cubes, range is [0, 1]
    Returns:
        ndarray, a composed volume
    """
    mask = np.zeros(ndimage_shape)
    ndimage = np.zeros(ndimage_shape)
    crop_size = ndcube_list[0].shape

    fold, overlap = fit_ndimage_param(ndimage_shape, crop_size, min_overlap_rate)
    start_point_list = []

    for dim_fold, dim_overlap, dim_len in zip(fold, overlap, crop_size):
        start_point = np.asarray(range(dim_fold)) * (dim_len - dim_overlap)
        start_point = np.floor(start_point)
        start_point = start_point.astype('int')
        start_point_list.append(start_point)
    cnt = 0
    for i in range(fold[0]):
        for j in range(fold[1]):
            for k in range(fold[2]):
                crop_slice = (slice(start_point_list[0][i], start_point_list[0][i] + crop_size[0]),
                              slice(start_point_list[1][j], start_point_list[1][j] + crop_size[1]),
                              slice(start_point_list[2][k], start_point_list[2][k] + crop_size[2]))
                mask[crop_slice] = mask[crop_slice] + 1.0
                ndimage[crop_slice] = ndimage[crop_slice] + ndcube_list[cnt]
                cnt = cnt + 1
    ndimage[mask > 0] = ndimage[mask > 0] / mask[mask > 0]

    return ndimage


def cal_dist(mask_name: List[str]):
    for idx, mask_path in enumerate(tqdm(mask_name, desc='Processing')):
        label = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        num_cls = len(np.unique(label))
        seg_onehot = mask_to_onehot(label[np.newaxis, ], np.arange(num_cls), axis=0)
        dist_map_onehot = one_hot2dist(seg_onehot)
        dist_map_onehot = dist_map_onehot / np.max(abs(dist_map_onehot))
        dist_map_onehot = dist_map_onehot.astype(np.float16)
        dist_path = mask_path.replace('.nii.gz', '').replace('mask', 'dist')
        np.save(dist_path, dist_map_onehot)


def dice_ratio_channel_wise(seg, gt, channel):
    """
    define the dice ratio
    :param seg: segmentation result
    :param gt: ground truth
    :param channel: which channel to cal
    :return:
    """
    # select specific channel data
    seg = seg[:, channel, :, :, :]
    gt = gt[:, channel, :, :, :]

    seg = seg.flatten()
    seg[seg > 0.5] = 1
    seg[seg <= 0.5] = 0

    gt = gt.flatten()
    gt[gt > 0.5] = 1
    gt[gt <= 0.5] = 0

    same = (seg * gt).sum()

    dice = (2*float(same) + 0.001)/float(gt.sum() + seg.sum() + 0.001)
    print(f'Dice over Channel{channel} is {dice}')
    return dice
