import os
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk


class MyDataset(Dataset):
    """
        |-- data
        |   |-- image
        |   |   |-- patient1.nii
        |   |   |-- patient2.nii
        |   |   |-- ...
        |   |
        |   |-- mask
        |       |-- patient1.nii
        |       |-- patient2.nii
        |       |-- ...
    """
    def __init__(self, data_path, joint_transform=None, cal_dist=False):
        self.cal_dist = cal_dist
        self.joint_transform = joint_transform
        self.data_path = data_path
        self.image_name = [self.data_path + x for x in os.listdir(self.data_path)]
        self.image_name.sort()
        self.mask_path = self.data_path.replace('image', 'mask')
        self.mask_name = [self.mask_path + x for x in os.listdir(self.mask_path)]
        self.mask_name.sort()
        if len(self.image_name) != len(self.mask_name):
            raise ValueError("Number of data and mask unmatched.")

    def __len__(self):
        return len(self.image_name)

    @staticmethod
    def transform_ctdata(image, windowWidth, windowCenter, normal=False):
        """
            return: trucated image according to window center and window width
        """
        minWindow = float(windowCenter) - 0.5 * float(windowWidth)
        img = (image - minWindow) / float(windowWidth)
        if normal:
            newimg = (img - np.min(img)) / (np.max(img) - np.min(img))
            return newimg
        else:
            return img

    def __getitem__(self, index):
        image = sitk.GetArrayFromImage(sitk.ReadImage(self.image_name[index]))
        label = sitk.GetArrayFromImage(sitk.ReadImage(self.mask_name[index]))

        image = self.transform_ctdata(image, 300, 400, True)[np.newaxis, :, :, :]
        label = label.astype(np.int16)
        if self.joint_transform:
            if self.cal_dist:
                dist_map_onehot = np.load(self.mask_name[index].replace('.nii.gz', '.npy').replace('mask', 'dist'))
                image = np.concatenate((image, dist_map_onehot), axis=0)
            return self.joint_transform(image, label)
        else:
            return image, label


class MyTestDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_name = [self.data_path + x for x in os.listdir(self.data_path)]
        self.data_name.sort()

    def __len__(self):
        return len(self.data_name)

    @ staticmethod
    def transform_ctdata(image, windowWidth, windowCenter, normal=False):
        """
            return: trucated image according to window center and window width
        """
        minWindow = float(windowCenter) - 0.5 * float(windowWidth)
        img = (image - minWindow) / float(windowWidth)
        if normal:
            newimg = (img - np.min(img)) / (np.max(img) - np.min(img))
            return newimg
        else:
            return img

    def __getitem__(self, index):
        image = sitk.ReadImage(self.data_name[index])
        image_arr = sitk.GetArrayFromImage(image)
        image_arr = self.transform_ctdata(image_arr, 300, 400, True)[np.newaxis, :, :, :]
        image_meta = {
            'Data': image_arr,
            'Size': image.GetSize(),
            'Spacing': image.GetSpacing(),
            'Origin': image.GetOrigin(),
            'Direction': image.GetDirection(),
            'FileName': os.path.basename(self.data_name[index])
        }
        return image_meta
