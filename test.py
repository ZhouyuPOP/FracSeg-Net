import os
import torch
import time
import numpy as np
import SimpleITK as sitk
from pelvic_dataset import MyTestDataset
from torch.utils.data import DataLoader
from utils import decompose_ndimage, compose_ndcube
from train import create_model


def binary(image):
    image[image < 0.5] = 0
    image[image >= 0.5] = 1
    return image


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    test_data_path = 'Path to Your Test Data Directory'
    test_dataset = MyTestDataset(test_data_path)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    num_cls = 2
    net = create_model(num_classes=num_cls)
    model = net.cuda()
    model_path = 'Path to Your Model Directory'
    model.load_state_dict(torch.load(model_path))
    model.eval()

    result_save_path = 'Path to Save Prediction'
    for batch_index, data_batch in enumerate(test_dataloader):
        decompose_start_time = time.time()
        data_cubes_list = decompose_ndimage(data_batch['Data'], [96, 128, 128], min_overlap_rate=0.33)
        num_cubes = len(data_cubes_list)
        decompose_end_time = time.time()
        print(f'Crop Patches Cost:{decompose_end_time - decompose_start_time} seconds.')

        pred_cubes_list = [[] for _ in range(num_cls)]
        segment_start_time = time.time()
        for cube_index in range(num_cubes):
            data_cube_tensor = torch.from_numpy(data_cubes_list[cube_index][np.newaxis]).float().cuda()
            with torch.no_grad():
                pred_cube_tensor = model(data_cube_tensor)                   # [1, num_cls, 96, 128, 128]
            pred_cube = np.squeeze(pred_cube_tensor.detach().cpu().numpy())  # [num_cls, 96, 128, 128]
            for channel_index, pred_channel in enumerate(pred_cube):
                pred_cubes_list[channel_index].append(np.squeeze(pred_channel))
        segment_end_time = time.time()
        print(f'Segment All Patches Cost:{segment_end_time - segment_start_time} seconds.')

        pred_tensors = [compose_ndcube(pred_list, data_batch['Size'], min_overlap_rate=0.33) for pred_list in pred_cubes_list]
        pred_tensor = np.argmax(np.stack(pred_tensors, axis=0), axis=0).astype(np.int8)
        print(f'Composed the Predict Patches Cost:{time.time() - segment_end_time} seconds.')

        pred_tensor = binary(pred_tensor)
        predict_nii = sitk.GetImageFromArray(pred_tensor)
        predict_nii.SetSpacing(data_batch['Spacing'])
        predict_nii.SetOrigin(data_batch['Origin'])
        predict_nii.SetDirection(data_batch['Direction'])
        sitk.WriteImage(predict_nii, os.path.join(result_save_path, data_batch['FileName']))


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    main()


