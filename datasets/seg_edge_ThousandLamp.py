import os
import random
import torch
import torchvision.transforms as TF
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from utils.misc import get_loc



root='/data/home/xc/iris_data/'

def make_dataset_list(dataset_name, mode, val_percent=0.2):

    assert dataset_name in ['Lamp', 'Thousand']
    assert mode in ['train', 'val', 'test']

    dataset_name = correct_dataset_name(dataset_name)

    data_path = {
        'images_path': os.path.join(root, dataset_name, 'Images'),
        'masks_path': os.path.join(root, dataset_name, 'Masks'),
        'irises_edge_path': os.path.join(root, dataset_name, 'iris_edge'),
        'irises_edge_mask_path': os.path.join(root, dataset_name, 'iris_edge_mask'),
        'pupils_edge_path': os.path.join(root, dataset_name, 'pupil_edge'),
        'pupils_edge_mask_path': os.path.join(root, dataset_name, 'pupil_edge_mask'),
        'heatmaps_path': os.path.join(root, dataset_name, 'seg_edge_image'),
    }

    if mode == 'test':
        test_file = open(os.path.join(root, dataset_name, 'test.txt'))
        test_filenames_list = []
        for line in test_file:
            test_filenames_list.append(line.strip())
        test_file.close()
        return data_path, test_filenames_list

    trainval_file = open(os.path.join(root, dataset_name, 'trainval.txt'))
    trainval_filenames_list = []
    for line in trainval_file:
        trainval_filenames_list.append(line.strip())
    trainval_file.close()
    random.seed(42)
    random.shuffle(trainval_filenames_list)
    train_filenames_list = trainval_filenames_list[:int((1-val_percent)*len(trainval_filenames_list))]
    val_filenames_list = trainval_filenames_list[int((1-val_percent)*len(trainval_filenames_list)):]

    if mode == 'train':
        return data_path, train_filenames_list
    else:
        return data_path, val_filenames_list


def correct_dataset_name(dataset_name):
    if dataset_name == 'Lamp':
        return 'CASIA-iris-Lamp'
    elif dataset_name == 'Thousand':
        return 'CASIA-iris-thousands'
        

class myDataset(Dataset):
    '''
    args:
        dataset_name(str): support for 'CASIA-Iris-Africa','CASIA-distance', 'Occlusion', 'Off_angle', 'CASIA-Iris-Mobile-V1.0'
        mode(str): 'train', 'val', 'test'
        transform(dict): {'train': train_augment, 'test': test_augment}

    return(dict): {
        'image': aug_img,
        'mask': aug_mask,
        'iris_edge': aug_iris_edge
        'iris_edge_mask': aug_iris_edge_mask
        'pupil_edge': aug_pupil_edge
        'pupil_edge_mask': aug_pupil_edge_mask
    }
    '''
    def __init__(self, dataset_name, mode, transform=None, val_percent=0.2):
        self.dataset_name = dataset_name
        self.mode = mode
        self.transform = transform
        self.data_path, self.data_list = make_dataset_list(dataset_name, mode, val_percent=val_percent)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        
        image_name = self.data_list[idx]
        image = Image.open(os.path.join(self.data_path['images_path'], image_name + '.jpg'))
        mask = Image.open(os.path.join(self.data_path['masks_path'], image_name + '.png'))
        iris_edge = Image.open(os.path.join(self.data_path['irises_edge_path'], image_name + '.png'))
        iris_edge_mask = Image.open(os.path.join(self.data_path['irises_edge_mask_path'], image_name + '.png'))
        pupil_edge = Image.open(os.path.join(self.data_path['pupils_edge_path'], image_name + '.png'))
        pupil_edge_mask = Image.open(os.path.join(self.data_path['pupils_edge_mask_path'], image_name + '.png'))
        pupil_edge_mask = Image.open(os.path.join(self.data_path['pupils_edge_mask_path'], image_name + '.png'))
        heatmap = Image.open(os.path.join(self.data_path['heatmaps_path'], image_name + '.png'))

        if self.transform is not None and self.mode != 'test':
            image = np.asarray(image)
            mask = np.asarray(mask)
            iris_edge = np.asarray(iris_edge)
            iris_edge_mask = np.asarray(iris_edge_mask)
            pupil_edge = np.asarray(pupil_edge)
            pupil_edge_mask = np.asarray(pupil_edge_mask)
            heatmap = np.asarray(heatmap)
            # loc = get_loc(iris_edge, pupil_edge)
            mask_list = [mask, iris_edge, iris_edge_mask, pupil_edge, pupil_edge_mask, heatmap]

            aug_data = self.transform(image=image, masks=mask_list)
            aug_image, aug_mask_list = aug_data['image'], aug_data['masks']
            
            image = Image.fromarray(aug_image)
            mask = Image.fromarray(aug_mask_list[0])
            iris_edge = Image.fromarray(aug_mask_list[1])
            iris_edge_mask = Image.fromarray(aug_mask_list[2])
            pupil_edge = Image.fromarray(aug_mask_list[3])
            pupil_edge_mask = Image.fromarray(aug_mask_list[4])
            heatmap = Image.fromarray(aug_mask_list[5])

        elif self.transform is not None and self.mode == 'test':
            image = np.asarray(image)
            heatmap = np.asarray(heatmap)
            aug_data = self.transform(image=image, mask=heatmap)
            aug_image, aug_heatmap = aug_data['image'], aug_data['mask']
            image = Image.fromarray(aug_image)
            heatmap = Image.fromarray(aug_heatmap)


        aug_image = TF.ToTensor()(image)
        aug_image = torch.cat([aug_image, aug_image, aug_image], dim=0)
        aug_mask = TF.ToTensor()(mask)
        aug_iris_edge = TF.ToTensor()(iris_edge)
        aug_iris_edge_mask = TF.ToTensor()(iris_edge_mask)
        aug_pupil_edge = TF.ToTensor()(pupil_edge)
        aug_pupil_edge_mask = TF.ToTensor()(pupil_edge_mask)
        aug_heatmap = TF.ToTensor()(heatmap)

        return {
            'image_name': image_name,
            'image': aug_image,
            'mask': aug_mask,
            'iris_edge': aug_iris_edge,
            'iris_edge_mask': aug_iris_edge_mask,
            'pupil_edge': aug_pupil_edge,
            'pupil_edge_mask': aug_pupil_edge_mask,
            'heatmap': aug_heatmap
        }

