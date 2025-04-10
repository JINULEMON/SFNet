import os
import time
import logging
from datetime import datetime
import random
import argparse
import pandas as pd
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
import albumentations as A
import cv2
from thop import profile
from torchstat import stat

import sys
sys.path.append('/data/home/xc/iris_segmentation_open')
from datasets.seg_edge_ThousandLamp import myDataset
from evaluation import evaluate_circle, evaluate_iris
from utils.misc import check_mkdir, get_circle_edge


from models.SFNet import ssa_PLD

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cudnn.benchmark = True

dataset_name = 'Thousand'
example_name = 'Thousand-SFNet'
pretrain_model_path ="/data/home/xc/iris_segmentation_open/experiments/Thousand-SFNet_20230905_030904_logging/checkpoints/logging.pth"

def get_args():
    parser = argparse.ArgumentParser(description='Test paprmeters', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log', type=str, default='logging.log', dest='log_name')
    parser.add_argument('--ckp', type=str, default=pretrain_model_path, help='Load checkpoints from a xxx/xxx.pth', dest='checkpoints')
    return parser.parse_args()


def main(train_args):
    ############################################# define a CNN #################################################
    net = ssa_PLD(3).cuda()
    net.load_state_dict(torch.load(train_args['checkpoints']), strict=True)

    ########################################### dataset #############################################
    test_augment = A.Compose([
        A.CenterCrop(448, 576),
    ])
    test_dataset = myDataset(dataset_name, mode='test', transform=test_augment)
    test_loader = DataLoader(test_dataset, batch_size=1, drop_last=False)
    logging.info(f'The dataset {dataset_name} is ready!')

    ######################################### test ############################################
    test(test_loader, net)


def test(test_loader, net):
    net.eval()
    print('start test......')

    e1, iou, dice, f1 = 0, 0, 0, 0
    iris_e1, iris_dice, iris_iou = 0, 0, 0
    pupil_e1, pupil_dice, pupil_iou = 0, 0, 0
    names, iris_circles_params, pupil_circles_params = [], [], []

    L = len(test_loader)

    for vi, data in enumerate(test_loader):
        image_name, image, mask, iris_edge, iris_mask, pupil_edge, pupil_mask, loc = \
            data['image_name'][0], data['image'], data['mask'], data['iris_edge'],\
            data['iris_edge_mask'], data['pupil_edge'], data['pupil_edge_mask'], data['heatmap']

        print('testing the {}-th image: {}'.format(vi+1, image_name))
        logging.info('The {}-th image: {}'.format(vi+1, image_name))
        
        N = image.size()[0]
        assert N == 1
        image = image.cuda()
        mask = mask.cuda()
        iris_edge = iris_edge.cuda()
        iris_mask = iris_mask.cuda()
        pupil_edge = pupil_edge.cuda()
        pupil_mask = pupil_mask.cuda()
        loc = loc.cuda()

        _input = torch.cat([image, loc], dim=1)
        
        with torch.no_grad():
            _output = net(_input)

        pred_masks = _output['pred_masks']

        pred_mask = torch.nn.ZeroPad2d((32, 32, 16, 16))(pred_masks[0][:,0:1,:,:])
        pred_iris_mask = torch.nn.ZeroPad2d((32, 32, 16, 16))(pred_masks[0][:,1:2,:,:])
        pred_pupil_mask = torch.nn.ZeroPad2d((32, 32, 16, 16))(pred_masks[0][:,2:3,:,:])

        if image_name == '678L_S5678R06':
            mask = transforms.CenterCrop((480, 640))(mask)


        pred_iris_circle_mask, pred_iris_edge, iris_circles_param = get_circle_edge(pred_iris_mask)
        pred_pupil_circle_mask, pred_pupil_egde, pupil_circles_param = get_circle_edge(pred_pupil_mask)

        ################### val for iris mask ##################

        val_results = evaluate_iris(pred_mask, mask, dataset_name)
        e1 += val_results['E1']/L
        iou += val_results['IoU']/L
        dice += val_results['Dice']/L
        f1 += val_results['F1']/L
        logging.info('mask>   E1: {:.5f},   IoU: {:.5f},   Dice:{:.5f},   F1: {:.5f}'. \
            format(val_results['E1'], val_results['IoU'], val_results['Dice'], val_results['F1']))

        ################### val for iris edge ##################

        iris_val_results = evaluate_circle(pred_iris_circle_mask, iris_mask, pred_iris_edge, iris_edge, dataset_name)
        iris_e1 += iris_val_results['E1']/L
        iris_dice += iris_val_results['Dice']/L
        iris_iou += iris_val_results['IoU']/L
        logging.info('outer>   E1: {:.5f},   IoU: {:.5f},   Dice: {:.5f}'. \
            format(iris_val_results['E1'], iris_val_results['IoU'], iris_val_results['Dice']))

        #################### val for pupil edge ##################


        pupil_val_results = evaluate_circle(pred_pupil_circle_mask, pupil_mask, pred_pupil_egde, pupil_edge, dataset_name)
        pupil_e1 += pupil_val_results['E1']/L
        pupil_dice += pupil_val_results['Dice']/L  
        pupil_iou += pupil_val_results['IoU']/L
        logging.info('inner>   E1: {:.5f},   IoU: {:.5f},   Dice: {:.5f}'. \
            format(pupil_val_results['E1'], pupil_val_results['IoU'], pupil_val_results['Dice']))

        
        names.append(image_name)
        iris_circles_params.append(iris_circles_param.cpu().numpy()[0])
        pupil_circles_params.append(pupil_circles_param.cpu().numpy()[0])

        pred_mask_pil = transforms.ToPILImage()((pred_mask[0]>0).to(dtype=torch.uint8)*255).convert('L')
        pred_iris_mask_pil = transforms.ToPILImage()((pred_iris_mask[0]>0).to(dtype=torch.uint8)*255).convert('L')
        pred_iris_circle_mask_pil = transforms.ToPILImage()((pred_iris_circle_mask[0]>0).to(dtype=torch.uint8)*255).convert('L')
        pred_iris_edge_pil = transforms.ToPILImage()((pred_iris_edge[0]>0).to(dtype=torch.uint8)*255).convert('L')
        pred_pupil_mask_pil = transforms.ToPILImage()((pred_pupil_mask[0]>0).to(dtype=torch.uint8)*255).convert('L')
        pred_pupil_circle_mask_pil = transforms.ToPILImage()((pred_pupil_circle_mask[0]>0).to(dtype=torch.uint8)*255).convert('L')
        pred_pupil_egde_pil = transforms.ToPILImage()((pred_pupil_egde[0]>0).to(dtype=torch.uint8)*255).convert('L')

        pred_mask_pil.save(os.path.join(SegmentationClass_save_dir, image_name+'.png'))
        pred_iris_mask_pil.save(os.path.join(iris_edge_mask_raw_save_dir, image_name+'.png'))
        pred_iris_circle_mask_pil.save(os.path.join(iris_edge_mask_save_dir, image_name+'.png'))
        pred_iris_edge_pil.save(os.path.join(Outer_Boundary_save_dir, image_name+'.png'))
        pred_pupil_mask_pil.save(os.path.join(pupil_edge_mask_raw_save_dir, image_name+'.png'))
        pred_pupil_circle_mask_pil.save(os.path.join(pupil_edge_mask_save_dir, image_name+'.png'))
        pred_pupil_egde_pil.save(os.path.join(Inner_Boundary_save_dir, image_name+'.png'))

    iris_circles_params = np.asarray(iris_circles_params)
    pupil_circles_params =np.asarray(pupil_circles_params)
    params_path = save_dir + '/test_params.xlsx'
    params_data = pd.DataFrame({
        'name':names,
        'ix':iris_circles_params[:,0],
        'iy':iris_circles_params[:,1],
        'ih':iris_circles_params[:,2],
        'iw':iris_circles_params[:,3],
        'ir':iris_circles_params[:,4],
        'px':pupil_circles_params[:,0],
        'py':pupil_circles_params[:,1],
        'ph':pupil_circles_params[:,2],
        'pw':pupil_circles_params[:,3],
        'pr':pupil_circles_params[:,4]
        })
    params_data.to_excel(params_path)

    logging.info('------------------------------------------------test result-----------------------------------------------')    
    logging.info('>maks      E1:{:.7f}   Dice:{:.7f}   IOU:{:.7f}   F1:{:.7f}'.format(e1, dice, iou, f1))
    logging.info('>iris      E1:{:.7}    Dice:{:.7f}   IOU:{:.7f}'.format(iris_e1, iris_dice, iris_iou))
    logging.info('>pupil     E1:{:.7}    Dice:{:.7f}   IOU:{:.7f}'.format(pupil_e1, pupil_dice, pupil_iou))
    
    print('test done!')
    net.train()


if __name__ == '__main__':
    args = get_args()
    train_args = {
        'log_name': args.log_name,
        'checkpoints': args.checkpoints
    }

    save_dir = os.path.join('/'.join(train_args['checkpoints'].split('/')[0:-1]), 'test')
    check_mkdir(save_dir)

    logging.basicConfig(
        filename=os.path.join(save_dir, train_args['log_name']),
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )


    SegmentationClass_save_dir = os.path.join(save_dir, 'SegmentationClass')
    check_mkdir(SegmentationClass_save_dir)
    Inner_Boundary_save_dir = os.path.join(save_dir, 'Inner_Boundary')
    check_mkdir(Inner_Boundary_save_dir)
    Outer_Boundary_save_dir = os.path.join(save_dir, 'Outer_Boundary')
    check_mkdir(Outer_Boundary_save_dir)
    iris_edge_mask_raw_save_dir = os.path.join(save_dir, 'iris_edge_mask_raw')
    check_mkdir(iris_edge_mask_raw_save_dir)
    iris_edge_mask_save_dir = os.path.join(save_dir, 'iris_edge_mask')
    check_mkdir(iris_edge_mask_save_dir)
    pupil_edge_mask_raw_save_dir = os.path.join(save_dir, 'pupil_edge_mask_raw')
    check_mkdir(pupil_edge_mask_raw_save_dir)
    pupil_edge_mask_save_dir = os.path.join(save_dir, 'pupil_edge_mask')
    check_mkdir(pupil_edge_mask_save_dir)

    main(train_args)
