#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os.path as osp
import os
import time

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import nibabel as nib
import numpy as np
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from dataset import DmriDataset
from model import *
from utils import *

def build_model(model_cfg):
    if model_cfg['backbone']['type'] == 'CNN':
        model = CNN(model_cfg['backbone']['in_channel'], model_cfg['backbone']['hidden'],
                    model_cfg['backbone']['out_channel'])
    elif model_cfg['backbone']['type'] == 'MLP':
        model = MLP(model_cfg['backbone']['in_channel'], model_cfg['backbone']['hidden'],
                    model_cfg['backbone']['out_channel'], model_cfg['backbone']['drop_out'])
    elif model_cfg['backbone']['type'] == 'GCNN':
        model = GCNN(model_cfg['backbone']['in_channel'], model_cfg['backbone']['hidden'],
                    model_cfg['backbone']['out_channel'], model_cfg['backbone']['drop_out'],
                    model_cfg['backbone']['K'], model_cfg['backbone']['in_channel'],
                    model_cfg['backbone']['hidden'])
    elif model_cfg['backbone']['type'] == 'UNet':
        model = UNet(model_cfg['backbone']['out_channel'], model_cfg['backbone']['in_channel'])
    elif model_cfg['backbone']['type'] == 'NestedUNet':
        model = NestedUNet(model_cfg['backbone']['out_channel'], model_cfg['backbone']['in_channel'])
    elif model_cfg['backbone']['type'] == 'HGT':
        model = None
    else:
        print('No {} model'.format(model_cfg['backbone']['type']))
    return model

def build_optimizer(model, optimizer_cfg):
    if optimizer_cfg['type'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_cfg['lr'])
    elif optimizer_cfg['type'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=optimizer_cfg['lr'],
                                    momentum=optimizer_cfg['momentum'],
                                    weight_decay=optimizer_cfg['weight_decay'])
    else:
        print('no use {} optimizer'.format(optimizer_cfg['type']))
    return optimizer

def build_optimizer(lr_config):
    if lr_config['type'] == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.5)
    else:
        print('no use {} scheduler'.format(lr_config['type']))
    return scheduler

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="predictive microstructure model")
    parser.add_argument("--config", default='/data/jht/HGT/config/cnn_config.py',
                        type=str, help="Run the model config file address")
    parser.add_argument("--train_subject_id", default=['748662', '751348', '859671', '761957', '833148',
                                                       '837560', '845458', '896778', '898176', '100610',
                                                       '102311', '102816', '104416', '105923', '108323',
                                                       '109123', '599671', '613538', '622236', '654754'],
                        type=list, help="Training file name")

    parser.add_argument("--valid_subject_id", default=['125525', '683256', '899885'],
                        type=list, help="Validing file name")
    parser.add_argument("--data_path", default="./data",
                        type=str, help="Name of the data path")
    parser.add_argument("--mask_name", default="nodif_brain_mask.nii.gz",
                        type=str, help="Name of the mask file for the data")
    parser.add_argument("--train_data_name", default="train_unet_data_30_DKI.npy",
                        type=str, help="Name of the training file")
    parser.add_argument("--train_gt_data_name", default="train_gt_unet_data_30_DKI.npy",
                        type=str, help="Name of the training gold standard file")
    parser.add_argument("--test_data_name", default="test_unet_data_30_DKI.npy",
                        type=str, help="Name of the testing file")
    parser.add_argument("--test_gt_data_name", default="test_gt_unet_data_30_DKI.npy",
                        type=str, help="Name of the testing gold standard file")
    parser.add_argument("--is_train", default=True,
                        type=bool, help="Name of the testing gold standard file")
    parser.add_argument("--save_parameter_path", default="./parameter",
                        type=str, help="Name of the microstructure model")
    parser.add_argument("--microstructure_name", default="NODDI", choices=['NODDI', 'DKI'],
                        type=str, help="Name of the microstructure model")
    parser.add_argument("--brain_max_lenght", default="./hcp_brain_max_lenght.npy",
                        type=str, help="Storage file for the maximum length and width of the model")
    args = parser.parse_args()
    return args

def test(test_loader, edge_index, model, test_start_time, subject_id, args):
    for i, data in tqdm(enumerate(test_loader, 0), total=len(test_loader), desc='test'):
        model.eval()
        graph_data_list = []
        test_input, test_label = data
        B, H, W, C = test_input.shape
        test_input = test_input.view(-1, C)
        voxel_number = test_input.shape[0]
        for j in range(0, voxel_number, 1):
            graph_data_list.append(Data(x=test_input[j, :].view(-1, 1).type(torch.FloatTensor), edge_index=edge_index, num_nodes=C))
        graph_loader = DataLoader(graph_data_list, batch_size=voxel_number, shuffle=False)
        for data in graph_loader:
            data = data.to(device)
            out = model(data, B, H, W)
        if i==0:
            prediction = out.cpu().detach().numpy()
            gt = test_label.numpy()
        else:
            temp_out = out.cpu().detach().numpy()
            temp_gt = test_label.numpy()
            prediction = np.concatenate((prediction, temp_out), axis=0)
            gt = np.concatenate((gt, temp_gt), axis=0)
    test_time_cost = time.time() - test_start_time
    print(' took {:.2f} seconds'.format(test_time_cost))
    return restore_img(subject_id, gt, prediction, args)





def train(train_loader, model, optimizer, edge_index):
    total_loss = 0
    total_index1_loss = 0
    total_index2_loss = 0
    total_index3_loss = 0
    batch_number = 0
    epoch_time = time.time()
    for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), desc='train'):
        graph_data_list = []
        model.train()
        batch_number += 1
        optimizer.zero_grad()
        input, label= data
        B, H, W, C = input.shape
        label = label.type(torch.FloatTensor)
        label = label.to(device)
        input = input.view(-1, C)
        input = input.type(torch.FloatTensor)
        input = input.to(device)
        voxel_number = input.shape[0]
        for j in range(0, voxel_number,1):
            graph_data_list.append(Data(x=input[j, :].view(-1, 1).type(torch.FloatTensor), edge_index=edge_index, num_nodes=C))
        graph_loader = DataLoader(graph_data_list, batch_size=voxel_number, shuffle=False)
        for data in graph_loader:
            data = data.to(device)
            out = model(data, B, H, W)
            loss = F.mse_loss(out, label)
            index1_loss = F.mse_loss(out[:, :, :, 0], label[:, :, :, 0])
            index2_loss = F.mse_loss(out[:, :, :, 1], label[:, :, :, 1])
            index3_loss = F.mse_loss(out[:, :, :, 2], label[:, :, :, 2])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_index1_loss += index1_loss.item()
            total_index2_loss += index2_loss.item()
            total_index3_loss += index3_loss.item()
    total_loss = total_loss / batch_number
    total_index1_loss = total_index1_loss / batch_number
    total_index2_loss = total_index2_loss / batch_number
    total_index3_loss = total_index3_loss / batch_number
    epoch_time_cost = time.time() - epoch_time
    return total_loss, total_index1_loss, total_index2_loss, total_index3_loss, epoch_time_cost


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_cfg, data_cfg, lr_config, optimizer_cfg = file2dict(args.config)
    args.epoch = data_cfg['train']['epoches']
    if not os.path.exists(args.save_parameter_path + '/' + model_cfg['backbone']['type']):
        os.mkdir(args.save_parameter_path + '/' + model_cfg['backbone']['type'])
    if data_cfg['train']['pretrained_flag'] == True:
        args.save_parameter_path = args.save_parameter_path + '/' + model_cfg['backbone']['type'] + '/_pre_best_parameter.pth'
    else:
        args.save_parameter_path = args.save_parameter_path + '/' + model_cfg['backbone']['type'] + '/_best_parameter.pth'

    InputPatch, TargetPatch = get_concatenate_data(args.data_path, args.train_subject_id, args.train_data_name
                                                   , args.train_gt_data_name)
    input = list(InputPatch)
    targets = list(TargetPatch)

    train_dataset = DmriDataset(input, targets)
    train_loader = DataLoader(train_dataset, batch_size=data_cfg['batch_size'], shuffle=True,
                              drop_last=True, num_workers=data_cfg['num_workers'])

    model = build_model(model_cfg)
    model.to(device)

    optimizer = build_optimizer(model, optimizer_cfg)

    if data_cfg['need_edge'] == True:
        edge_index = make_edge(args.data_path, data_cfg['edge']['angle'], model_cfg['backbone']['in_channel'],
                               data_cfg['edge']['bval_name'], data_cfg['edge']['bvec_name'])
        print('edge_index size is:', edge_index.size())
    else:
        edge_index = None


    if data_cfg['train']['pretrained_flag'] == True:
        if os.path.isfile(data_cfg['train']['pretrained_weights']):
            model.load_state_dict(torch.load(data_cfg['train']['pretrained_weights']))
            print('Load parameters success!')
        else:
            print('Load parameters unsuccess!')

    if lr_config['is_scheduler'] == True:
        scheduler = build_optimizer(lr_config)
    else:
        scheduler = None

    train_start_time = time.time()
    min_loss = 1000
    all_psnr_max = 0
    for epoch in range(0, data_cfg['train']['epoches']):
        total_loss, total_index1_loss, total_index2_loss, total_index3_loss, epoch_time_cost = train(train_loader, model, optimizer, scheduler, edge_index, train_start_time, args)
        if args.microstructure_name == 'NODDI':
            print('Epoch {} time {:.2f} seconds total_loss is: {} icvf_loss is: {}  isovf_loss is: {}  od_loss is: {} '.format(
                    epoch+ 1, epoch_time_cost, total_loss, total_index1_loss, total_index2_loss, total_index3_loss))
        if args.microstructure_name == 'DKI':
            print('Epoch {} time {:.2f} seconds total_loss is: {} ak_loss is: {} mk_loss is: {} rk_loss is: {} '.format(
                epoch+ 1, epoch_time_cost, total_loss, total_index1_loss, total_index2_loss, total_index3_loss))
        if ((epoch + 1) % 1 == 0) and (total_loss < min_loss):  # % 1
            min_loss = total_loss
            print('min loss is {}'.format(min_loss))
            if (epoch > 10):
                index1_psnr, index2_psnr, index3_psnr, all_psnr = 0, 0, 0, 0
                subject_number = len(valid_subject_id)
                for subject_id in args.valid_subject_id:
                    test_start_time = time.time()
                    subject_list = []
                    subject_list.append(subject_id)
                    Test_InputPatch, Test_TargetPatch = np.ones([1, 120, 160, 30]), np.ones([1, 120, 160, 3])
                    # Test_InputPatch, Test_TargetPatch = get_concatenate_data(args.data_path, subject_id,
                    #                                                         args.test_data_name, args.test_gt_data_name)
                    test_input = list(Test_InputPatch.astype('float32'))
                    test_targets = list(Test_TargetPatch.astype('float32'))
                    test_dataset = DmriDataset(test_input, test_targets)
                    test_loader = DataLoader(test_dataset, batch_size=data_cfg['test']['batch_size'], shuffle=False,
                                             drop_last=False, num_workers=data_cfg['num_workers'])
                    if is_train:
                        index1_psnr_value, index2_psnr_value, index3_psnr_value, all_psnr_value = test(test_loader,
                                                                    edge_index, model,test_start_time, subject_id,args)
                    else:
                        print('not train')
                    index1_psnr += index1_psnr_value
                    index2_psnr += index2_psnr_value
                    index3_psnr += index3_psnr_value
                    all_psnr += all_psnr_value
                index1_psnr = index1_psnr / subject_number
                index2_psnr = index2_psnr / subject_number
                index3_psnr = index3_psnr / subject_number
                all_psnr = all_psnr / subject_number
                if args.microstructure_name == 'NODDI':
                    print('valid suject mean psnr icvf is: {}  isovf is: {}  od is: {} all_noddi is: {} '.format(
                        index1_psnr, index2_psnr, index3_psnr, all_psnr))
                if args.microstructure_name == 'DKI':
                    print('valid suject mean psnr ak is: {}  mk is: {}  rk is: {} all_dki is: {}'.format(
                        index1_psnr, index2_psnr, index3_psnr,all_psnr))
                if all_psnr_max < all_psnr:
                    all_psnr_max = all_psnr
                    torch.save(model.state_dict(), args.save_parameter_path)
                    print('{} model parameters have been stored!'.format(epoch))
        train_time_cost = time.time() - train_start_time
        print(' took {:.2f} seconds'.format(train_time_cost))

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists("./patameter"):
        os.mkdir("./patameter")
    main(args)
