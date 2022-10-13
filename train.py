#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Sat Mar 24 09:59:10 2021

@author: Jiquan

Graph-based on pytorch geometric
'''
import os.path as osp
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
        model = MLP(model_cfg['backbone']['in_channel'], model_cfg['backbone']['features'],
                    model_cfg['backbone']['out_channel'], model_cfg['backbone']['drop_out'])
    elif model_cfg['backbone']['type'] == 'GCNN':
        model = GCNN(model_cfg['backbone']['in_channel'], model_cfg['backbone']['hidden'],
                     model_cfg['backbone']['out_channel'], model_cfg['backbone']['K'],
                     model_cfg['backbone']['hidden'])
    elif model_cfg['backbone']['type'] == 'UNet':
        model = UNet(model_cfg['backbone']['out_channel'], model_cfg['backbone']['in_channel'])
    elif model_cfg['backbone']['type'] == 'NestedUNet':
        model = NestedUNet(model_cfg['backbone']['out_channel'], model_cfg['backbone']['in_channel'])
    elif model_cfg['backbone']['type'] == 'HGT':
        model = HGT(in_channel=model_cfg['backbone']['in_channel'], embed_dims=model_cfg['backbone']['embed_dims'],
                    num_heads=model_cfg['backbone']['num_heads'], mlp_ratios=model_cfg['backbone']['mlp_ratios'],
                    qkv_bias=model_cfg['backbone']['qkv_bias'], depths=model_cfg['backbone']['depths'],
                    sr_ratios=model_cfg['backbone']['sr_ratios'], drop_rate=model_cfg['backbone']['drop_rate'],
                    attn_drop_rate=model_cfg['backbone']['attn_drop_rate'],
                    drop_path_rate=model_cfg['backbone']['drop_path_rate'],
                    num_stages=model_cfg['backbone']['num_stages'],
                    gradient_direction_number=model_cfg['backbone']['gradient_direction_number'],
                    gnn_dim=model_cfg['backbone']['gnn_dim'],
                    gnn_out=model_cfg['backbone']['gnn_out'], K=model_cfg['backbone']['K'])
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

def build_scheduler(lr_config, optimizer):
    if lr_config['type'] == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.5)
    else:
        print('no use {} scheduler'.format(lr_config['type']))
    return scheduler

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="predictive microstructure model")
    parser.add_argument("--config", default='./config/hgt_config.py',
                        type=str, help="Run the model config file address")
    parser.add_argument("--train_subject_id", default=['748662', '751348', '859671', '761957', '833148',
                                                       '837560', '845458', '896778', '898176', '100610',
                                                       '102311', '102816', '104416', '105923', '108323',
                                                       '109123', '599671', '613538', '622236', '654754'],
                        type=list, help="Training file name")

    parser.add_argument("--valid_subject_id", default=['125525', '683256', '899885'],
                        type=list, help="Validing file name")
    parser.add_argument("--data_path", default="/data/jht/gnn_pvt/data",
                        type=str, help="Name of the data path")
    parser.add_argument("--mask_name", default="nodif_brain_mask.nii.gz",
                        type=str, help="Name of the mask file for the data")
    parser.add_argument("--train_data_name", default="train_ght_data_30_NODDI.npy",
                        type=str, help="Name of the training file")
    parser.add_argument("--train_gt_data_name", default="train_gt_ght_data_30_NODDI.npy",
                        type=str, help="Name of the training gold standard file")
    parser.add_argument("--test_data_name", default="train_ght_data_30_NODDI.npy",
                        type=str, help="Name of the testing file")
    parser.add_argument("--test_gt_data_name", default="train_gt_ght_data_30_NODDI.npy",
                        type=str, help="Name of the testing gold standard file")
    parser.add_argument("--is_train", default=True,
                        type=bool, help="Name of the testing gold standard file")
    parser.add_argument("--is_generate_image", default=False,
                        type=bool, help="Whether to generate prediction")
    parser.add_argument("--generate_image_save_path", default='./image',
                        type=bool, help="Whether to generate prediction")
    parser.add_argument("--save_parameter_path", default="./parameter",
                        type=str, help="Name of the microstructure model")
    parser.add_argument("--microstructure_name", default="NODDI", choices=['NODDI', 'DKI'],
                        type=str, help="Name of the microstructure model")
    parser.add_argument("--brain_max_lenght", default="/data/jht/gnn_pvt/hcp_brain_max_lenght.npy",
                        type=str, help="Storage file for the maximum length and width of the model")
    parser.add_argument("--edge_save_name", default="edge",
                        type=str, help="Storage name of the edge")
    parser.add_argument("--gpu_id", default="3",
                        type=str, help="Which GPU to be used")
    args = parser.parse_args()
    return args

def test(test_loader, edge_index, model, test_start_time, subject_id, device, is_voxel, model_name, kwargs):
    if edge_index is not None:
        edge_index = edge_index[1].to(device)
    for i, data in tqdm(enumerate(test_loader, 0), total=len(test_loader), desc='test'):
        model.eval()
        if model_name == 'GCNN':
            test_input = data.to(device)
            test_label = data.y.cpu()
        else:
            test_input, test_label = data
            test_input = test_input.type(torch.FloatTensor)
            test_input = test_input.to(device)

        if edge_index is not None:
            out = model(test_input, edge_index)
        else:
            out = model(test_input)

        if i == 0:
            prediction = out.cpu().detach().numpy()
            gt = test_label.numpy()
        else:
            temp_out = out.cpu().detach().numpy()
            temp_gt = test_label.numpy()
            prediction = np.concatenate((prediction, temp_out), axis=0)
            gt = np.concatenate((gt, temp_gt), axis=0)
    test_time_cost = time.time() - test_start_time
    print(' took {:.2f} seconds'.format(test_time_cost))
    return restore_img(subject_id, gt, prediction, is_voxel, kwargs)





def train(train_loader, model, optimizer, scheduler, edge_index, device, is_voxel, model_name):
    total_loss = 0
    total_index1_loss = 0
    total_index2_loss = 0
    total_index3_loss = 0
    batch_number = 0
    epoch_time = time.time()
    if edge_index is not None:
        edge_index = edge_index[0].to(device)
    for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), desc='train'):
        model.train()
        batch_number += 1
        optimizer.zero_grad()
        if model_name == 'GCNN':
            input = data.to(device)
            label = data.y
        else:
            input, label = data
            label = label.type(torch.FloatTensor)
            label = label.to(device)
            input = input.type(torch.FloatTensor)
            input = input.to(device)
        if edge_index is not None:
            out = model(input, edge_index)
        else:
            out = model(input)

        loss = F.mse_loss(out, label)
        if is_voxel:
            index1_loss = F.mse_loss(out[:, 0], label[:, 0])
            index2_loss = F.mse_loss(out[:, 1], label[:, 1])
            index3_loss = F.mse_loss(out[:, 2], label[:, 2])
        else:
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
    if scheduler is not None:
        scheduler.step()
    return total_loss, total_index1_loss, total_index2_loss, total_index3_loss, epoch_time_cost


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_cfg, data_cfg, lr_config, optimizer_cfg = file2dict(args.config)
    if not os.path.exists(args.save_parameter_path + '/' + model_cfg['backbone']['type']):
        os.mkdir(args.save_parameter_path + '/' + model_cfg['backbone']['type'])

    if data_cfg['train']['pretrained_flag'] == True:
        args.save_parameter_path = args.save_parameter_path + '/' + model_cfg['backbone']['type'] + '_pre_best_parameter.pth'
    else:
        args.save_parameter_path = args.save_parameter_path + '/' + model_cfg['backbone']['type'] + '_best_parameter.pth'

    #InputPatch, TargetPatch = np.ones([1, 120, 160, 30]), np.ones([1, 120, 160, 3])
    InputPatch, TargetPatch = get_concatenate_data(args.data_path, args.train_subject_id, args.train_data_name
                                                   , args.train_gt_data_name)
    if model_cfg['backbone']['type'] == 'GCNN':
        InputPatch = torch.from_numpy(InputPatch)
        TargetPatch = torch.from_numpy(TargetPatch)
        graph_data_list = []
        voxel_number = InputPatch.shape[0]
        edge_index = make_edge(args.data_path, data_cfg['edge']['angle'], model_cfg['backbone']['in_channel'],
                      data_cfg['edge']['bval_name'], data_cfg['edge']['bvec_name'], data_cfg['edge']['image_shape'],
                      data_cfg['batch_size'], None, model_cfg['backbone']['type'])
        for j in range(0, voxel_number, 1):
            graph_data_list.append(Data(x=InputPatch[j, :].view(-1, 1).type(torch.FloatTensor), edge_index=edge_index,
                                        y=TargetPatch[j, :].view(1, -1).type(torch.FloatTensor)))
        from torch_geometric.loader import DataLoader as graph_loader
        train_loader = graph_loader(graph_data_list, batch_size=data_cfg['batch_size'], shuffle=True, drop_last=True
                                  , num_workers=data_cfg['num_workers'])
    else:
        input = list(InputPatch)
        targets = list(TargetPatch)

        train_dataset = DmriDataset(input, targets)
        train_loader = DataLoader(train_dataset, batch_size=data_cfg['batch_size'], shuffle=True,
                                  drop_last=True, num_workers=data_cfg['num_workers'])

    model = build_model(model_cfg)
    model.to(device)

    optimizer = build_optimizer(model, optimizer_cfg)

    edge_index = []
    if data_cfg['need_edge'] == True:
        if os.path.isfile('./train_' + args.edge_save_name + '.npy') and os.path.isfile('./test_' + args.edge_save_name + '.npy'):
            train_edge_index = torch.load('./train_' + args.edge_save_name + '.npy')
            train_edge_index = torch.LongTensor(train_edge_index)
            edge_index.append(train_edge_index)

            test_edge_index = torch.load('./test_' + args.edge_save_name + '.npy')
            test_edge_index = torch.LongTensor(test_edge_index)
            edge_index.append(test_edge_index)
        else:
            make_edge(args.data_path, data_cfg['edge']['angle'], model_cfg['backbone']['in_channel'],
                      data_cfg['edge']['bval_name'], data_cfg['edge']['bvec_name'], data_cfg['edge']['image_shape'],
                      data_cfg['batch_size'], './train_' + args.edge_save_name + '.npy', model_cfg['backbone']['type'])
            train_edge_index = torch.load('./train_' + args.edge_save_name + '.npy')
            train_edge_index = torch.LongTensor(train_edge_index)
            edge_index.append(train_edge_index)

            make_edge(args.data_path, data_cfg['edge']['angle'], model_cfg['backbone']['in_channel'],
                      data_cfg['edge']['bval_name'], data_cfg['edge']['bvec_name'], data_cfg['edge']['image_shape'],
                      data_cfg['test']['batch_size'], './test_' + args.edge_save_name + '.npy', model_cfg['backbone']['type'])
            test_edge_index = torch.load('./test_' + args.edge_save_name + '.npy')
            test_edge_index = torch.LongTensor(test_edge_index)
            edge_index.append(test_edge_index)
    else:
        edge_index = None


    if data_cfg['train']['pretrained_flag'] == True:
        if os.path.isfile(data_cfg['train']['pretrained_weights']):
            model.load_state_dict(torch.load(data_cfg['train']['pretrained_weights']))
            print('Load parameters success!')
        else:
            print('Load parameters unsuccess!')

    if lr_config['is_scheduler'] == True:
        scheduler = build_scheduler(lr_config, optimizer)
    else:
        scheduler = None

    train_start_time = time.time()
    min_loss = 1000
    all_psnr_max = 0
    early_stop_number = 0
    for epoch in range(0, data_cfg['train']['epoches']):
        total_loss, total_index1_loss, total_index2_loss, total_index3_loss, epoch_time_cost = train(train_loader,
                                                                                                     model,
                                                                                                     optimizer,
                                                                                                     scheduler,
                                                                                                     edge_index,
                                                                                                     device,
                                                                                                     data_cfg['is_voxel'],
                                                                                                     model_cfg['backbone']['type'])
        if args.microstructure_name == 'NODDI':
            print('Epoch {} time {:.2f} seconds total_loss is: {} icvf_loss is: {}  isovf_loss is: {}  od_loss is: {} '.format(
                    epoch+ 1, epoch_time_cost, total_loss, total_index1_loss, total_index2_loss, total_index3_loss))
        if args.microstructure_name == 'DKI':
            print('Epoch {} time {:.2f} seconds total_loss is: {} ak_loss is: {} mk_loss is: {} rk_loss is: {} '.format(
                epoch+ 1, epoch_time_cost, total_loss, total_index1_loss, total_index2_loss, total_index3_loss))
        if ((epoch + 1) % 1 == 0) and (total_loss < min_loss):  # % 1
            min_loss = total_loss
            print('min loss is {}'.format(min_loss))
            index1_psnr, index2_psnr, index3_psnr, all_psnr = 0, 0, 0, 0
            subject_number = len(args.valid_subject_id)
            for subject_id in args.valid_subject_id:
                test_start_time = time.time()
                subject_list = []
                subject_list.append(subject_id)
                #Test_InputPatch, Test_TargetPatch = np.ones([145, 120, 160, 30]), np.ones([145, 120, 160, 3])
                Test_InputPatch, Test_TargetPatch = get_concatenate_data(args.data_path, subject_list,
                                                                         args.test_data_name, args.test_gt_data_name)
                if model_cfg['backbone']['type'] == 'GCNN':
                    Test_InputPatch = torch.from_numpy(Test_InputPatch.astype('float32'))
                    Test_TargetPatch = torch.from_numpy(Test_TargetPatch.astype('float32'))
                    graph_data_list = []
                    voxel_number = Test_InputPatch.shape[0]
                    edge_index = make_edge(args.data_path, data_cfg['edge']['angle'],
                                           model_cfg['backbone']['in_channel'],
                                           data_cfg['edge']['bval_name'], data_cfg['edge']['bvec_name'],
                                           data_cfg['edge']['image_shape'],
                                           data_cfg['batch_size'], None,
                                           model_cfg['backbone']['type'])
                    for j in range(0, voxel_number, 1):
                        graph_data_list.append(
                            Data(x=Test_InputPatch[j, :].view(-1, 1).type(torch.FloatTensor), edge_index=edge_index,
                                 y=Test_TargetPatch[j, :].view(1, -1).type(torch.FloatTensor)))
                    from torch_geometric.loader import DataLoader as graph_loader
                    test_loader = graph_loader(graph_data_list, batch_size=data_cfg['test']['batch_size'],
                                                shuffle=False, drop_last=False, num_workers=data_cfg['num_workers'])
                    edge_index = None
                else:
                    test_input = list(Test_InputPatch.astype('float32'))
                    test_targets = list(Test_TargetPatch.astype('float32'))
                    test_dataset = DmriDataset(test_input, test_targets)
                    test_loader = DataLoader(test_dataset, batch_size=data_cfg['test']['batch_size'], shuffle=False,
                                             drop_last=False, num_workers=data_cfg['num_workers'])
                if args.is_train:
                    index1_psnr_value, index2_psnr_value, index3_psnr_value, all_psnr_value = test(test_loader,
                                                                edge_index, model, test_start_time, subject_id,
                                                                device, data_cfg['is_voxel'],
                                                                model_cfg['backbone']['type'], args)
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
                early_stop_number = 0
                print('{} model parameters have been stored!'.format(epoch))
            else:
                early_stop_number += 1
                print('break number is {}'.format(early_stop_number))
        else:
            early_stop_number += 1
            print('break number is {}'.format(early_stop_number))
        if early_stop_number > 9:
            break
    train_time_cost = time.time() - train_start_time
    print(' took {:.2f} seconds'.format(train_time_cost))

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists("./parameter"):
        os.mkdir("./parameter")
    main(args)