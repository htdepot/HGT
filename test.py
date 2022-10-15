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

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="predictive microstructure model")
    parser.add_argument("--config", default='./config/hgt_config.py',
                        type=str, help="Run the model config file address")
    parser.add_argument("--test_subject_id", default=['672756', '677968', '680957', '685058', '111312',
                                                      '111514', '114823', '130518', '144226', '896879',
                                                      '177746', '185442', '195041', '200614', '204521',
                                                      '146129', '158035', '562345', '627549', '783462'],
                        type=list, help="Training file name")
    parser.add_argument("--data_path", default="/data/jht/gnn_pvt/data",
                        type=str, help="Name of the data path")
    parser.add_argument("--mask_name", default="nodif_brain_mask.nii.gz",
                        type=str, help="Name of the mask file for the data")
    parser.add_argument("--test_data_name", default="train_ght_data_30_NODDI.npy",
                        type=str, help="Name of the testing file")
    parser.add_argument("--test_gt_data_name", default="train_gt_ght_data_30_NODDI.npy",
                        type=str, help="Name of the testing gold standard file")
    parser.add_argument("--is_train", default=False,
                        type=bool, help="Name of the testing gold standard file")
    parser.add_argument("--is_generate_image", default=False,
                        type=bool, help="Whether to generate prediction")
    parser.add_argument("--generate_image_save_path", default='./image',
                        type=str, help="Whether to generate prediction")
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

def test(test_loader, edge_index, model, test_start_time, device, model_name):
    if edge_index is not None:
        edge_index = edge_index.to(device)
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
    return prediction, gt


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_cfg, data_cfg, _, _ = file2dict(args.config)

    model = build_model(model_cfg)
    model.to(device)

    if data_cfg['need_edge'] == True:
        if os.path.isfile('./test_' + args.edge_save_name + '.npy'):
            test_edge_index = torch.load('./test_' + args.edge_save_name + '.npy')
            edge_index = torch.LongTensor(test_edge_index)
        else:
            make_edge(args.data_path, data_cfg['edge']['angle'], model_cfg['backbone']['in_channel'],
                      data_cfg['edge']['bval_name'], data_cfg['edge']['bvec_name'], data_cfg['edge']['image_shape'],
                      data_cfg['test']['batch_size'], './test_' + args.edge_save_name + '.npy', model_cfg['backbone']['type'])
            test_edge_index = torch.load('./test_' + args.edge_save_name + '.npy')
            edge_index = torch.LongTensor(test_edge_index)
    else:
        edge_index = None

    if data_cfg['train']['pretrained_flag'] == True:
        if os.path.isfile(data_cfg['train']['pretrained_weights']):
            model.load_state_dict(torch.load(data_cfg['train']['pretrained_weights']))
            print('Load parameters success!')
        else:
            print('Load parameters unsuccess!')

    if args.microstructure_name == 'NODDI':
        icvf_nrmse_values = []
        icvf_psnr_values = []
        icvf_ssim_values = []

        isovf_nrmse_values = []
        isovf_psnr_values = []
        isovf_ssim_values = []

        od_nrmse_values = []
        od_psnr_values = []
        od_ssim_values = []

        all_noddi_nrmse_values = []
        all_noddi_psnr_values = []
        all_noddi_ssim_values = []

    if args.microstructure_name == 'DKI':
        ak_nrmse_values = []
        ak_psnr_values = []
        ak_ssim_values = []

        mk_nrmse_values = []
        mk_psnr_values = []
        mk_ssim_values = []

        rk_nrmse_values = []
        rk_psnr_values = []
        rk_ssim_values = []

        all_dki_nrmse_values = []
        all_dki_psnr_values = []
        all_dki_ssim_values = []

    test_all_start_time = time.time()

    for subject_id in args.test_subject_id:
        test_start_time = time.time()
        subject_list = []
        subject_list.append(subject_id)

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

        prediction, gt = test(test_loader, edge_index, model, test_start_time, device, model_cfg['backbone']['type'])

        if args.microstructure_name == 'NODDI':
            icvf_psnr_value, isovf_psnr_value, od_psnr_value, all_noddi_psnr_value, \
            icvf_nrmse_value, isovf_nrmse_value, od_nrmse_value, all_noddi_nrmse_value, \
            icvf_ssim_value, isovf_ssim_value, od_ssim_value, all_noddi_ssim_value = \
            restore_img(subject_id, gt, prediction, data_cfg['is_voxel'], args)
            icvf_nrmse_values.append(icvf_nrmse_value)
            icvf_psnr_values.append(icvf_psnr_value)
            icvf_ssim_values.append(icvf_ssim_value)

            isovf_nrmse_values.append(isovf_nrmse_value)
            isovf_psnr_values.append(isovf_psnr_value)
            isovf_ssim_values.append(isovf_ssim_value)

            od_nrmse_values.append(od_nrmse_value)
            od_psnr_values.append(od_psnr_value)
            od_ssim_values.append(od_ssim_value)

            all_noddi_nrmse_values.append(all_noddi_nrmse_value)
            all_noddi_psnr_values.append(all_noddi_psnr_value)
            all_noddi_ssim_values.append(all_noddi_ssim_value)
        elif args.microstructure_name == 'DKI':
            ak_psnr_value, mk_psnr_value, rk_psnr_value, all_dki_psnr_value, \
            ak_nrmse_value, mk_nrmse_value, rk_nrmse_value, all_dki_nrmse_value, \
            ak_ssim_value, mk_ssim_value, rk_ssim_value, all_dki_ssim_value = \
            restore_img(subject_id, gt, prediction, data_cfg['is_voxel'], args)
            ak_nrmse_values.append(ak_nrmse_value)
            ak_psnr_values.append(ak_psnr_value)
            ak_ssim_values.append(ak_ssim_value)
            mk_nrmse_values.append(mk_nrmse_value)
            mk_psnr_values.append(mk_psnr_value)
            mk_ssim_values.append(mk_ssim_value)
            rk_nrmse_values.append(rk_nrmse_value)
            rk_psnr_values.append(rk_psnr_value)
            rk_ssim_values.append(rk_ssim_value)
            all_dki_nrmse_values.append(all_dki_nrmse_value)
            all_dki_psnr_values.append(all_dki_psnr_value)
            all_dki_ssim_values.append(all_dki_ssim_value)
        else:
            print('not have prediciton model')

    if args.microstructure_name == 'NODDI':
        icvf_nrmse_values = np.array(icvf_nrmse_values)
        icvf_psnr_values = np.array(icvf_psnr_values)
        icvf_ssim_values = np.array(icvf_ssim_values)

        isovf_nrmse_values = np.array(isovf_nrmse_values)
        isovf_psnr_values = np.array(isovf_psnr_values)
        isovf_ssim_values = np.array(isovf_ssim_values)

        od_nrmse_values = np.array(od_nrmse_values)
        od_psnr_values = np.array(od_psnr_values)
        od_ssim_values = np.array(od_ssim_values)

        all_noddi_nrmse_values = np.array(all_noddi_nrmse_values)
        all_noddi_psnr_values = np.array(all_noddi_psnr_values)
        all_noddi_ssim_values = np.array(all_noddi_ssim_values)

        print('mean icvf nrmse {} psnr {} ssim {}'.format(icvf_nrmse_values.mean(), icvf_psnr_values.mean(),
                                                          icvf_ssim_values.mean()))
        print('mean isovf nrmse {} psnr {} ssim {}'.format(isovf_nrmse_values.mean(), isovf_psnr_values.mean(),
                                                           isovf_ssim_values.mean()))
        print(
            'mean od nrmse {} psnr {} ssim {}'.format(od_nrmse_values.mean(), od_psnr_values.mean(),
                                                      od_ssim_values.mean()))
        print(
            'mean all nrmse {} psnr {} ssim {}'.format(all_noddi_nrmse_values.mean(), all_noddi_psnr_values.mean(),
                                                       all_noddi_ssim_values.mean()))

    if args.microstructure_name == 'DKI':
        ak_nrmse_values = np.array(ak_nrmse_values)
        ak_psnr_values = np.array(ak_psnr_values)
        ak_ssim_values = np.array(ak_ssim_values)

        mk_nrmse_values = np.array(mk_nrmse_values)
        mk_psnr_values = np.array(mk_psnr_values)
        mk_ssim_values = np.array(mk_ssim_values)

        rk_nrmse_values = np.array(rk_nrmse_values)
        rk_psnr_values = np.array(rk_psnr_values)
        rk_ssim_values = np.array(rk_ssim_values)

        all_dki_nrmse_values = np.array(all_dki_nrmse_values)
        all_dki_psnr_values = np.array(all_dki_psnr_values)
        all_dki_ssim_values = np.array(all_dki_ssim_values)

        print('mean ak nrmse {} psnr {} ssim {}'.format(ak_nrmse_values.mean(), ak_psnr_values.mean(),
                                                        ak_ssim_values.mean()))
        print('mean mk nrmse {} psnr {} ssim {}'.format(mk_nrmse_values.mean(), mk_psnr_values.mean(),
                                                        mk_ssim_values.mean()))
        print('mean rk nrmse {} psnr {} ssim {}'.format(rk_nrmse_values.mean(), rk_psnr_values.mean(),
                                                        rk_ssim_values.mean()))
        print('mean all nrmse {} psnr {} ssim {}'.format(all_dki_nrmse_values.mean(), all_dki_psnr_values.mean(),
                                                         all_dki_ssim_values.mean()))
    test_all_time_cost = time.time() - test_all_start_time
    print(' took {:.2f} seconds'.format(test_all_time_cost))

if __name__ == "__main__":
    args = parse_args()
    main(args)