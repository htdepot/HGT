import os
import sys
import types
sys.path.insert(0,os.getcwd())
import importlib
import numpy as np
import nibabel as nib
import math
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def file2dict(filename):
    (path,file) = os.path.split(filename)
    abspath = os.path.abspath(os.path.expanduser(path))
    sys.path.insert(0,abspath)
    mod = importlib.import_module(file.split('.')[0])
    sys.path.pop(0)
    cfg_dict = {
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith('__')
                and not isinstance(value, types.ModuleType)
                and not isinstance(value, types.FunctionType)
                    }
    return cfg_dict.get('model_cfg'), cfg_dict.get('data_cfg'),cfg_dict.get('lr_config'),cfg_dict.get('optimizer_cfg')

def get_concatenate_data(data_path, subject_id, file_name, file_gt_name):
    data_list = []
    gt_data_list = []
    for file_name_id in subject_id:
        print('Reading Reading hcp data sub_ID: '+ str(file_name_id))
        data = np.load(data_path + str(file_name_id) + '/' + file_name)
        gt_data = np.load(data_path + str(file_name_id) + '/' + file_gt_name)
        data_list.append(data)
        gt_data_list.append(gt_data)
        print('Reading finished!')
    for i in range(len(data_list)):
        if i == 0 :
            temporary_test_data =data_list[i]
            print('InputPatch shape is {}'.format(temporary_test_data.shape))
            temporary_gt_data =gt_data_list[i]
            print('TargetPatch shape is {}'.format(temporary_gt_data.shape))
        else:
            temporary_test_data = np.concatenate((temporary_test_data, data_list[i]), 0)
            temporary_gt_data = np.concatenate((temporary_gt_data, gt_data_list[i]), 0)
            print('InputPatch shape is {}'.format(temporary_test_data.shape))
            print('TargetPatch shape is {}'.format(temporary_gt_data.shape))
    print('InputPatch shape is {}'.format(temporary_test_data.shape))
    print('TargetPatch shape is {}'.format(temporary_gt_data.shape))
    return temporary_test_data, temporary_gt_data


def computer_psnr(gt_data, prediction_data, model_name, is_train):
    if model_name == 'NODDI':
        gt_icvf = gt_data[:, 0]
        pre_icvf = prediction_data[:, 0]

        icvf_nrmse_value = nrmse(gt_icvf, pre_icvf)
        icvf_psnr_value = psnr(gt_icvf, pre_icvf, data_range=1)
        icvf_ssim_value = ssim(gt_icvf, pre_icvf, data_range=1)
        print('icvf')
        print('NRMSE value is {}'.format(icvf_nrmse_value))
        print('PSNR value is {}'.format(icvf_psnr_value))
        print('SSIM value is {}'.format(icvf_ssim_value))

        gt_isovf = gt_data[:, 1]
        pre_isovf = prediction_data[:, 1]

        isovf_nrmse_value = nrmse(gt_isovf, pre_isovf)
        isovf_psnr_value = psnr(gt_isovf, pre_isovf, data_range=1)
        isovf_ssim_value = ssim(gt_isovf, pre_isovf, data_range=1)
        print('isovf')
        print('NRMSE value is {}'.format(isovf_nrmse_value))
        print('PSNR value is {}'.format(isovf_psnr_value))
        print('SSIM value is {}'.format(isovf_ssim_value))

        gt_od = gt_data[:, 2]
        pre_od = prediction_data[:, 2]

        od_nrmse_value = nrmse(gt_od, pre_od)
        od_psnr_value = psnr(gt_od, pre_od, data_range=1)
        od_ssim_value = ssim(gt_od, pre_od, data_range=1)
        print('od')
        print('NRMSE value is {}'.format(od_nrmse_value))
        print('PSNR value is {}'.format(od_psnr_value))
        print('SSIM value is {}'.format(od_ssim_value))

        gt_all = gt_data[:, 0:3]
        pre_all = prediction_data[:, 0:3]

        all_noddi_nrmse_value = nrmse(gt_all, pre_all)
        all_noddi_psnr_value = psnr(gt_all, pre_all, data_range=1)
        all_noddi_ssim_value = ssim(gt_all, pre_all, data_range=1, multichannel=True)
        print('all_NODDI')
        print('NRMSE value is {}'.format(all_noddi_nrmse_value))
        print('PSNR value is {}'.format(all_noddi_psnr_value))
        print('SSIM value is {}'.format(all_noddi_ssim_value))
        if is_train:
            return icvf_psnr_value, isovf_psnr_value, od_psnr_value, all_noddi_psnr_value
        else:
            return icvf_psnr_value, isovf_psnr_value, od_psnr_value, all_noddi_psnr_value, icvf_nrmse_value, isovf_nrmse_value, od_nrmse_value, all_noddi_nrmse_value, \
                   icvf_ssim_value, isovf_ssim_value, od_ssim_value, all_noddi_ssim_value
    if model_name == 'DKI':
        gt_ak = gt_data[:, 0]
        pre_ak = prediction_data[:, 0]

        ak_nrmse_value = nrmse(gt_ak, pre_ak)
        ak_psnr_value = psnr(gt_ak, pre_ak, data_range=3)
        ak_ssim_value = ssim(gt_ak, pre_ak, data_range=3)
        print('ak')
        print('NRMSE value is {}'.format(ak_nrmse_value))
        print('PSNR value is {}'.format(ak_psnr_value))
        print('SSIM value is {}'.format(ak_ssim_value))

        gt_mk = gt_data[:, 1]
        pre_mk = prediction_data[:, 1]

        mk_nrmse_value = nrmse(gt_mk, pre_mk)
        mk_psnr_value = psnr(gt_mk, pre_mk, data_range=3)
        mk_ssim_value = ssim(gt_mk, pre_mk, data_range=3)
        print('mk')
        print('NRMSE value is {}'.format(mk_nrmse_value))
        print('PSNR value is {}'.format(mk_psnr_value))
        print('SSIM value is {}'.format(mk_ssim_value))

        gt_rk = gt_data[:, 2]
        pre_rk = prediction_data[:, 2]

        rk_nrmse_value = nrmse(gt_rk, pre_rk)
        rk_psnr_value = psnr(gt_rk, pre_rk, data_range=3)
        rk_ssim_value = ssim(gt_rk, pre_rk, data_range=3)
        print('rk')
        print('NRMSE value is {}'.format(rk_nrmse_value))
        print('PSNR value is {}'.format(rk_psnr_value))
        print('SSIM value is {}'.format(rk_ssim_value))

        gt_dki_all = gt_data[:, 0:3]
        pre_dki_all = prediction_data[:, 0:3]

        all_dki_nrmse_value = nrmse(gt_dki_all, pre_dki_all)
        all_dki_psnr_value = psnr(gt_dki_all, pre_dki_all, data_range=3)
        all_dki_ssim_value = ssim(gt_dki_all, pre_dki_all, data_range=3,multichannel=True)
        print('All_DKI')
        print('NRMSE value is {}'.format(all_dki_nrmse_value))
        print('PSNR value is {}'.format(all_dki_psnr_value))
        print('SSIM value is {}'.format(all_dki_ssim_value))
        if is_train:
            return ak_psnr_value, mk_psnr_value, rk_psnr_value, all_dki_psnr_value
        else:
            return ak_psnr_value, mk_psnr_value, rk_psnr_value, all_dki_psnr_value, ak_nrmse_value, mk_nrmse_value, rk_nrmse_value, all_dki_nrmse_value, \
                   ak_ssim_value, mk_ssim_value, rk_ssim_value, all_dki_ssim_value

def restore_img(subject_id, gt, prediction, **args):
    mask_file = args.data_path + subject_id + '/' + args.ask_name
    mask = nib.load(mask_file)
    mask = mask.get_fdata()
    x_size = mask.shape[0]
    y_size = mask.shape[1]
    z_size = mask.shape[2]
    gt_like = np.zeros([x_size, y_size, z_size, 3])
    prediction_like = np.zeros([x_size, y_size, z_size, 3])
    psnr_gt = []
    psnr_prediction = []


    if os.path.exists(args.brain_max_lenght):
        list_lenght = np.load(args.brain_max_lenght)
        xx_min, xx_max, yy_min, yy_max = list_lenght[0], list_lenght[1], list_lenght[2], list_lenght[3]
    else:
        print('Missing brain length file')

    edge_distance_x_start = xx_min
    edge_distance_x_ed = xx_max
    x_length = xx_max - xx_min
    x_final_length = math.ceil(x_length / 10) * 10
    x_add_length = x_final_length - x_length
    x_left_add_length = math.floor(x_add_length / 2)

    edge_distance_y_start = yy_min
    edge_distance_y_ed = yy_max
    y_length = yy_max - yy_min
    y_final_length = math.ceil(y_length / 10) * 10
    y_add_length = y_final_length - y_length
    y_left_add_length = math.floor(y_add_length / 2)

    for i in range(0, z_size, 1):
        prediction_like[edge_distance_x_start:edge_distance_x_ed, edge_distance_y_start:edge_distance_y_ed, i, :] = prediction[i, x_left_add_length:x_left_add_length + x_length, y_left_add_length:y_left_add_length + y_length, :]
        gt_like[edge_distance_x_start:edge_distance_x_ed, edge_distance_y_start:edge_distance_y_ed, i, :] = gt[i, x_left_add_length:x_left_add_length + x_length, y_left_add_length:y_left_add_length + y_length, :]


    for xx in range(edge_distance_x_start, edge_distance_x_ed, 1):
        for yy in range(edge_distance_y_start, edge_distance_y_ed, 1):
            for zz in range(0, z_size, 1):
                if mask[xx, yy, zz] > 0:
                    psnr_gt.append(gt_like[xx, yy, zz, :])
                    psnr_prediction.append(prediction_like[xx, yy, zz, :])
                else:
                    gt_like[xx, yy, zz, :] = 0
                    prediction_like[xx, yy, zz, :] = 0

    psnr_gt = np.array(psnr_gt)
    psnr_prediction = np.array(psnr_prediction)

    if is_generate_image:
        affine = np.eye(4)
        img_prediction = nib.Nifti1Image(prediction_like, affine)
        nib.save(img_prediction, image_file_path)
        print('save img successful')
    return computer_psnr(psnr_gt, psnr_prediction, args.microstructure_name, args.is_train)