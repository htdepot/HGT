import os
import numpy as np
import nibabel as nib
import math

def get_b0(bvals):
    b0 = []
    i = 0
    for val in bvals:
        if (val < 15):
            b0.append(i)
        i += 1
    return b0

def delete_9_b0(bvals):
    select = []
    i = 0
    number = 0
    for val in bvals:
        if(val < 1200 and val > 900):
            if(number+1 > 9 ):
                break
            select.append(i)
            number += 1
        i += 1
    return select

def delete_30_b0(bvals):
    select = []
    i = 0
    number = 0
    for val in bvals:
        if(val < 1200 and val > 900):
            if(number+1 > 30 ):
                break
            select.append(i)
            number += 1
        i += 1
    return select


def delete_60_b0(bvals):
    select = []
    i = 0
    j = 0
    number = 0
    number1 = 0
    for val in bvals:

        if(val < 1200 and val > 900):
            if(number+1 > 30 ):
                break
            select.append(i)
            number += 1
        i += 1

    for val in bvals:

        if(val < 2300 and val > 1300):
            if(number1+1 > 30 ):
                break
            select.append(j)
            number1 += 1
        j += 1
    return select


def ReadBVal(bvalfile):
    # Read bval file
    bvalf = open(bvalfile, 'r')
    bvalstr = bvalf.readline()
    bvalarr = np.fromstring(bvalstr, dtype=int, sep=' ')
    bvalf.close()
    return bvalarr

def calculate_brain_max_lenght(subject_id, data_path, mask_name):
    xx_min, yy_min = float('inf'), float('inf')
    xx_max, yy_max = 0, 0
    for name_id in subject_id:
        mask = nib.load(data_path + '/' + name_id + '/' + mask_name)
        mask_data = mask.get_fdata()
        x_size, y_size, z_size = mask_data.shape
        for i in range(0, x_size, 1):
            for j in range(0, y_size, 1):
                for k in range(0, z_size, 1):
                    if mask_data[i, j, k] > 0:
                        if(i < xx_min):
                            xx_min = i
                        if(j < yy_min):
                            yy_min = j
                        if(i > xx_max):
                            xx_max = i
                        if(j > yy_max):
                            yy_max = j
    return xx_min, xx_max, yy_min, yy_max



def normalization(data, b0_data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            mean_b0 = b0_data[i, j, :].mean()
            if(mean_b0 == 0):
                continue
            else:
                data[i, j, :] = data[i, j, :] / mean_b0
            i += 1
        j += 1

    data[data > 1] = 1
    data[data < 0] = 0
    nor_data = data
    return nor_data


def prepare_data(gradient_direction, is_train, model_name, list_lenght, dmri_file_path, mask_file_path, bvals_file_path, index1_file_path, index2_file_path, index3_file_path, save_hgt_data_path, save_hgt_gt_data_path):
    img = nib.load(dmri_file_path)
    img_mask = nib.load(mask_file_path)

    if model_name == 'NODDI':
        icvf = nib.load(index1_file_path)
        isovf = nib.load(index2_file_path)
        od = nib.load(index3_file_path)
        icvf_data = icvf.get_fdata()
        icvf_data = np.array(icvf_data)
        isovf_data = isovf.get_fdata()
        isovf_data = np.array(isovf_data)
        od_data = od.get_fdata()
        od_data = np.array(od_data)
        icvf_set = []
        isovf_set = []
        od_set = []
    if model_name == 'DKI':
        ak = nib.load(index1_file_path)
        mk = nib.load(index2_file_path)
        rk = nib.load(index3_file_path)
        ak_data = ak.get_fdata()
        ak_data = np.array(ak_data)
        mk_data = mk.get_fdata()
        mk_data = np.array(mk_data)
        rk_data = rk.get_fdata()
        rk_data = np.array(rk_data)
        ak_set = []
        mk_set = []
        rk_set = []

    xx_min, xx_max, yy_min, yy_max = list_lenght[0], list_lenght[1], list_lenght[2], list_lenght[3]

    data = img.get_fdata()
    x_size = data.shape[0]
    y_size = data.shape[1]
    z_size = data.shape[2]

    data = np.array(data)
    mask = img_mask.get_fdata()


    bvals = ReadBVal(bvals_file_path)
    b0 = get_b0(bvals)
    if gradient_direction == 30:
        delete_select_b0 = delete_30_b0(bvals)
    elif gradient_direction == 60:
        delete_select_b0 = delete_60_b0(bvals)
    elif gradient_direction == 9:
        delete_select_b0 = delete_9_b0(bvals)
    else:
        print('gradient direction is error')

    data_like = np.zeros([x_size, y_size, z_size, gradient_direction])


    for xx in range(0, x_size, 1):
        for yy in range(0, y_size, 1):
            for zz in range(0, z_size, 1):
                if mask[xx, yy, zz] > 0:
                    select_data = data[xx, yy, zz, delete_select_b0]
                    select_b0 = data[xx, yy, zz, b0]
                    if(select_b0.mean() != 0):
                        data_like[xx, yy, zz, :] = select_data / select_b0.mean()
                else:
                    if model_name == 'NODDI':
                        icvf_data[xx, yy, zz] = 0
                        isovf_data[xx, yy, zz] = 0
                        od_data[xx, yy, zz] = 0
                    if model_name == 'DKI':
                        ak_data[xx, yy, zz] = 0
                        mk_data[xx, yy, zz] = 0
                        rk_data[xx, yy, zz] = 0

    data_like[data_like > 1] = 1

    section_set = []

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
        data_section = data_like[edge_distance_x_start:edge_distance_x_ed, edge_distance_y_start:edge_distance_y_ed, i, 0:gradient_direction]
        data_section = np.pad(data_section, ((x_left_add_length, x_add_length - x_left_add_length), (y_left_add_length, y_add_length - y_left_add_length),  (0, 0)))
        section_set.append(data_section)

        if model_name == 'NODDI':
            select_icvf = icvf_data[edge_distance_x_start:edge_distance_x_ed, edge_distance_y_start:edge_distance_y_ed, i]
            select_icvf = np.pad(select_icvf, ((x_left_add_length, x_add_length - x_left_add_length), (y_left_add_length, y_add_length - y_left_add_length)))
            icvf_set.append(select_icvf)

            select_isovf = isovf_data[edge_distance_x_start:edge_distance_x_ed, edge_distance_y_start:edge_distance_y_ed, i]
            select_isovf = np.pad(select_isovf, ((x_left_add_length, x_add_length - x_left_add_length), (y_left_add_length, y_add_length - y_left_add_length)))
            isovf_set.append(select_isovf)

            select_od = od_data[edge_distance_x_start:edge_distance_x_ed, edge_distance_y_start:edge_distance_y_ed, i]
            select_od = np.pad(select_od, ((x_left_add_length, x_add_length - x_left_add_length), (y_left_add_length, y_add_length - y_left_add_length)))
            od_set.append(select_od)

        if model_name == 'DKI':
            select_ak = ak_data[edge_distance_x_start:edge_distance_x_ed, edge_distance_y_start:edge_distance_y_ed, i]
            select_ak = np.pad(select_ak, ((x_left_add_length, x_add_length - x_left_add_length), (y_left_add_length, y_add_length - y_left_add_length)))
            ak_set.append(select_ak)

            select_mk = mk_data[edge_distance_x_start:edge_distance_x_ed, edge_distance_y_start:edge_distance_y_ed, i]
            select_mk = np.pad(select_mk, ((x_left_add_length, x_add_length - x_left_add_length), (y_left_add_length, y_add_length - y_left_add_length)))
            mk_set.append(select_mk)

            select_rk = rk_data[edge_distance_x_start:edge_distance_x_ed, edge_distance_y_start:edge_distance_y_ed, i]
            select_rk = np.pad(select_rk, ((x_left_add_length, x_add_length - x_left_add_length), (y_left_add_length, y_add_length - y_left_add_length)))
            rk_set.append(select_rk)

    section_set = np.array(section_set)

    if model_name == 'NODDI':
        icvf_set = np.array(icvf_set)
        icvf_set = np.expand_dims(icvf_set, axis=3)

        isovf_set = np.array(isovf_set)
        isovf_set = np.expand_dims(isovf_set, axis=3)

        od_set = np.array(od_set)
        od_set = np.expand_dims(od_set, axis=3)

    if model_name == 'DKI':
        ak_set = np.array(ak_set)
        ak_set = np.expand_dims(ak_set, axis=3)

        mk_set = np.array(mk_set)
        mk_set = np.expand_dims(mk_set, axis=3)

        rk_set = np.array(rk_set)
        rk_set = np.expand_dims(rk_set, axis=3)

    if model_name == 'NODDI':
        gt_set = np.concatenate([icvf_set, isovf_set, od_set], 3)

    if model_name == 'DKI':
        gt_set = np.concatenate([ak_set, mk_set, rk_set], 3)

    np.save(save_hgt_data_path, section_set)
    print(section_set.shape)
    np.save(save_hgt_gt_data_path, gt_set)
    print(gt_set.shape)

if __name__ == '__main__':
    base_path = ''
    subject_id = ['748662', '751348', '859671', '761957', '833148',
                  '837560', '845458', '896778', '898176', '100610',
                  '102311', '102816', '104416', '105923', '108323',
                  '109123', '599671', '613538', '622236', '654754',
                  '672756', '677968', '680957', '685058', '111312',
                  '111514', '114823', '125525', '130518', '144226',
                  '177746', '185442', '195041', '200614', '204521',
                  '146129', '158035', '562345', '627549', '783462',
                  '896879', '683256', '899885']
    train_subject_id = ['748662', '751348', '859671', '761957', '833148',
                        '837560', '845458', '896778', '898176', '100610',
                        '102311', '102816', '104416', '105923', '108323',
                        '109123', '599671', '613538', '622236', '654754',
                        '125525',  '683256', '899885']
    test_subject_id =  [
                        '672756', '677968', '680957', '685058', '111312',
                        '111514', '114823', '130518', '144226', '896879',
                        '177746', '185442', '195041', '200614', '204521',
                        '146129', '158035', '562345', '627549', '783462']

    gradient_direction = 30
    file_location_name = 'HCP_NODDI_train'
    data_name = 'data.nii.gz'
    mask_name = 'brain_mask.nii.gz'
    bvals_name = 'bvals'
    icvf_name = 'AMICO/NODDI/FIT_ICVF.nii.gz'
    isovf_name = 'AMICO/NODDI/FIT_ISOVF.nii.gz'
    od_name = 'AMICO/NODDI/FIT_OD.nii.gz'
    ak_name = 'DKI/ak.nii.gz'
    mk_name = 'DKI/mk.nii.gz'
    rk_name = 'DKI/rk.nii.gz'

    data_path = base_path + file_location_name
    brain_max_lenght_name = 'hcp_brain_max_lenght.npy'
    brain_max_lenght_path = '' #base_path + brain_max_lenght_name
    model_name = 'NODDI'
    out_name = 'ght_data_' + str(gradient_direction) + '_' + model_name +'_1shell.npy'
    if os.path.exists(brain_max_lenght_path):
        list_lenght = np.load(brain_max_lenght_path)
        xx_min, xx_max, yy_min, yy_max = list_lenght[0], list_lenght[1], list_lenght[2], list_lenght[3]
    else:
        xx_min, xx_max, yy_min, yy_max = calculate_brain_max_lenght(subject_id, data_path, mask_name)
        list_lenght = []
        list_lenght.append(xx_min)
        list_lenght.append(xx_max)
        list_lenght.append(yy_min)
        list_lenght.append(yy_max)
        np.save(brain_max_lenght_name, np.array(list_lenght))
    print('load lenght')
    for file_name_id in train_subject_id:
        is_train = True
        dmri_file_path = base_path + file_location_name + '/' + file_name_id + '/' + data_name
        mask_file_path = base_path + file_location_name + '/' + file_name_id + '/' + mask_name
        bvals_file_path = base_path + file_location_name + '/' + file_name_id + '/' + bvals_name
        if model_name == 'NODDI':
            index1_file_path = base_path + file_location_name + '/' + file_name_id + '/' + icvf_name
            index2_file_path = base_path + file_location_name + '/' + file_name_id + '/' + isovf_name
            index3_file_path = base_path + file_location_name + '/' + file_name_id + '/' + od_name
        if model_name == 'DKI':
            index1_file_path = base_path + file_location_name + '/' + file_name_id + '/' + ak_name
            index2_file_path = base_path + file_location_name + '/' + file_name_id + '/' + mk_name
            index3_file_path = base_path + file_location_name + '/' + file_name_id + '/' + rk_name
        save_hgt_train_data_path = base_path + file_location_name + '/' + file_name_id + '/train_' + out_name
        save_hgt_gt_data_path = base_path + file_location_name + '/' + file_name_id + '/train_gt_' + out_name
        prepare_data(gradient_direction, is_train, model_name, list_lenght, dmri_file_path, mask_file_path, bvals_file_path, index1_file_path, index2_file_path, index3_file_path, save_hgt_train_data_path, save_hgt_gt_data_path)
    for file_name_id in test_subject_id:
        is_train = False
        dmri_file_path = base_path + file_location_name + '/' + file_name_id + '/' + data_name
        mask_file_path = base_path + file_location_name + '/' + file_name_id + '/' + mask_name
        bvals_file_path = base_path + file_location_name + '/' + file_name_id + '/' + bvals_name
        if model_name == 'NODDI':
            index1_file_path = base_path + file_location_name + '/' + file_name_id + '/' + icvf_name
            index2_file_path = base_path + file_location_name + '/' + file_name_id + '/' + isovf_name
            index3_file_path = base_path + file_location_name + '/' + file_name_id + '/' + od_name
        if model_name == 'DKI':
            index1_file_path = base_path + file_location_name + '/' + file_name_id + '/' + ak_name
            index2_file_path = base_path + file_location_name + '/' + file_name_id + '/' + mk_name
            index3_file_path = base_path + file_location_name + '/' + file_name_id + '/' + rk_name
        save_hgt_train_data_path = base_path + file_location_name + '/' + file_name_id + '/test_' + out_name
        save_hgt_gt_data_path = base_path + file_location_name + '/' + file_name_id + '/test_gt_' + out_name
        prepare_data(gradient_direction, is_train, model_name, list_lenght, dmri_file_path, mask_file_path,
                     bvals_file_path, index1_file_path, index2_file_path, index3_file_path, save_hgt_train_data_path,
                     save_hgt_gt_data_path)





