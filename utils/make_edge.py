import os
from tqdm import tqdm
import torch
import numpy as np

def ReadBValAndBVec(bvalfile, bvecfile):
    # Read bval file
    bvalf = open(bvalfile, 'r')
    bvalstr = bvalf.readline()
    bvalarr = np.fromstring(bvalstr, dtype=int, sep=' ')
    bvalf.close()
    # Read bvec file
    bvecf = open(bvecfile, 'r')
    bveclines = bvecf.readlines()
    if len(bveclines) < 3:
        print('bvec file is not completed.')
        return 0,0

    linecount = 0
    bvecmat = np.zeros([3, len(bvalarr)])
    for l in range(len(bveclines)):
        line = bveclines[l]
        if (len(line) == 0):
            continue
        bvecarr = np.fromstring(line, dtype=np.float32, sep=' ')
        if len(bvecarr) != len(bvalarr):
            print('Length of bvec array is not same as that of bval array')
            return 0,0
        bvecmat[linecount, :] = bvecarr
        linecount += 1
        if (linecount == 3):
            break
    return bvalarr, bvecmat

def edge_index_from_adj(adj):
    # print(num_nodes)
    edge = []
    # edge_plot = []
    for row in range(0,adj.size(0)):
        for col in range(0,adj.size(1)):
            if (adj[row,col]) == 1:
                edge.append([row, col])
    edge_index = torch.tensor(np.transpose(edge))
    return edge_index

def NormalizeVector(v):
    d = np.sqrt(np.vdot(v, v))
    return v/d

def ConvertBVal(bvalarr, offset = 10, div = 1000):
    bvalarr = bvalarr + offset
    bvalarr = bvalarr/div
    return bvalarr

def RemoveB0Element(bval, bvec):
    noB0Vol = np.count_nonzero(bval)
    newbval = np.zeros(noB0Vol)
    newbvec = np.zeros([3, noB0Vol])

    c = 0
    for q in range(len(bval)):
        if bval[q] == 0:
            continue
        newbval[c] = bval[q]
        newbvec[0, c] = bvec[0, q]
        newbvec[1, c] = bvec[1, q]
        newbvec[2, c] = bvec[2, q]
        c += 1
    return newbval, newbvec

def delete_30_b0(bvals, bvecs):
    select_bvals = []
    select_bvecs = []
    i = 0
    number = 0
    for val in bvals:

        if(val < 1200 and val > 900):
            if(number+1 > 30 ):
                break
            select_bvals.append(val)
            select_bvecs.append(bvecs[:, i])
            number += 1
        i += 1
    return np.array(select_bvals), np.array(select_bvecs)

def delete_60_b0(bvals, bvecs):
    select_bvals = []
    select_bvecs = []
    i = 0
    j = 0
    number = 0
    number1 = 0
    for val in bvals:

        if(val < 1200 and val > 900):
            if(number + 1 > 30 ):
                break
            select_bvals.append(val)
            select_bvecs.append(bvecs[:, i])
            number += 1
        i += 1

    for val in bvals:

        if(val < 2300 and val > 1300):
            if(number1 + 1 > 30 ):
                break
            select_bvals.append(val)
            select_bvecs.append(bvecs[:, j])
            number1 += 1
        j += 1
    return np.array(select_bvals), np.array(select_bvecs)


def calculate_adjacency_matrix(bval, bvec, threshold_angle):
    size = len(bval)
    adj = np.zeros((size, size))
    angle_adj = np.zeros((size, size))
    for x in range(0, size):
        normal1 = NormalizeVector(bvec[x, :])
        b1 = bval[x]
        for y in range(0, size):
            normal2 = NormalizeVector(bvec[y, :])
            b2 = bval[y]
            if(b1 != b2):
                adj[x, y] = 0
                continue
            data_M = np.sqrt(np.sum(normal1 * normal1, axis=0))
            data_N = np.sqrt(np.sum(normal2 * normal2, axis=0))
            cos_theta = np.sum(normal1 * normal2, axis=0) / (data_M * data_N)
            if(cos_theta >= 1):
                theta = np.degrees(0)
            elif(cos_theta <= -1):
                theta = np.degrees(math.pi)
            else:
                theta = np.degrees(np.arccos(cos_theta))
            angle_adj[x, y] = theta
            if(theta < threshold_angle):
                adj[x, y] = 1
            else:
                adj[x, y] = 0
    return adj


def make_edge(data_path, angel_threshold, gradient_direaction, bval_name, bvec_name, image_shape, batch, save_path, model_name):
    number = image_shape[0] * image_shape[1] * batch
    bvalfile = os.path.join(data_path, bval_name)
    bvecfile = os.path.join(data_path, bvec_name)
    bval, bvec = ReadBValAndBVec(bvalfile, bvecfile)
    if gradient_direaction == 60:
        bval, bvec = delete_60_b0(bval, bvec)
    elif gradient_direaction == 30:
        bval, bvec = delete_30_b0(bval, bvec)
    else:
        print('gradient direction is error')
    bval = ConvertBVal(bval, offset=50, div=1000)
    bval = np.round(bval)
    adj = calculate_adjacency_matrix(bval, bvec, angel_threshold)
    adj = torch.tensor(adj)
    eye = torch.eye(adj.size(0))
    one = torch.ones_like(adj)
    mask = one - eye
    adj = adj * mask
    edge_index = edge_index_from_adj(adj)
    if model_name == 'GCNN':
        edge_index = np.array(edge_index, dtype=np.int64)
        print(edge_index.shape)
        edge_index = torch.LongTensor(edge_index)
        return edge_index
    else:
        ori_edge_index = edge_index
        for i in tqdm(range(1, number), total=number - 1, desc='edge'):
            next_edge_index = ori_edge_index + i * gradient_direaction
            edge_index = np.concatenate((edge_index, next_edge_index), axis=1)
        print(edge_index.shape)
        torch.save(edge_index, save_path)