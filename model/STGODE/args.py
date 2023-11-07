import numpy as np
import os
from fastdtw import fastdtw
from tqdm import tqdm
import csv
import torch
import pandas as pd
import configparser

def read_data(args):
    """read data, generate spatial adjacency matrix and semantic adjacency matrix by dtw

    Args:
        sigma1: float, default=0.1, sigma for the semantic matrix
        sigma2: float, default=10, sigma for the spatial matrix
        thres1: float, default=0.6, the threshold for the semantic matrix
        thres2: float, default=0.5, the threshold for the spatial matrix

    Returns:
        data: tensor, T * N * 1
        dtw_matrix: array, semantic adjacency matrix
        sp_matrix: array, spatial adjacency matrix
    """
    filename = args.filename
    # filepath = "./data/"
    filepath = args.filepath
    # if args.remote:
    #     filepath = '/home/lantu.lqq/ftemp/data/'
    data = np.load(filepath + filename + '.npz')['data']
    if data.ndim != 3:
        data = np.expand_dims(data, axis=-1)
    # print(data.shape)
    # PEMS04 == shape: (16992, 307, 3)    feature: flow,occupy,speed
    # PEMSD7M == shape: (12672, 228, 1)
    # PEMSD7L == shape: (12672, 1026, 1)
    num_node = data.shape[1]
    mean_value = np.mean(data, axis=(0, 1)).reshape(1, 1, -1)
    std_value = np.std(data, axis=(0, 1)).reshape(1, 1, -1)
    data = (data - mean_value) / std_value
    mean_value = mean_value.reshape(-1)[0]
    std_value = std_value.reshape(-1)[0]
    # print(sss)

    if not os.path.exists(f'../data/STGODE/{filename}/{filename}_dtw_distance.npy'):
        data_mean = np.mean([data[:, :, 0][24 * 12 * i: 24 * 12 * (i + 1)] for i in range(data.shape[0] // (24 * 12))],
                            axis=0)
        data_mean = data_mean.squeeze().T
        dtw_distance = np.zeros((num_node, num_node))
        for i in tqdm(range(num_node)):
            for j in range(i, num_node):
                dtw_distance[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]
        for i in range(num_node):
            for j in range(i):
                dtw_distance[i][j] = dtw_distance[j][i]
        np.save(f'../data/STGODE/{filename}/{filename}_dtw_distance.npy', dtw_distance)

    dist_matrix = np.load(f'../data/STGODE/{filename}/{filename}_dtw_distance.npy')

    mean = np.mean(dist_matrix)
    std = np.std(dist_matrix)
    dist_matrix = (dist_matrix - mean) / std
    sigma = args.sigma1
    dist_matrix = np.exp(-dist_matrix ** 2 / sigma ** 2)
    dtw_matrix = np.zeros_like(dist_matrix)
    dtw_matrix[dist_matrix > args.thres1] = 1

    # # use continuous semantic matrix
    # if not os.path.exists(f'data/{filename}_dtw_c_matrix.npy'):
    #     dist_matrix = np.load(f'data/{filename}_dtw_distance.npy')
    #     # normalization
    #     std = np.std(dist_matrix[dist_matrix != np.float('inf')])
    #     mean = np.mean(dist_matrix[dist_matrix != np.float('inf')])
    #     dist_matrix = (dist_matrix - mean) / std
    #     sigma = 0.1
    #     dtw_matrix = np.exp(- dist_matrix**2 / sigma**2)
    #     dtw_matrix[dtw_matrix < 0.5] = 0
    #     np.save(f'data/{filename}_dtw_c_matrix.npy', dtw_matrix)
    # dtw_matrix = np.load(f'data/{filename}_dtw_c_matrix.npy')

    # use continuous spatial matrix
    if not os.path.exists(f'../data/STGODE/{filename}/{filename}_spatial_distance.npy'):
        with open(filepath + filename + '.csv', 'r') as fp:
            if filename == 'PEMS07M' or filename == 'METR_LA':
                dist_matrix = pd.read_csv(filepath + filename + '.csv', header=None).values
                print(dist_matrix.shape)
            elif filename == 'SZ_TAXI':
                dist_matrix = pd.read_csv(filepath + filename + '.csv', header=None).values.astype(np.float32)
                dist_matrix[dist_matrix==0] = 1000
                dist_matrix[dist_matrix==1] = 5
                print(dist_matrix)
                # print(sss)
            elif filename == 'NYC_TAXI' or filename == 'NYC_BIKE':
                dist_matrix = pd.read_csv(filepath + filename + '.csv', header=None).values.astype(np.float32)
                dist_matrix = (1 - dist_matrix) * 1000
                # dist_matrix[dist_matrix==0] = 1000
                # dist_matrix[dist_matrix==1] = 5
            else:
                dist_matrix = np.zeros((num_node, num_node)) + np.float('inf')
                file = csv.reader(fp)
                for line in file:
                    break
                for line in file:
                    print(line[0])
                    start = int(line[0])
                    end = int(line[1])
                    dist_matrix[start][end] = float(line[2])
                    dist_matrix[end][start] = float(line[2])
            np.save(f'../data/STGODE/{filename}/{filename}_spatial_distance.npy', dist_matrix)

    # use 0/1 spatial matrix
    # if not os.path.exists(f'data/{filename}_sp_matrix.npy'):
    #     dist_matrix = np.load(f'data/{filename}_spatial_distance.npy')
    #     sp_matrix = np.zeros((num_node, num_node))
    #     sp_matrix[dist_matrix != np.float('inf')] = 1
    #     np.save(f'data/{filename}_sp_matrix.npy', sp_matrix)
    # sp_matrix = np.load(f'data/{filename}_sp_matrix.npy')

    dist_matrix = np.load(f'../data/STGODE/{filename}/{filename}_spatial_distance.npy')
    # normalization
    std = np.std(dist_matrix[dist_matrix != np.float('inf')])
    mean = np.mean(dist_matrix[dist_matrix != np.float('inf')])
    dist_matrix = (dist_matrix - mean) / std
    sigma = args.sigma2
    sp_matrix = np.exp(- dist_matrix ** 2 / sigma ** 2)
    sp_matrix[sp_matrix < args.thres2] = 0
    # np.save(f'data/{filename}_sp_c_matrix.npy', sp_matrix)
    # sp_matrix = np.load(f'data/{filename}_sp_c_matrix.npy')

    print(f'average degree of spatial graph is {np.sum(sp_matrix > 0) / 2 / num_node}')
    print(f'average degree of semantic graph is {np.sum(dtw_matrix > 0) / 2 / num_node}')
    return torch.from_numpy(data.astype(np.float32)), mean_value, std_value, dtw_matrix, sp_matrix

def get_normalized_adj(A):
    """
    Returns a tensor, the degree normalized adjacency matrix.
    """
    alpha = 0.8
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    A_reg = alpha / 2 * (np.eye(A.shape[0]) + A_wave)
    return torch.from_numpy(A_reg.astype(np.float32))


def parse_args(DATASET, parser):
    # get configuration
    config_file = '../conf/STGODE/{}.conf'.format(DATASET)
    config = configparser.ConfigParser()
    config.read(config_file)
    out_channels_str = config.get('model', 'out_channels')
    out_channels = eval(out_channels_str)

    # data
    parser.add_argument('--num_nodes', type=int, default=config['data']['num_nodes'])
    parser.add_argument('--num_timesteps_input', type=int, default=config['data']['num_timesteps_input'])
    parser.add_argument('--num_timesteps_output', type=int, default=config['data']['num_timesteps_output'])
    # model
    parser.add_argument('--sigma1', type=float, default=config['model']['sigma1'], help='sigma for the semantic matrix')
    parser.add_argument('--sigma2', type=float, default=config['model']['sigma2'], help='sigma for the spatial matrix')
    parser.add_argument('--thres1', type=float, default=config['model']['thres1'], help='the threshold for the semantic matrix')
    parser.add_argument('--thres2', type=float, default=config['model']['thres2'], help='the threshold for the spatial matrix')
    parser.add_argument('--in_channels', type=int, default=config['model']['in_channels'])
    parser.add_argument('--out_channels', type=list, default=out_channels)
    parser.add_argument('--n_layers', type=int, default=config['model']['n_layers'])
    # train
    parser.add_argument('--seed', type=int, default=config['train']['seed'])
    parser.add_argument('--seed_mode', type=eval, default=config['train']['seed_mode'])
    parser.add_argument('--xavier', type=eval, default=config['train']['xavier'])
    parser.add_argument('--loss_func', type=str, default=config['train']['loss_func'])

    args, _ = parser.parse_known_args()
    args.filepath = '../data/' + DATASET +'/'
    args.filename = DATASET
    _, _, _, dtw_matrix, sp_matrix = read_data(args)
    args.A_sp_wave, args.A_se_wave = get_normalized_adj(sp_matrix), get_normalized_adj(dtw_matrix)
    return args



