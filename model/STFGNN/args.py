import numpy as np
import configparser
from lib.load_dataset import load_st_dataset
from lib.predifineGraph import get_adjacency_matrix, load_pickle, weight_matrix
import os
# from fastdtw import fastdtw
import torch
import pandas as pd

def gen_data(data, ntr, N, DATASET):
    '''
    if flag:
        data=pd.read_csv(fname)
    else:
        data=pd.read_csv(fname,header=None)
    '''
    #data=data.as_matrix()
    if DATASET == 'SZ_TAXI':
        data=np.reshape(data,[-1,96,N])
    elif DATASET == 'NYC_TAXI' or DATASET == 'NYC_BIKE':
        data = np.reshape(data, [-1, 48, N])
    else:
        data = np.reshape(data, [-1, 288, N])
    return data[0:ntr]

def normalize(a):
    mu=np.mean(a,axis=1,keepdims=True)
    std=np.std(a,axis=1,keepdims=True)
    return (a-mu)/std

def compute_dtw(a,b,order=1,Ts=12,normal=True):
    if normal:
        a=normalize(a)
        b=normalize(b)
    T0=a.shape[1]
    d=np.reshape(a,[-1,1,T0])-np.reshape(b,[-1,T0,1])
    d=np.linalg.norm(d,axis=0,ord=order)
    D=np.zeros([T0,T0])
    for i in range(T0):
        for j in range(max(0,i-Ts),min(T0,i+Ts+1)):
            if (i==0) and (j==0):
                D[i,j]=d[i,j]**order
                continue
            if (i==0):
                D[i,j]=d[i,j]**order+D[i,j-1]
                continue
            if (j==0):
                D[i,j]=d[i,j]**order+D[i-1,j]
                continue
            if (j==i-Ts):
                D[i,j]=d[i,j]**order+min(D[i-1,j-1],D[i-1,j])
                continue
            if (j==i+Ts):
                D[i,j]=d[i,j]**order+min(D[i-1,j-1],D[i,j-1])
                continue
            D[i,j]=d[i,j]**order+min(D[i-1,j-1],D[i-1,j],D[i,j-1])
    return D[-1,-1]**(1.0/order)

def construct_dtw(data, DATASET):
    data = data[:, :, 0]
    if DATASET == 'SZ_TAXI':
        total_day = data.shape[0] / 96
    else:
        total_day = data.shape[0] / 288
    tr_day = int(total_day * 0.6)
    n_route = data.shape[1]
    xtr = gen_data(data, tr_day, n_route, DATASET)
    print(np.shape(xtr))
    T0 = 288
    T = 12
    N = n_route
    d = np.zeros([N, N])
    for i in range(N):
        for j in range(i + 1, N):
            d[i, j] = compute_dtw(xtr[:, :, i], xtr[:, :, j])

    print("The calculation of time series is done!")
    dtw = d + d.T
    n = dtw.shape[0]
    w_adj = np.zeros([n, n])
    adj_percent = 0.01
    top = int(n * adj_percent)
    for i in range(dtw.shape[0]):
        a = dtw[i, :].argsort()[0:top]
        for j in range(top):
            w_adj[i, a[j]] = 1

    for i in range(n):
        for j in range(n):
            if (w_adj[i][j] != w_adj[j][i] and w_adj[i][j] == 0):
                w_adj[i][j] = 1
            if (i == j):
                w_adj[i][j] = 1

    print("Total route number: ", n)
    print("Sparsity of adj: ", len(w_adj.nonzero()[0]) / (n * n))
    print("The weighted matrix of temporal graph is generated!")
    dtw = w_adj
    return dtw

def construct_adj_fusion(A, A_dtw, steps):
    '''
    construct a bigger adjacency matrix using the given matrix

    Parameters
    ----------
    A: np.ndarray, adjacency matrix, shape is (N, N)

    steps: how many times of the does the new adj mx bigger than A

    Returns
    ----------
    new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)

    ----------
    This is 4N_1 mode:

    [T, 1, 1, T
     1, S, 1, 1
     1, 1, S, 1
     T, 1, 1, T]

    '''

    N = len(A)
    adj = np.zeros([N * steps] * 2) # "steps" = 4 !!!

    for i in range(steps):
        if (i == 1) or (i == 2):
            adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A
        else:
            adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A_dtw
    #'''
    for i in range(N):
        for k in range(steps - 1):
            adj[k * N + i, (k + 1) * N + i] = 1
            adj[(k + 1) * N + i, k * N + i] = 1
    #'''
    adj[3 * N: 4 * N, 0:  N] = A_dtw #adj[0 * N : 1 * N, 1 * N : 2 * N]
    adj[0 : N, 3 * N: 4 * N] = A_dtw #adj[0 * N : 1 * N, 1 * N : 2 * N]

    adj[2 * N: 3 * N, 0 : N] = adj[0 * N : 1 * N, 1 * N : 2 * N]
    adj[0 : N, 2 * N: 3 * N] = adj[0 * N : 1 * N, 1 * N : 2 * N]
    adj[1 * N: 2 * N, 3 * N: 4 * N] = adj[0 * N : 1 * N, 1 * N : 2 * N]
    adj[3 * N: 4 * N, 1 * N: 2 * N] = adj[0 * N : 1 * N, 1 * N : 2 * N]


    for i in range(len(adj)):
        adj[i, i] = 1

    return adj


def parse_args(DATASET, args_base, parser):
    # get configuration
    config_file = '../conf/STFGNN/{}.conf'.format(DATASET)
    config = configparser.ConfigParser()
    config.read(config_file)

    hidden_dims_str = config.get('model', 'hidden_dims')
    hidden_dims = eval(hidden_dims_str)
    # hidden_dims = [[int(j) for j in i.split(',')] for i in hidden_dims_str.strip().split('\n')]

    # data
    parser.add_argument('--num_nodes', type=int, default=config['data']['num_nodes'])
    parser.add_argument('--window', type=int, default=config['data']['window'])
    parser.add_argument('--horizon', type=int, default=config['data']['horizon'])
    parser.add_argument('--order', type=int, default=config['data']['order'])
    parser.add_argument('--lag', type=int, default=config['data']['lag'])
    parser.add_argument('--period', type=int, default=config['data']['period'])
    parser.add_argument('--sparsity', type=float, default=config['data']['sparsity'])
    # model
    parser.add_argument('--hidden_dims', type=list, default=hidden_dims)
    parser.add_argument('--first_layer_embedding_size', type=int, default=config['model']['first_layer_embedding_size'])
    parser.add_argument('--out_layer_dim', type=int, default=config['model']['out_layer_dim'])
    parser.add_argument('--output_dim', type=int, default=config['model']['output_dim'])
    parser.add_argument('--strides', type=int, default=config['model']['strides'])
    parser.add_argument('--temporal_emb', type=eval, default=config['model']['temporal_emb'])
    parser.add_argument('--spatial_emb', type=eval, default=config['model']['spatial_emb'])
    parser.add_argument('--use_mask', type=eval, default=config['model']['use_mask'])
    parser.add_argument('--activation', type=str, default=config['model']['activation'])
    parser.add_argument('--module_type', type=str, default=config['model']['module_type'])
    # train
    parser.add_argument('--seed', type=int, default=config['train']['seed'])
    parser.add_argument('--seed_mode', type=eval, default=config['train']['seed_mode'])
    parser.add_argument('--xavier', type=eval, default=config['train']['xavier'])
    parser.add_argument('--loss_func', type=str, default=config['train']['loss_func'])

    args, _ = parser.parse_known_args()
    args.filepath = '../data/' + DATASET +'/'
    args.filename = DATASET
    data = load_st_dataset(DATASET, args_base)

    filename = DATASET

    if not os.path.exists(f'../data/STFGNN/{filename}/{filename}_adj_mx.npy'):
        if DATASET == 'METR_LA':
            sensor_ids, sensor_id_to_ind, A = load_pickle(pickle_file=args.filepath + 'adj_mx.pkl')
        elif DATASET == 'NYC_TAXI' or DATASET == 'NYC_BIKE':
            A = pd.read_csv(args.filepath + DATASET + ".csv", header=None).values.astype(np.float32)
        else:
            A, Distance = get_adjacency_matrix(
                distance_df_filename=args.filepath + DATASET + '.csv',
                num_of_vertices=args.num_nodes)
        dtw = construct_dtw(data, DATASET)
        adj_mx = construct_adj_fusion(A, dtw, args.strides)
        np.save(f'../data/STFGNN/{filename}/{filename}_adj_mx.npy', adj_mx)

    adj_STFGNN = np.load(f'../data/STFGNN/{filename}/{filename}_adj_mx.npy')
    args.adj = torch.Tensor(adj_STFGNN)
    return args