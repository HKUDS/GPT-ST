import numpy as np
import configparser
import torch
from scipy.sparse.linalg import eigs
from lib.predifineGraph import load_pickle, weight_matrix, get_adjacency_matrix
import pandas as pd


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])

def parse_args(DATASET, parser):
    # get configuration
    config_file = '../conf/ST-WA/{}.conf'.format(DATASET)
    config = configparser.ConfigParser()
    config.read(config_file)

    # data
    parser.add_argument('--data', default=DATASET, help='data path', type=str, )
    parser.add_argument('--num_nodes', type=int, default=config['data']['num_nodes'])
    parser.add_argument('--lag', type=int, default=config['data']['lag'])
    parser.add_argument('--horizon', type=int, default=config['data']['horizon'])
    # model
    parser.add_argument('--in_dim', type=int, default=config['model']['in_dim'])
    parser.add_argument('--out_dim', type=int, default=config['model']['out_dim'])
    parser.add_argument('--channels', type=int, default=config['model']['channels'])
    parser.add_argument('--dynamic', type=str, default=config['model']['dynamic'])
    parser.add_argument('--memory_size', type=int, default=config['model']['memory_size'])
    parser.add_argument('--column_wise', type=bool, default=False)
    # train
    parser.add_argument('--seed', type=int, default=config['train']['seed'])
    parser.add_argument('--seed_mode', type=eval, default=config['train']['seed_mode'])
    parser.add_argument('--xavier', type=eval, default=config['train']['xavier'])
    parser.add_argument('--loss_func', type=str, default=config['train']['loss_func'])
    args, _ = parser.parse_known_args()
    args.filepath = '../data/' + DATASET + '/'
    args.filename = DATASET

    if DATASET == 'METR_LA':
        sensor_ids, sensor_id_to_ind, A = load_pickle(pickle_file=args.filepath + 'adj_mx.pkl')
    elif DATASET == 'NYC_BIKE' or DATASET == 'NYC_TAXI':
        A = pd.read_csv(args.filepath + DATASET + '.csv', header=None).values.astype(np.float32)
    else:
        A, Distance = get_adjacency_matrix(
            distance_df_filename=args.filepath + DATASET + '.csv',
            num_of_vertices=args.num_nodes)

    adj_mx = scaled_Laplacian(A)
    adj_mx = [adj_mx]
    args.adj_mx = adj_mx

    return args