import numpy as np
import configparser
import torch
import pandas as pd
import h5py
from scipy.spatial.distance import cdist
from CCRNN_demand import normalization

# def normalized_laplacian(w: np.ndarray) -> sp.coo_matrix:
#     w = sp.coo_matrix(w)
#     d = np.array(w.sum(1))
#     d_inv_sqrt = np.power(d, -0.5).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#     return sp.eye(w.shape[0]) - w.dot(d_mat_inv_sqrt).T.dot(d_mat_inv_sqrt).tocoo()

def normalized_laplacian(w: np.ndarray) -> np.matrix:
    d = np.array(w.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    print(d,d_inv_sqrt)
    d_mat_inv_sqrt = np.eye(d_inv_sqrt.shape[0]) * d_inv_sqrt.shape
    return np.identity(w.shape[0]) - d_mat_inv_sqrt.dot(w).dot(d_mat_inv_sqrt)


# def random_walk_matrix(w) -> sp.coo_matrix:
#     w = sp.coo_matrix(w)
#     d = np.array(w.sum(1))
#     d_inv = np.power(d, -1).flatten()
#     d_inv[np.isinf(d_inv)] = 0.
#     d_mat_inv = sp.diags(d_inv)
#     return d_mat_inv.dot(w).tocoo()
def random_walk_matrix(w) -> np.matrix:
    d = np.array(w.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = np.eye(d_inv.shape[0]) * d_inv
    return d_mat_inv.dot(w)

def preprocessing_for_metric(data_category: list,
                             dataset:str,
                             hidden_size:int,
                             Normal_Method: str,
                             _len: list,
                             normalized_category):
    data = []
    normal_method = getattr(normalization, Normal_Method)
    for category in data_category:
        normal = normal_method()
        with h5py.File(f"../data/{dataset}/{category}_data.h5", 'r') as hf:
            data_pick = hf[f'{category}_pick'][:]
        with h5py.File(f"../data/{dataset}/{category}_data.h5", 'r') as hf:
            data_drop = hf[f'{category}_drop'][:]
        data.append(normal.fit_transform(np.stack([data_pick, data_drop], axis=2)))


    data = np.concatenate(data, axis=1).transpose((0,2,1))
    data = data[:-(_len[0]+_len[1])]
    T, input_dim, N = data.shape
    inputs = data.reshape(-1, N)
    u, s, v = np.linalg.svd(inputs)
    w = np.diag(s[:hidden_size]).dot(v[:hidden_size,:]).T



    graph = cdist(w, w, metric='euclidean')
    support = graph * -1 / np.std(graph) ** 2
    support = np.exp(support)

    support = support - np.identity(support.shape[0])
    if normalized_category == 'randomwalk':
        support = random_walk_matrix(support)
    elif normalized_category == 'laplacian':
        support = normalized_laplacian(support)

    return support


def parse_args(DATASET, parser):
    if DATASET != 'NYC_TAXI' and DATASET != 'NYC_BIKE':
        raise ValueError('Demand prediction baseline. Please select NYC_TAXI or NYC_BIKE dataset!')

    # get configuration
    config_file = '../conf/CCRNN_demand/{}.conf'.format(DATASET)
    config = configparser.ConfigParser()
    config.read(config_file)

    # data
    parser.add_argument('--Normal_Method', type=str, default=config['data']['Normal_Method'])
    parser.add_argument('--normalized_category', type=str, default=config['data']['normalized_category'])
    parser.add_argument('--hidden_size_data', type=int, default=config['data']['hidden_size_data'])
    parser.add_argument('--num_input', type=int, default=config['data']['num_input'])
    parser.add_argument('--num_predict', type=int, default=config['data']['num_predict'])
    parser.add_argument('--num_nodes', type=int, default=config['data']['num_nodes'])
    # model
    parser.add_argument('--hidden_size', type=int, default=config['model']['hidden_size'])
    parser.add_argument('--n_dim', type=int, default=config['model']['n_dim'])
    parser.add_argument('--n_supports', type=int, default=config['model']['n_supports'])
    parser.add_argument('--k_hop', type=int, default=config['model']['k_hop'])
    parser.add_argument('--n_rnn_layers', type=int, default=config['model']['n_rnn_layers'])
    parser.add_argument('--n_gconv_layers', type=int, default=config['model']['n_gconv_layers'])
    parser.add_argument('--input_dim', type=int, default=config['model']['input_dim'])
    parser.add_argument('--output_dim', type=int, default=config['model']['output_dim'])
    parser.add_argument('--cl_decay_steps', type=int, default=config['model']['cl_decay_steps'])
    # train
    parser.add_argument('--seed', type=int, default=config['train']['seed'])
    parser.add_argument('--seed_mode', type=eval, default=config['train']['seed_mode'])
    parser.add_argument('--xavier', type=eval, default=config['train']['xavier'])
    parser.add_argument('--loss_func', type=str, default=config['train']['loss_func'])
    args, _ = parser.parse_known_args()

    if DATASET == 'NYC_TAXI':
        _len = [850, 850]
        support = preprocessing_for_metric(['taxi'], 'NYC_TAXI', args.hidden_size_data, args.Normal_Method, _len, args.normalized_category)
    elif DATASET == 'NYC_BIKE':
        _len = [850, 850]
        support = preprocessing_for_metric(['bike'], 'NYC_BIKE', args.hidden_size_data, args.Normal_Method, _len, args.normalized_category)
    else:
        raise ValueError
    args.support = torch.FloatTensor(support)
    return args