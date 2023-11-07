import numpy as np
import configparser
from lib.predifineGraph import get_adjacency_matrix, load_pickle, weight_matrix
import torch
import pandas as pd

def parse_args(DATASET, parser):
    # get configuration
    config_file = '../conf/TGCN/{}.conf'.format(DATASET)
    config = configparser.ConfigParser()
    config.read(config_file)

    # data
    parser.add_argument('--num_nodes', type=int, default=config['data']['num_nodes'])
    parser.add_argument('--input_window', type=int, default=config['data']['input_window'])
    parser.add_argument('--output_window', type=int, default=config['data']['output_window'])
    # model
    parser.add_argument('--rnn_units', type=int, default=config['model']['rnn_units'])
    parser.add_argument('--lam', type=float, default=config['model']['lam'])
    parser.add_argument('--output_dim', type=int, default=config['model']['output_dim'])
    # train
    parser.add_argument('--seed', type=int, default=config['train']['seed'])
    parser.add_argument('--seed_mode', type=eval, default=config['train']['seed_mode'])
    parser.add_argument('--xavier', type=eval, default=config['train']['xavier'])
    parser.add_argument('--loss_func', type=str, default=config['train']['loss_func'])

    args, _ = parser.parse_known_args()
    args.filepath = '../data/' + DATASET +'/'
    args.filename = DATASET

    if DATASET == 'METR_LA':
        sensor_ids, sensor_id_to_ind, A = load_pickle(pickle_file=args.filepath + 'adj_mx.pkl')
    else:
        if DATASET == 'NYC_BIKE' or DATASET == 'NYC_TAXI':
            A = pd.read_csv(args.filepath + DATASET + '.csv', header=None).values.astype(np.float32)
        else:
            A, Distance = get_adjacency_matrix(
                distance_df_filename=args.filepath + DATASET + '.csv',
                num_of_vertices=args.num_nodes)

    args.adj_mx = A
    return args