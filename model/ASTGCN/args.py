import numpy as np
import configparser
from lib.predifineGraph import get_adjacency_matrix, load_pickle, weight_matrix
import torch
import pandas as pd

def parse_args(DATASET, parser):
    # get configuration
    config_file = '../conf/ASTGCN/{}.conf'.format(DATASET)
    config = configparser.ConfigParser()
    config.read(config_file)

    # data
    parser.add_argument('--num_nodes', type=int, default=config['data']['num_nodes'])
    parser.add_argument('--len_input', type=int, default=config['data']['len_input'])
    parser.add_argument('--num_for_predict', type=int, default=config['data']['num_for_predict'])
    # model
    parser.add_argument('--nb_block', type=int, default=config['model']['nb_block'])
    parser.add_argument('--K', type=int, default=config['model']['K'])
    parser.add_argument('--nb_chev_filter', type=int, default=config['model']['nb_chev_filter'])
    parser.add_argument('--nb_time_filter', type=int, default=config['model']['nb_time_filter'])
    parser.add_argument('--time_strides', type=int, default=config['model']['time_strides'])
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
    elif DATASET == 'NYC_BIKE' or DATASET == 'NYC_TAXI':
        A = pd.read_csv(args.filepath + DATASET + '.csv', header=None).values.astype(np.float32)
    else:
        A, Distance = get_adjacency_matrix(
            distance_df_filename=args.filepath + DATASET + '.csv',
            num_of_vertices=args.num_nodes)
    args.A = A
    return args