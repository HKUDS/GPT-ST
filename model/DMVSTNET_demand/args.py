import numpy as np
import configparser
import torch
import pandas as pd

def parse_args(DATASET, parser):
    if DATASET != 'NYC_TAXI' and DATASET != 'NYC_BIKE':
        raise ValueError('Demand prediction baseline. Please select NYC_TAXI or NYC_BIKE dataset!')

    # get configuration
    config_file = '../conf/DMVSTNET_demand/{}.conf'.format(DATASET)
    config = configparser.ConfigParser()
    config.read(config_file)

    # data
    parser.add_argument('--input_window', type=int, default=config['data']['input_window'])
    parser.add_argument('--output_window', type=int, default=config['data']['output_window'])
    parser.add_argument('--num_nodes', type=int, default=config['data']['num_nodes'])
    # model
    parser.add_argument('--hidden_dim', type=int, default=config['model']['hidden_dim'])
    parser.add_argument('--topo_embedded_dim', type=int, default=config['model']['topo_embedded_dim'])
    # train
    parser.add_argument('--seed', type=int, default=config['train']['seed'])
    parser.add_argument('--seed_mode', type=eval, default=config['train']['seed_mode'])
    parser.add_argument('--xavier', type=eval, default=config['train']['xavier'])
    parser.add_argument('--loss_func', type=str, default=config['train']['loss_func'])
    args, _ = parser.parse_known_args()

    args.filepath = '../data/' + DATASET +'/'
    args.filename = DATASET

    A = pd.read_csv( args.filepath + DATASET + ".csv", header=None).values.astype(np.float32)
    args.adj_mx = torch.FloatTensor(A)
    return args