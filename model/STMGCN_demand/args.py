import numpy as np
import configparser
import torch
import pandas as pd
from .GCN import Adj_Preprocessor

def parse_args(DATASET, parser):
    if DATASET != 'NYC_TAXI' and DATASET != 'NYC_BIKE':
        raise ValueError('Demand prediction baseline. Please select NYC_TAXI or NYC_BIKE dataset!')

    # get configuration
    config_file = '../conf/STMGCN_demand/{}.conf'.format(DATASET)
    config = configparser.ConfigParser()
    config.read(config_file)
    # data
    parser.add_argument('--seq_len', type=int, default=config['data']['seq_len'])
    parser.add_argument('--M', type=int, default=config['data']['M'])
    parser.add_argument('--n_nodes', type=int, default=config['data']['n_nodes'])
    # model
    parser.add_argument('--lstm_hidden_dim', type=int, default=config['model']['lstm_hidden_dim'])
    parser.add_argument('--lstm_num_layers', type=int, default=config['model']['lstm_num_layers'])
    parser.add_argument('--gcn_hidden_dim', type=int, default=config['model']['gcn_hidden_dim'])
    parser.add_argument('--gconv_use_bias', type=eval, default=config['model']['gconv_use_bias'])
    parser.add_argument('--gconv_activation', type=str, default=config['model']['gconv_activation'])
    # train
    parser.add_argument('--seed', type=int, default=config['train']['seed'])
    parser.add_argument('--seed_mode', type=eval, default=config['train']['seed_mode'])
    parser.add_argument('--xavier', type=eval, default=config['train']['xavier'])
    parser.add_argument('--loss_func', type=str, default=config['train']['loss_func'])
    args, _ = parser.parse_known_args()

    args.filepath = '../data/STMGCN_demand/'
    args.filename = DATASET

    if DATASET == 'NYC_TAXI':
        dis_graph = pd.read_csv(args.filepath + "dis_tt.csv", header=None).values.astype(np.float32)
        pcc_graph = pd.read_csv(args.filepath + "pcc_tt.csv", header=None).values.astype(np.float32)
    elif DATASET == 'NYC_BIKE':
        dis_graph = pd.read_csv(args.filepath + "dis_bb.csv", header=None).values.astype(np.float32)
        pcc_graph = pd.read_csv(args.filepath + "pcc_bb.csv", header=None).values.astype(np.float32)
    else:
        raise ValueError
    args.sta_kernel_config = {'kernel_type': 'chebyshev', 'K': 2}
    # localpool
    adj_preprocessor = Adj_Preprocessor(**args.sta_kernel_config)
    dis_graph = torch.FloatTensor(dis_graph)
    pcc_graph = torch.FloatTensor(pcc_graph)
    dis_graph = adj_preprocessor.process(dis_graph)
    pcc_graph = adj_preprocessor.process(pcc_graph)
    dis_graph = torch.nan_to_num(dis_graph)
    pcc_graph = torch.nan_to_num(pcc_graph)
    args.dis_graph = dis_graph
    args.pcc_graph = pcc_graph
    # args.sta_adj_list = sta_adj_list

    return args