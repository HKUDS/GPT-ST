import numpy as np
import pandas as pd
import configparser
from lib.predifineGraph import get_adjacency_matrix, load_pickle, weight_matrix

def parse_args(DATASET, parser):
    # get configuration
    config_file = '../conf/GWN/{}.conf'.format(DATASET)
    config = configparser.ConfigParser()
    config.read(config_file)

    # data
    parser.add_argument('--num_nodes', type=int, default=config['data']['num_nodes'])
    parser.add_argument('--input_window', type=int, default=config['data']['input_window'])
    parser.add_argument('--output_window', type=int, default=config['data']['output_window'])
    parser.add_argument('--output_dim', type=int, default=config['data']['output_dim'])
    # model
    parser.add_argument('--dropout', type=float, default=config['model']['dropout'])
    parser.add_argument('--blocks', type=int, default=config['model']['blocks'])
    parser.add_argument('--layers', type=int, default=config['model']['layers'])
    parser.add_argument('--gcn_bool', type=eval, default=config['model']['gcn_bool'])
    parser.add_argument('--addaptadj', type=eval, default=config['model']['addaptadj'])
    parser.add_argument('--adjtype', type=str, default=config['model']['adjtype'])
    parser.add_argument('--randomadj', type=eval, default=config['model']['randomadj'])
    parser.add_argument('--aptonly', type=eval, default=config['model']['aptonly'])
    parser.add_argument('--kernel_size', type=int, default=config['model']['kernel_size'])
    parser.add_argument('--nhid', type=int, default=config['model']['nhid'])
    parser.add_argument('--residual_channels', type=int, default=config['model']['residual_channels'])
    parser.add_argument('--dilation_channels', type=int, default=config['model']['dilation_channels'])
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

    args.adj_mx = A
    return args