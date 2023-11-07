import numpy as np
import configparser
from lib.predifineGraph import get_adjacency_matrix, load_pickle, weight_matrix
import pandas as pd

def parse_args(DATASET, parser):
    # get configuration
    config_file = '../conf/STSGCN/{}.conf'.format(DATASET)
    config = configparser.ConfigParser()
    config.read(config_file)

    filter_list_str = config.get('model', 'filter_list')
    filter_list = eval(filter_list_str)

    # data
    parser.add_argument('--num_nodes', type=int, default=config['data']['num_nodes'])
    parser.add_argument('--input_window', type=int, default=config['data']['input_window'])
    parser.add_argument('--output_window', type=int, default=config['data']['output_window'])

    # model
    parser.add_argument('--filter_list', type=list, default=config['model']['filter_list'])
    parser.add_argument('--rho', type=int, default=config['model']['rho'])
    parser.add_argument('--feature_dim', type=int, default=config['model']['feature_dim'])
    parser.add_argument('--module_type', type=str, default=config['model']['module_type'])
    parser.add_argument('--activation', type=str, default=config['model']['activation'])
    parser.add_argument('--temporal_emb', type=eval, default=config['model']['temporal_emb'])
    parser.add_argument('--spatial_emb', type=eval, default=config['model']['spatial_emb'])
    parser.add_argument('--use_mask', type=eval, default=config['model']['use_mask'])
    parser.add_argument('--steps', type=int, default=config['model']['steps'])
    parser.add_argument('--first_layer_embedding_size', type=int, default=config['model']['first_layer_embedding_size'])
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
    elif DATASET == 'NYC_TAXI' or DATASET == 'NYC_BIKE':
        A = pd.read_csv(args.filepath + DATASET + ".csv", header=None).values.astype(np.float32)
    else:
        A, Distance = get_adjacency_matrix(
            distance_df_filename=args.filepath + DATASET + '.csv',
            num_of_vertices=args.num_nodes)
    args.filter_list = filter_list
    args.A = A
    return args