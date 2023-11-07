import argparse
# import numpy as np
import configparser
# import pandas as pd

def parse_args(device):


    # parser
    args = argparse.ArgumentParser(prefix_chars='-', description='pretrain_arguments')
    args.add_argument('-dataset', default='METR_LA', type=str, required=True)
    args.add_argument('-mode', default='ori', type=str, required=True)
    args.add_argument('-device', default=device, type=str, help='indices of GPUs')
    args.add_argument('-model', default='TGCN', type=str)
    args.add_argument('-cuda', default=True, type=bool)

    args_get, _ = args.parse_known_args()

    # get configuration
    config_file = '../conf/GPTST_pretrain/{}.conf'.format(args_get.dataset)
    config = configparser.ConfigParser()
    config.read(config_file)

    # data
    args.add_argument('-val_ratio', default=config['data']['val_ratio'], type=float)
    args.add_argument('-test_ratio', default=config['data']['test_ratio'], type=float)
    args.add_argument('-lag', default=config['data']['lag'], type=int)
    args.add_argument('-horizon', default=config['data']['horizon'], type=int)
    args.add_argument('-num_nodes', default=config['data']['num_nodes'], type=int)
    args.add_argument('-tod', default=config['data']['tod'], type=eval)
    args.add_argument('-normalizer', default=config['data']['normalizer'], type=str)
    args.add_argument('-column_wise', default=config['data']['column_wise'], type=eval)
    args.add_argument('-default_graph', default=config['data']['default_graph'], type=eval)
    # model
    args.add_argument('-input_base_dim', default=config['model']['input_base_dim'], type=int)
    args.add_argument('-input_extra_dim', default=config['model']['input_extra_dim'], type=int)
    args.add_argument('-output_dim', default=config['model']['output_dim'], type=int)
    args.add_argument('-embed_dim', default=config['model']['embed_dim'], type=int)
    args.add_argument('-embed_dim_spa', default=config['model']['embed_dim_spa'], type=int)
    args.add_argument('-hidden_dim', default=config['model']['hidden_dim'], type=int)
    args.add_argument('-HS', default=config['model']['HS'], type=int)
    args.add_argument('-HT', default=config['model']['HT'], type=int)
    args.add_argument('-HT_Tem', default=config['model']['HT_Tem'], type=int)
    args.add_argument('-num_route', default=config['model']['num_route'], type=int)
    args.add_argument('-mask_ratio', default=config['model']['mask_ratio'], type=float)
    args.add_argument('-ada_mask_ratio', default=config['model']['ada_mask_ratio'], type=float)
    args.add_argument('-ada_type', default=config['model']['ada_type'], type=str)
    # train
    args.add_argument('-loss_func', default=config['train']['loss_func'], type=str)
    args.add_argument('-seed', default=config['train']['seed'], type=int)
    args.add_argument('-batch_size', default=config['train']['batch_size'], type=int)
    args.add_argument('-epochs', default=config['train']['epochs'], type=int)
    args.add_argument('-lr_init', default=config['train']['lr_init'], type=float)
    args.add_argument('-lr_decay', default=config['train']['lr_decay'], type=eval)
    args.add_argument('-lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
    args.add_argument('-lr_decay_step', default=config['train']['lr_decay_step'], type=str)
    args.add_argument('-early_stop', default=config['train']['early_stop'], type=eval)
    args.add_argument('-early_stop_patience', default=config['train']['early_stop_patience'], type=int)
    args.add_argument('-change_epoch', default=config['train']['change_epoch'], type=int)
    args.add_argument('-up_epoch', default=config['train']['up_epoch'], type=str)
    args.add_argument('-grad_norm', default=config['train']['grad_norm'], type=eval)
    args.add_argument('-max_grad_norm', default=config['train']['max_grad_norm'], type=int)
    args.add_argument('-debug', default=config['train']['debug'], type=eval)
    args.add_argument('-real_value', default=config['train']['real_value'], type=eval, help='use real value for loss calculation')
    args.add_argument('-seed_mode', default=config['train']['seed_mode'], type=eval)
    args.add_argument('-xavier', default=config['train']['xavier'], type=eval)
    args.add_argument('-load_pretrain_path', default=config['train']['load_pretrain_path'], type=str)
    args.add_argument('-save_pretrain_path', default=config['train']['save_pretrain_path'], type=str)
    # test
    args.add_argument('-mae_thresh', default=config['test']['mae_thresh'], type=eval)
    args.add_argument('-mape_thresh', default=config['test']['mape_thresh'], type=float)
    # log
    args.add_argument('-log_dir', default='./', type=str)
    args.add_argument('-log_step', default=config['log']['log_step'], type=int)
    args.add_argument('-plot', default=config['log']['plot'], type=eval)
    args, _ = args.parse_known_args()
    return args