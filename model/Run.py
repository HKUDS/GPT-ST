
import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

import torch
import torch.nn as nn
import configparser
from model.Pretrain_model.GPTST import GPTST_Model as Network_Pretrain
from model.Model import Enhance_model as Network_Predict
from model.BasicTrainer import Trainer
from lib.TrainInits import init_seed
from lib.dataloader import get_dataloader
from lib.TrainInits import print_model_parameters
from lib.metrics import MAE_torch, MSE_torch, huber_loss
from lib.Params_pretrain import parse_args
from lib.Params_predictor import get_predictor_params

# *************************************************************************#

# Mode = 'ori'            #ori, eval, pretrain
# DATASET = 'PEMS08'      #PEMS08, METR_LA, NYC_BIKE, NYC_TAXI
# model = 'MSDR'     # ASTGCN CCRNN DMVSTNET GWN MSDR MTGNN STWA STFGNN STGCN STGODE STMGCN STSGCN TGCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Mkdir(path):
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)

args = parse_args(device)
args_predictor = get_predictor_params(args)
if args.mode !='pretrain':
    attr_list = []
    for arg in vars(args):
        attr_list.append(arg)
    for attr in attr_list:
        if hasattr(args, attr) and hasattr(args_predictor, attr):
            setattr(args, attr, getattr(args_predictor, attr))
    for arg in vars(args):
        print(arg, ':', getattr(args, arg))
    print('==========')
    for arg in vars(args_predictor):
        print(arg, ':', getattr(args_predictor, arg))
init_seed(args.seed, args.seed_mode)

print('mode: ', args.mode, '  model: ', args.model, '  dataset: ', args.dataset, '  load_pretrain_path: ', args.load_pretrain_path, '  save_pretrain_path: ', args.save_pretrain_path)


#config log path
current_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(current_dir,'SAVE', args.dataset)
Mkdir(log_dir)
args.log_dir = log_dir
args.load_pretrain_path = args.load_pretrain_path
args.save_pretrain_path = args.save_pretrain_path

#load dataset
train_loader, val_loader, test_loader, scaler_data, scaler_day, scaler_week, scaler_holiday = get_dataloader(args,
                                                               normalizer=args.normalizer,
                                                               tod=args.tod, dow=False,
                                                               weather=False, single=False)
args.scaler_zeros = scaler_data.transform(0)
args.scaler_zeros_day = scaler_day.transform(0)
args.scaler_zeros_week = scaler_week.transform(0)
# args.scaler_zeros_holiday = scaler_holiday.transform(0)
#init model
if args.mode == 'pretrain':
    model = Network_Pretrain(args)
    model = model.to(args.device)
else:
    model = Network_Predict(args, args_predictor)
    model = model.to(args.device)

if args.xavier:
    for p in model.parameters():
        if p.requires_grad==True:
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

print_model_parameters(model, only_num=False)

#init loss function, optimizer

def scaler_mae_loss(scaler, mask_value):
    def loss(preds, labels, mask=None):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        if args.mode == 'pretrain' and mask is not None:
            preds = preds * mask
            labels = labels * mask
        mae, mae_loss = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae, mae_loss
    return loss

def scaler_huber_loss(scaler, mask_value):
    def loss(preds, labels, mask=None):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        if args.mode == 'pretrain' and mask is not None:
            preds = preds * mask
            labels = labels * mask
        mae, mae_loss = huber_loss(pred=preds, true=labels, mask_value=mask_value)
        return mae, mae_loss
    return loss

if args.loss_func == 'mask_mae':
    loss = scaler_mae_loss(scaler_data, mask_value=args.mape_thresh)
    print('============================scaler_mae_loss')
elif args.loss_func == 'mask_huber':
    if args.mode != 'pretrain':
        loss = scaler_huber_loss(scaler_data, mask_value=args.mape_thresh)
        print('============================scaler_huber_loss')
    else:
        loss = scaler_mae_loss(scaler_data, mask_value=args.mape_thresh)
        print('============================scaler_mae_loss')
    # print(args.model, Mode)
elif args.loss_func == 'mae':
    loss = torch.nn.L1Loss().to(args.device)
elif args.loss_func == 'mse':
    loss = torch.nn.MSELoss().to(args.device)
else:
    raise ValueError
loss_kl = nn.KLDivLoss(reduction='sum').to(args.device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                             weight_decay=0, amsgrad=False)
#learning rate decay
lr_scheduler = None
if args.lr_decay:
    print('Applying learning rate decay.')
    lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=lr_decay_steps,
                                                        gamma=args.lr_decay_rate)

# #config log path
# current_dir = os.path.dirname(os.path.realpath(__file__))
# log_dir = os.path.join(current_dir,'SAVE', args.dataset)
# Mkdir(log_dir)
# args.log_dir = log_dir

#start training

trainer = Trainer(model, loss, loss_kl, optimizer, train_loader, val_loader, test_loader, scaler_data,
                  args, lr_scheduler=lr_scheduler)
if args.mode == 'pretrain':
    trainer.train()
elif args.mode == 'eval':
    trainer.train()
elif args.mode == 'ori':
    trainer.train()
elif args.mode == 'test':
    model.load_state_dict(torch.load(log_dir + '/best_model.pth'))
    print("Load saved model")
    trainer.test(model, trainer.args, test_loader, scaler_data, trainer.logger)
else:
    raise ValueError
