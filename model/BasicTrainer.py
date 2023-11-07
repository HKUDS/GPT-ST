import torch
import os
import time
import copy
import numpy as np
from lib.logger import get_logger
from lib.metrics import All_Metrics

class Trainer(object):
    def __init__(self, model, loss, loss_kl, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        # self.model_stu = model_stu
        self.args = args
        self.loss = loss
        self.loss_kl = loss_kl
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        self.batch_seen = 0
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, self.args.save_pretrain_path)
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        #log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        #if not args.debug:
        #self.logger.info("Argument: %r", args)
        # for arg, value in sorted(vars(args).items()):
        #     self.logger.info("Argument %s: %r", arg, value)

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0
        # val_pred = []
        # val_true = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data = data[..., :self.args.input_base_dim + self.args.input_extra_dim]
                if self.args.mode == 'pretrain':
                    label = data[..., :self.args.input_base_dim + self.args.input_extra_dim]
                else:
                    label = target[..., :self.args.input_base_dim + self.args.input_extra_dim]
                output, _, mask, _, _ = self.model(data, label=None)
                # if self.args.real_value:
                #     label = self.scaler.inverse_transform(label[..., :self.args.output_dim])
                if self.args.mode == 'pretrain':
                    loss, loss_base = self.loss(output, label[..., :self.args.output_dim], mask)
                else:
                    loss, _ = self.loss(output, label[..., :self.args.output_dim])
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))

        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_flow_loss = 0
        total_s_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.batch_seen += 1
            data = data[..., :self.args.input_base_dim + self.args.input_extra_dim]
            if self.args.mode == 'pretrain':
                label = data[..., :self.args.input_base_dim + self.args.input_extra_dim]
            else:
                label = target[..., :self.args.input_base_dim + self.args.input_extra_dim]
            self.optimizer.zero_grad()

            if self.args.mode == 'pretrain':
                out, out_time, mask, probability, eb = self.model(data, label, self.batch_seen, epoch)
                loss_flow, loss_base = self.loss(out, label[..., :self.args.output_dim], mask)
                if epoch > self.args.change_epoch :
                    loss_s = self.loss_kl(probability.log(), eb) * 0.1
                    loss = loss_flow + loss_s
                else:
                    loss = loss_flow
            else:
                out, out_time, mask, probability, eb2 = self.model(data, label, self.batch_seen)
                loss, _ = self.loss(out, label[..., :self.args.output_dim])
            loss.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()
            # calculate total loss
            if self.args.mode == 'pretrain':
                total_flow_loss += loss_flow.item()
                if epoch > self.args.change_epoch:
                    total_s_loss += loss_s.item()
            #log information
            if batch_idx % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))
        train_epoch_loss = total_loss/self.train_per_epoch
        if self.args.mode == 'pretrain':
            train_epoch_flow_loss = total_flow_loss/self.train_per_epoch
            train_epoch_s_loss = total_s_loss / self.train_per_epoch
            self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f} averaged Loss_s: {:.6f}'.format(epoch, train_epoch_flow_loss, train_epoch_s_loss))
        else:
            self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f}'.format(epoch, train_epoch_loss))

        #learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        # train_epoch_flow_loss for params selecting
        if self.args.mode == 'pretrain':
            return train_epoch_flow_loss
        else:
            return train_epoch_loss

    def train(self):
        best_model = None
        best_model_test = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        up_epoch = [int(i) for i in list(self.args.up_epoch.split(','))]
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            # epoch_time = time.time()
            train_epoch_loss = self.train_epoch(epoch)
            # print(time.time()-epoch_time)
            if epoch in up_epoch:
                best_loss = float('inf')
            if self.args.mode == 'pretrain':
                if train_epoch_loss < best_loss:
                    best_loss = train_epoch_loss
                    not_improved_count = 0
                    best_state = True
                else:
                    not_improved_count += 1
                    best_state = False
            else:
                if self.val_loader == None:
                    val_dataloader = self.test_loader
                else:
                    val_dataloader = self.val_loader
                val_epoch_loss = self.val_epoch(epoch, val_dataloader)
                val_loss_list.append(val_epoch_loss)
                if val_epoch_loss < best_loss:
                    best_loss = val_epoch_loss
                    not_improved_count = 0
                    best_state = True
                else:
                    not_improved_count += 1
                    best_state = False

            #print('LR:', self.optimizer.param_groups[0]['lr'])
            train_loss_list.append(train_epoch_loss)

            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())
                best_model_test = self.model

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        #save the best model to file
        # if not self.args.debug:
        if self.args.debug:
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)

        #test
        # self.model.load_state_dict(best_model)
        #self.val_epoch(self.args.epochs, self.test_loader)
        if self.args.mode == 'pretrain':
            self.test(best_model_test, self.args, self.train_loader, self.scaler, self.logger)
        else:
            self.test(best_model_test, self.args, self.test_loader, self.scaler, self.logger)


    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, args, data_loader, scaler, logger, path=None):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data[..., :args.input_base_dim + args.input_extra_dim]
                # label = target[..., :args.input_base_dim + args.input_extra_dim]
                if args.mode == 'pretrain':
                    output, _, mask, _, _ = model(data, None, None, args.epochs)
                    label = data[..., :args.output_dim]
                    y_true.append(label*mask)
                    y_pred.append(output*mask)
                else:
                    output, _, mask, _, _ = model(data, label)
                    label = target[..., :args.output_dim]
                    y_true.append(label)
                    y_pred.append(output)
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        # if args.real_value:
        #     y_pred = torch.cat(y_pred, dim=0)
        # else:
        y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
        # np.save('./{}_true.npy'.format(args.dataset+'_'+args.model+'_'+args.mode), y_true.cpu().numpy())
        # np.save('./{}_pred.npy'.format(args.dataset+'_'+args.model+'_'+args.mode), y_pred.cpu().numpy())
        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, corr = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                args.mae_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}, CORR:{:.4f}%".format(
                t + 1, mae, rmse, mape*100, corr))
        mae, rmse, mape, _, corr = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%, CORR:{:.4f}".format(
                    mae, rmse, mape*100, corr))

