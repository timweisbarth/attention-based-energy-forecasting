import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, LSTM, DLinear, PatchTST, TSMixer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'LSTM': LSTM,
            'DLinear': DLinear,
            'PatchTST': PatchTST,
            'TSMixer': TSMixer,
        }
        if self.args.model == 'LSTM':
            model = model_dict[self.args.model].Model(self.args, self.device).float()
        else:
            model = model_dict[self.args.model].Model(self.args).float()
        print(f'Number of parameters in network = {sum(p.numel() for p in model.parameters())}\n')
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _predict(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # decoder input
        #print(batch_y.shape)
        #print(batch_x_mark.shape)
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder

        def _run_model():
            if 'Linear' in self.args.model or 'TST' in self.args.model:
                outputs = self.model(batch_x)
                #print(outputs.shape)
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                #print("batch_x in predict", batch_x.shape)
                if self.args.output_attention:
                    outputs = outputs[0]
            return outputs

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = _run_model()
        else:
            outputs = _run_model()
        #print(outputs.shape)
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        return outputs, batch_y

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    # Train model for args.train_epochs on batched train data
    # Validate after every epoch
    def train(self, setting):

        

        #print(start_time)
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        #test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        print("train_steps", train_steps)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        start_time = time.time()

        train_epoch_losses = []
        vali_epoch_losses = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_batch_losses = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                loss = criterion(outputs, batch_y)
                train_batch_losses.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | train batch loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count

                    # speed[s/iter] where iter is one batch, train_steps = number of iters/batches
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; \t left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                    #for param_group in model_optim.param_groups:
                    #    print(param_group['lr'])

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    #print("End of else")
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()
            # Fore each epoch, do
            print("Epoch train: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            time_validation_start = time.time()
            train_epoch_loss = np.average(train_batch_losses)
            vali_epoch_loss = self.vali(vali_data, vali_loader, criterion)
            print("Epoch validation cost time: ", time.time()-time_validation_start)
            train_epoch_losses.append(train_epoch_loss)
            vali_epoch_losses.append(vali_epoch_loss)

            #test_loss = self.vali(test_data, test_loader, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_epoch_loss, vali_epoch_loss))
            #print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            #    epoch + 1, train_steps, train_loss, vali_loss, test_loss))


            early_stopping(vali_epoch_loss, self.model, path)
            if early_stopping.early_stop:
                self.stop_after_epoch = epoch + 1
                print("Early stopping")
                break
            if self.args.lradj != 'TST':
                # In this case the scheduler is not used by the method
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=True)
            
        self.train_time = time.time() - start_time
        plt.plot(train_epoch_losses, label="Training loss")
        plt.plot(vali_epoch_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")

        plt.savefig(path + "/train_val_loss.pdf", format="pdf", bbox_inches="tight") 
        
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        

        return

    def test(self, setting, flag, test=0):
        """Calculates metrics, preds and truths on data with flag={val, test}
        and saves it in z_results folder. Also visualizes results in pdfs which
        are saved in z_test_results folder"""
        test_data, test_loader = self._get_data(flag=flag)
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './results/' + self.args.des + '/' + setting.split("_iter", 1)[0] + '/'
        #if not os.path.exists(folder_path):
        #    os.makedirs(folder_path)
        subfolder_path = './results/' + self.args.des + '/' + setting.split("_iter", 1)[0] + '/' + setting + '/'
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                if i % 150 == 0:
                    # If features is M, then only the target will be plotted
                    input = batch_x.detach().cpu().numpy()
                    if self.args.features == 'M':
                        num_plots = input.shape[-1]
                    else:
                        num_plots = 1
                        
                    for j in range(1, num_plots+1):
                        # Concatenate input (B, sl, c_out) and true/pred (B, pl, c_out)
                        # Choose 0th batch element and -jth column (sl,) with (pl,) --> (sl+pl,)
                        gt = np.concatenate((input[0, :, -j], true[0, :, -j]), axis=0)
                        pd = np.concatenate((input[0, :, -j], pred[0, :, -j]), axis=0)
                        visual(gt, pd, os.path.join(subfolder_path, str(i) + '_' + str(j) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('{0} shape:'.format(flag), preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('{0} shape:'.format(flag), preds.shape, trues.shape)

        # result save
        #folder_path = './results/' + setting + '/'
        #if not os.path.exists(folder_path):
        #    os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        max_memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 # MB
        model_size = sum(p.numel() for p in self.model.parameters())
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()



        np.save(subfolder_path + '_metrics.npy', np.array([mae, mse, rmse, mape, mspe, model_size, max_memory, self.stop_after_epoch, self.train_time]))
        np.save(subfolder_path + '_pred.npy', preds)
        np.save(subfolder_path + '_true.npy', trues)

        return

#    def predict(self, setting, load=False):
#        pred_data, pred_loader = self._get_data(flag='pred')
#
#        if load:
#            path = os.path.join(self.args.checkpoints, setting)
#            best_model_path = path + '/' + 'checkpoint.pth'
#            logging.info(best_model_path)
#            self.model.load_state_dict(torch.load(best_model_path))
#
#        preds = []
#
#        self.model.eval()
#        with torch.no_grad():
#            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
#                batch_x = batch_x.float().to(self.device)
#                batch_y = batch_y.float()
#                batch_x_mark = batch_x_mark.float().to(self.device)
#                batch_y_mark = batch_y_mark.float().to(self.device)
#
#                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)
#
#                pred = outputs.detach().cpu().numpy()  # .squeeze()
#                preds.append(pred)
#
#        preds = np.array(preds)
#        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
#
#        # result save
#        folder_path = './z_results/' + setting + '/'
#        if not os.path.exists(folder_path):
#            os.makedirs(folder_path)
#
#        np.save(folder_path + 'real_prediction.npy', preds)
#
#        return
