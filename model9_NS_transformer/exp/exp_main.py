from data_provider.data_factory import data_provider

from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

from model9_NS_transformer.ns_models import ns_Transformer, ns_iTransformer, Informer, Autoformer
from model9_NS_transformer.exp.exp_basic import Exp_Basic
from model9_NS_transformer.diffusion_models.diffusion_utils import *
from model9_NS_transformer.diffusion_models import DDPM

from model9_NS_transformer.diffusion_models.DDPM_CNNNet import *
from model9_NS_transformer.diffusion_models.DDPM_diffusion_worker import *

from model9_NS_transformer.diffusion_models.samplers.dpm_sampler import DPMSolverSampler
from model9_NS_transformer.diffusion_models.samplers.dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

from multiprocessing import Pool
import CRPS.CRPS as pscore

import warnings

import yaml
from tqdm import tqdm
import wandb
from layers.RevIN import *
from torch.nn.functional import mse_loss

warnings.filterwarnings('ignore')

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'DDPM': DDPM,
        }
        self.args.device = self.device
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self, mode='Model'):
        if mode == 'Model':
            # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
            model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.train_epochs)
        return model_optim, lr_scheduler

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def pretrain(self, setting):

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        time_now = time.time()

        train_steps = len(train_loader)

        model_optim = optim.Adam(self.model.parameters(), lr=0.0001)

        best_train_loss = 10000000.0
        for epoch in range(self.args.pretrain_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                dec_inp = batch_y

                loss = self.model.pretrain_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                train_loss.append(loss.item())

                loss.backward()

                model_optim.step()

            print("PreTraining Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            print("PreTraining Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} ".format(epoch + 1, train_steps, train_loss))
            # wandb.log({"Pretrain Loss": train_loss, "Epoch": epoch + 1})

            if train_loss < best_train_loss:
                print("-------------------------")
                best_train_loss = train_loss
                torch.save(self.model.cond_pred_model.state_dict(), path + '/' + 'cond_pred_model_checkpoint.pth')


    def train(self, setting):

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        best_model_path = path + '/' + 'cond_pred_model_checkpoint.pth'
        self.model.cond_pred_model.load_state_dict(torch.load(best_model_path))
        print("Successfully loading pretrained model!")

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        time_now = time.time()

        train_steps = len(train_loader)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim, lr_scheduler = self._select_optimizer()

        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # best_val_loss = float('inf')
        best_train_loss = 10000000.0
        training_process = {}
        training_process["train_loss"] = []
        training_process["val_loss"] = []
            
        for epoch in range(self.args.train_epochs):

            # Training the diffusion part
            iter_count = 0
            train_loss = []

            self.model.train()

            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device) # batch_y torch.Size([32, 240, 7]) 
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_dec_inp:  
                    loss = self.model.train_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:  
                    loss = self.model.train_forward(batch_x, batch_x_mark, batch_y, batch_y_mark)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            training_process["train_loss"].append(train_loss)

            if epoch % 1 == 0:
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Val Loss: {3:.7f} ".format(epoch + 1, train_steps, train_loss, vali_loss))
                wandb.log({
                            "Train Loss": train_loss,
                            "Val Loss": vali_loss,
                            "Epoch": epoch + 1
                        })
                training_process["val_loss"].append(vali_loss)
            
                if vali_loss < best_train_loss:
                        print("-------------------------")
                        best_train_loss = vali_loss
                        best_model_path = path + '/' + 'checkpoint.pth'
                        torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')
            lr_scheduler.step()

            
            # save the model checkpoints
            early_stopping(vali_loss, self.model, path)

            if (math.isnan(train_loss)):
                break

            if early_stopping.early_stop:
                print("Early stopping")
                break



        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        return self.model
    
    def vali(self, vali_data, vali_loader, criterion):
        inps = []    
        preds = []
        trues = []

        self.model.eval()

        # with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

            if self.args.use_dec_inp: 
                outputs, batch_x, batch_y, mean, label_part = self.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark, sample_times=1)
            else: 
                outputs, batch_x, batch_y, mean, label_part = self.model.forward(batch_x, batch_x_mark, batch_y, batch_y_mark, sample_times=1)

            if len(np.shape(outputs)) == 4:
                outputs = outputs.mean(dim=1)

            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()

            preds.append(pred)
            trues.append(true)
            inps.append(batch_x.detach().cpu().numpy())

            # if self.args.dataset_name not in ["Exchange"]:
            if i > 5:
                break

        inps = np.array(inps)
        preds = np.array(preds)
        trues = np.array(trues)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        wandb.log({
                        "Validation MSE": mse,
                        "Validation MAE": mae
                    })

        
        return mse


    def test(self, setting, test=0):

        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        print("Successfully loading trained model!")

        def compute_true_coverage_by_gen_QI(dataset_object, all_true_y, all_generated_y):
            n_bins = 10
            quantile_list = np.arange(n_bins + 1) * (100 / n_bins)
            # compute generated y quantiles
            y_pred_quantiles = np.percentile(all_generated_y.squeeze(), q=quantile_list, axis=1)
            y_true = all_true_y.T
            quantile_membership_array = ((y_true - y_pred_quantiles) > 0).astype(int)
            y_true_quantile_membership = quantile_membership_array.sum(axis=0)
            # y_true_quantile_bin_count = np.bincount(y_true_quantile_membership)
            y_true_quantile_bin_count = np.array(
                [(y_true_quantile_membership == v).sum() for v in np.arange(n_bins + 2)])

            # combine true y falls outside of 0-100 gen y quantile to the first and last interval
            y_true_quantile_bin_count[1] += y_true_quantile_bin_count[0]
            y_true_quantile_bin_count[-2] += y_true_quantile_bin_count[-1]
            y_true_quantile_bin_count_ = y_true_quantile_bin_count[1:-1]
            # compute true y coverage ratio for each gen y quantile interval
            # y_true_ratio_by_bin = y_true_quantile_bin_count_ / dataset_object.test_n_samples
            y_true_ratio_by_bin = y_true_quantile_bin_count_ / dataset_object
            assert np.abs(
                np.sum(y_true_ratio_by_bin) - 1) < 1e-10, "Sum of quantile coverage ratios shall be 1!"
            qice_coverage_ratio = np.absolute(np.ones(n_bins) / n_bins - y_true_ratio_by_bin).mean()
            return y_true_ratio_by_bin, qice_coverage_ratio, y_true
        
        def compute_PICP(y_true, all_gen_y, return_CI=False):
            """
            Another coverage metric.
            """
            low, high = [2.5, 97.5]
            CI_y_pred = np.percentile(all_gen_y.squeeze(), q=[low, high], axis=1)
            # compute percentage of true y in the range of credible interval
            y_in_range = (y_true >= CI_y_pred[0]) & (y_true <= CI_y_pred[1])
            coverage = y_in_range.mean()
            if return_CI:
                return coverage, CI_y_pred, low, high
            else:
                return coverage, low, high
    

        inps = []   
        preds = []
        all_generated_samples = []
        trues = []
        
        return_mean = []
        return_label = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader), total=len(test_loader), desc='Testing'):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                start_time = time.time()
                sample_times = self.args.sample_times

                if self.args.use_dec_inp:  
                    outputs, batch_x, dec_inp, mean, label_part = self.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark, sample_times=sample_times)
                else:  
                    outputs, batch_x, batch_y, mean, label_part = self.model.forward(batch_x, batch_x_mark, batch_y, batch_y_mark, sample_times=sample_times)
                
                end_time = time.time()
                elapsed_time_ms = (end_time - start_time) * 1000 / np.shape(batch_x)[0]

                if i < 5:
                    print(f"Elapsed time: {elapsed_time_ms:.2f} ms")

                # pred = outs.detach().cpu().numpy()  # outputs.detach().cpu().numpy()  # .squeeze()
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()  # .squeeze()

                return_mean_i = mean
                return_label_i = label_part

                if len(np.shape(pred)) == 4:
                    preds.append(pred.mean(axis=1))
                    if self.args.sample_times > 1:
                        all_generated_samples.append(pred)
                else:
                    preds.append(pred)
                trues.append(true)

                if return_mean_i is not None:
                    return_mean.append(return_mean_i.detach().cpu().numpy())
                if return_label_i is not None:
                    return_label.append(return_label_i.detach().cpu().numpy())

               
                if i%20 == 0:
                    # print('True')
                    input = batch_x.detach().cpu().numpy()
                    # print(f"input shape: {input[0, :, -1].shape}")
                    # print(f'true shape: {true[0,:,-1].shape}')
                    # print(f"pred shape: {pred.mean(axis=(0,1,3)).shape}")

                    gt = np.concatenate((input[0,:,-1], true[0,:,-1]), axis=0)
                    pd = np.concatenate((input[0,:,-1], pred.mean(axis=1)[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        inps = np.array(inps)
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        id_worst = None

        if self.args.sample_times > 1:
            all_generated_samples = np.array(all_generated_samples)

        # print("preds", np.shape(preds))
        preds = preds.reshape(-1, trues.shape[-2], trues.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        if self.args.sample_times > 1:
            all_generated_samples = all_generated_samples.reshape(-1, self.args.sample_times , trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        wandb.log({
                    "Test MSE": mse,
                    "Test MAE": mae
                })

        print('NT metrc: mse:{:.4f}, mae:{:.4f} , rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}'.format(mse, mae, rmse, mape,
                                                                                                mspe))


        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        return
    
    def test_pretrain(self, setting):
        """
        Pretrain 모델의 MSE 및 MAE를 계산하는 테스트 함수
        """
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        best_model_path = path + '/' + 'cond_pred_model_checkpoint.pth'

        self.model.cond_pred_model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        print("Successfully loading pretrained model for evaluation!")

        preds = []
        trues = []

        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader), total=len(test_loader), desc='Testing Pretrain Model'):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = batch_y  
                outputs = self.model.cond_pred_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                batch_y = batch_y[:, -self.args.pred_len:, :]

                if len(np.shape(outputs)) == 4:
                    outputs = outputs.mean(dim=1)

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        print('Pretrain test shape:', preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        # wandb.log({
        #     "Pretrain Test MSE": mse,
        #     "Pretrain Test MAE": mae
        # })

        print('Pretrain NT metric: mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}'.format(mse, mae, rmse, mape, mspe))

        f = open("pretrain_result.txt", 'a')
        f.write(setting + "  \n")
        f.write('Pretrain mse:{}, mae:{}\n'.format(mse, mae))
        f.close()

        return mse, mae
