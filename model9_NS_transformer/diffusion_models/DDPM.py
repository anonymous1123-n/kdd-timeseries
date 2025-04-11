'''
ILI
'''

from typing import List, Optional, Tuple, Union
from typing import Any, Dict
from functools import partial
from inspect import isfunction
from tqdm import tqdm
import math
import torch
import torch.nn as nn
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn.modules import loss
import numpy as np
import random

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from model9_NS_transformer.diffusion_models.DDPM_CNNNet import *
from model9_NS_transformer.diffusion_models.DDPM_diffusion_worker import *

from .samplers.dpm_sampler import DPMSolverSampler
from model9_NS_transformer.ns_models import ns_Transformer, ns_iTransformer, ns_Autoformer, Autoformer, Informer


class Model(nn.Module):
    
    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args
        self.device = args.device

        self.seq_len = args.seq_len 
        self.pred_len = args.pred_len 
        self.norm_len = args.label_len

        self.input_size = args.num_vars 
        self.diff_steps = args.diff_steps

        if args.UNet_Type == "CNN":
            u_net = CNN_DiffusionUnet(args, self.input_size, self.seq_len+self.pred_len, self.pred_len, self.diff_steps)
        
        self.diffusion_worker = Diffusion_Worker(args, u_net, self.diff_steps)

        if args.type_sampler == "none":
            pass
        elif args.type_sampler == "dpm":
            assert self.args.parameterization == "x_start"
            self.sampler = DPMSolverSampler(u_net, self.diffusion_worker)

        # self.cond_pred_model = ns_Transformer.Model(args)
        # self.cond_pred_model = ns_iTransformer.Model(args)
        # self.cond_pred_model = Autoformer.Model(args)
        self.cond_pred_model = Informer.Model(args)


    def pretrain_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # self.diffusion_worker.eval()
        self.cond_pred_model.train()

        cond = self.cond_pred_model(x_enc, x_mark_enc, x_dec, x_mark_dec) # y_0_hat_batch torch.Size([32, 54, 7])

        target = x_dec[:,-self.pred_len:,:] # target torch.Size([32, 36, 7])

        f_dim = -1 if self.args.features == 'MS' else 0

        # print('target', target.shape)
        # print('y_0_hat_batch', y_0_hat_batch.shape)

        loss = F.mse_loss(cond[:,:,f_dim:], target[:,:,f_dim:])
        
        return loss 

    def train_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        self.diffusion_worker.train()
        self.cond_pred_model.eval()

        # Conditional predictions from ns_Transformer        
        cond = self.cond_pred_model(x_enc, x_mark_enc, x_dec, x_mark_dec)

        if self.args.use_window_normalization:
            if self.args.use_dec_inp:
                seq_len = np.shape(x_enc)[1]
            
                mean_ = torch.mean(x_enc[:,-self.norm_len:,:], dim=1).unsqueeze(1)
                std_ = torch.ones_like(torch.std(x_enc, dim=1).unsqueeze(1))

                x_enc_i = (x_enc-mean_.repeat(1,seq_len,1))/(std_.repeat(1,seq_len,1)+0.00001)

                seq_len = np.shape(x_dec)[1]
                x_dec_i = (x_dec-mean_.repeat(1,seq_len,1))/(std_.repeat(1,seq_len,1)+0.00001)

                seq_len = np.shape(cond)[1]
                cond_i = (cond-mean_.repeat(1,seq_len,1))/(std_.repeat(1,seq_len,1)+0.00001)
                
            else:
                seq_len = np.shape(x_enc)[1]
                
                mean_ = torch.mean(x_enc[:,-self.norm_len:,:], dim=1).unsqueeze(1)
                std_ = torch.ones_like(torch.std(x_enc, dim=1).unsqueeze(1))

                x_enc_i = (x_enc-mean_.repeat(1,seq_len,1))/(std_.repeat(1,seq_len,1)+0.00001)

                seq_len = np.shape(x_dec[:, -self.args.pred_len:, :])[1]
                x_dec_i = (x_dec[:, -self.args.pred_len:, :]-mean_.repeat(1,seq_len,1))/(std_.repeat(1,seq_len,1)+0.00001)

                seq_len = np.shape(cond)[1]
                cond_i = (cond-mean_.repeat(1,seq_len,1))/(std_.repeat(1,seq_len,1)+0.00001)
                
        else:
            if self.args.use_dec_inp:
                x_enc_i = x_enc
                x_dec_i = x_dec
                cond_i = cond
            else:
                x_enc_i = x_enc
                x_dec_i = x_dec[:, -self.args.pred_len:, :]
                cond_i = cond
                

        if self.args.use_dec_inp:
            x_past = x_enc_i
            x_future = x_dec_i[:, -self.args.pred_len:, :] # ILI
        else:
            x_past = x_enc_i
            x_future = x_dec_i
            

        x_past = x_past.permute(0,2,1)     # torch.Size([64, 30, 24])
        x_future = x_future.permute(0,2,1) # [bsz, fea, seq_len]
        # print('x_past1', x_past.shape) x_past1 torch.Size([32, 7, 36])

        if self.args.cond_i:
            x_past = torch.cat([x_past, cond_i.permute(0,2,1)], dim=-1) # exchange
            # print('x_past_2', x_past.shape) x_past_2 torch.Size([32, 7, 72])
        else:
            x_past = torch.cat([x_past, cond.permute(0,2,1)], dim=-1) # ILI
        
        f_dim = -1 if self.args.features in ['MS'] else 0
        # print('cond', cond.shape) cond torch.Size([32, 36, 7])
        # print('y_0_hat_batch', y_0_hat_batch.shape) y_0_hat_batch torch.Size([32, 54, 7])
        loss = self.diffusion_worker(x=x_future[:,f_dim:,:], cond_ts=x_past, y_0_hat=cond)
        return loss

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, sample_times=5):

        self.diffusion_worker.eval()
        self.cond_pred_model.eval()

        cond= self.cond_pred_model(x_enc, x_mark_enc, x_dec, x_mark_dec)

        if self.args.use_window_normalization:
            if self.args.use_dec_inp: # ILI
                seq_len = np.shape(x_enc)[1]
            
                mean_ = torch.mean(x_enc[:,-self.norm_len:,:], dim=1).unsqueeze(1)
                std_ = torch.ones_like(torch.std(x_enc, dim=1).unsqueeze(1))

                x_enc_i = (x_enc-mean_.repeat(1,seq_len,1))/(std_.repeat(1,seq_len,1)+0.00001)

                seq_len = np.shape(x_dec)[1]
                dec_inp_nonzero = x_dec[:, :self.args.label_len, :]
                dec_inp_nonzero_norm = (dec_inp_nonzero - mean_.repeat(1, self.args.label_len, 1)) / (
                    std_.repeat(1, self.args.label_len, 1) + 1e-5
                )
                # x_dec_i = (x_dec-mean_.repeat(1,seq_len,1))/(std_.repeat(1,seq_len,1)+0.00001)
                x_dec_i = torch.cat([dec_inp_nonzero_norm, x_dec[:, self.args.label_len:, :]], dim=1)

                seq_len = np.shape(cond)[1]
                cond_i = (cond-mean_.repeat(1,seq_len,1))/(std_.repeat(1,seq_len,1)+0.00001)
                
            else: # exchange, ETTm2
                seq_len = np.shape(x_enc)[1]
                
                mean_ = torch.mean(x_enc[:,-self.norm_len:,:], dim=1).unsqueeze(1)
                std_ = torch.ones_like(torch.std(x_enc, dim=1).unsqueeze(1))

                x_enc_i = (x_enc-mean_.repeat(1,seq_len,1))/(std_.repeat(1,seq_len,1)+0.00001)

                seq_len = np.shape(x_dec[:, -self.args.pred_len:, :])[1]
                x_dec_i = (x_dec[:, -self.args.pred_len:, :]-mean_.repeat(1,seq_len,1))/(std_.repeat(1,seq_len,1)+0.00001)

                seq_len = np.shape(cond)[1]
                cond_i = (cond-mean_.repeat(1,seq_len,1))/(std_.repeat(1,seq_len,1)+0.00001)
                
        else:
            if self.args.use_dec_inp:
                x_enc_i = x_enc
                x_dec_i = x_dec
                cond_i = cond
            else:
                x_enc_i = x_enc
                x_dec_i = x_dec[:, -self.args.pred_len:, :]
                cond_i = cond
                

        if self.args.use_dec_inp:
            x_past = x_enc_i
            x_future = x_dec_i[:, -self.args.pred_len:, :] # ILI
        else:
            x_past = x_enc_i
            x_future = x_dec_i

        x_past = x_past.permute(0,2,1)     # torch.Size([64, 30, 24])
        x_future = x_future.permute(0,2,1) # [bsz, fea, seq_len]

        if self.args.cond_i:
            x_past = torch.cat([x_past, cond_i.permute(0,2,1)], dim=-1) # exchange
        else:
            x_past = torch.cat([x_past, cond.permute(0,2,1)], dim=-1) # ILI

        B, nF, nL = np.shape(x_past)[0], self.input_size, self.pred_len
        if self.args.features in ['MS']:
            nF = 1
        shape = [nF, nL]
        
        all_outs = []
        # step_list = [10, 15, 15, 20, 20, 20, 25, 25, 30, 35]
        step_list = random.choices(range(20, 41), k=10)
        for i in range(sample_times):
            start_code = torch.randn((B, nF, nL), device=self.device)

            S_i = step_list[i]
            
            if self.args.type_sampler == "none":
                f_dim = -1 if self.args.features in ['MS'] else 0
                outs_i = self.diffusion_worker.sample(x_future[:,f_dim:,:], x_past)
            else:
                samples_ddim, _ = self.sampler.sample(S=S_i,
                                             conditioning=x_past,
                                             batch_size=B,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=1.0,
                                             unconditional_conditioning=None,
                                             eta=0.,
                                             x_T=start_code)
                outs_i = samples_ddim.permute(0,2,1)

            if self.args.use_window_normalization:
                out_len = np.shape(outs_i)[1]
                outs_i = outs_i * std_.repeat(1,out_len,1) + mean_.repeat(1,out_len,1)

            all_outs.append(outs_i)
        all_outs = torch.stack(all_outs, dim=0)

        flag_return_all = True

        if flag_return_all:
            outs = all_outs.permute(1,0,2,3)
            f_dim = -1 if self.args.features in ['MS'] else 0
            outs = outs[:, :, -self.pred_len:, f_dim:] # - 0.4
        else:
            outs = all_outs.mean(0)
            f_dim = -1 if self.args.features == ['MS'] else 0
            outs = outs[:, -self.pred_len:, f_dim:] # - 0.4

        if self.args.use_window_normalization:
            
            out_len = np.shape(x_enc_i)[1]
            x_enc_i = x_enc_i * std_.repeat(1,out_len,1) + mean_.repeat(1,out_len,1)

            out_len = np.shape(x_dec_i)[1]
            x_dec_i = x_dec_i * std_.repeat(1,out_len,1) + mean_.repeat(1,out_len,1)

            # out_len = np.shape(linear_outputs_i)[1]
            # linear_outputs_i = linear_outputs_i * std_.repeat(1,out_len,1) + mean_.repeat(1,out_len,1)

        return outs, x_enc[:,:,f_dim:], x_dec[:, -self.args.pred_len:, f_dim:], None, None




