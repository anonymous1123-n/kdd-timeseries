import argparse
import torch
from model9_NS_transformer.exp.exp_main import Exp_Main
import random
import numpy as np
import setproctitle
import wandb

if __name__ == '__main__':
    setproctitle.setproctitle('LYX_thread')

    parser = argparse.ArgumentParser(description='Non-stationary Transformers for Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=bool, default=False, help='status')
    parser.add_argument('--model_id', type=str, default='exchange_96_192_informer', help='model id')
    parser.add_argument('--model', type=str, default='DDPM', help='model name, options: [DDPM]')

    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/exchange_rate/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='exchange_rate.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='d',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=192, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=8, help='encoder input size') # ETT family : 7, Exchange : 8, Electiricity: 321, Traffic:862
    parser.add_argument('--dec_in', type=int, default=8, help='decoder input size') # ETT family : 7, Exchange : 8, Electiricity: 321, Traffic:862
    parser.add_argument('--c_out', type=int, default=8, help='output size') # ETT family : 7, Exchange : 8, Electiricity: 321, Traffic:862
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    parser.add_argument('--k_z', type=float, default=1e-2, help='KL weight 1e-9')
    parser.add_argument('--k_cond', type=float, default=1, help='Condition weight')
    parser.add_argument('--d_z', type=int, default=8, help='KL weight')


    # optimization
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs') # 200
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')  # 32
    parser.add_argument('--test_batch_size', type=int, default=8, help='batch size of train input data')  # 32
    parser.add_argument('--patience', type=int, default=15, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate for diffusion model')
    parser.add_argument('--learning_rate_Cond', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=1, help='gpu')
    parser.add_argument('--use_multi_gpu', type=bool, default=False, help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='1', help='device ids of multile gpus')
    parser.add_argument('--seed', type=int, default=-1, help='random seed')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[64, 64],
                        help='hidden layer dimensions of projector (List)') # ETT : [64,64], ECL : [256,256]
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector') # ETT, ECL:2

    # CART related args
    parser.add_argument('--diffusion_config_dir', type=str, default='./model9_NS_transformer/configs/toy_8gauss.yml',
                        help='')

    # parser.add_argument('--cond_pred_model_dir', type=str,
    #                     default='./checkpoints/cond_pred_model_pertrain_NS_Transformer/checkpoint.pth', help='')
    parser.add_argument('--cond_pred_model_pertrain_dir', type=str,
                        default='./checkpoints/cond_pred_model_pertrain_NS_Transformer/checkpoint.pth', help='')

    parser.add_argument('--CART_input_x_embed_dim', type=int, default=32, help='feature dim for x in diffusion model')
    parser.add_argument('--mse_timestep', type=int, default=0, help='')

    parser.add_argument('--MLP_diffusion_net', type=bool, default=False, help='use MLP or Unet')

    # Some args for Ax (all about diffusion part)
    parser.add_argument('--timesteps', type=int, default=1000, help='')

    # 다경 추가
    parser.add_argument('--pretrain_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--UNet_Type', type=str, default='CNN', help=['CNN'])
    parser.add_argument('--type_sampler', type=str, default='dpm', help=["none", "dpm"])






    parser.add_argument('--use_window_normalization', type=bool, default=True)
    parser.add_argument('--diff_steps', type=int, default=100, help='number of diffusion steps')
    parser.add_argument('--sample_times', type=int, default=10)

    
    parser.add_argument('--ddpm_inp_embed', type=int, default=256) # 256
    parser.add_argument('--ddpm_dim_diff_steps', type=int, default=256)
    parser.add_argument('--ddpm_channels_conv', type=int, default=256)
    parser.add_argument('--ddpm_channels_fusion_I', type=int, default=256)
    parser.add_argument('--ddpm_layers_inp', type=int, default=5)
    parser.add_argument('--ddpm_layers_I', type=int, default=5)
    parser.add_argument('--ddpm_layers_II', type=int, default=5)
    parser.add_argument('--cond_ddpm_num_layers', type=int, default=5)
    parser.add_argument('--cond_ddpm_channels_conv', type=int, default=64)

    parser.add_argument('--ablation_study_case', type=str, default="none", help="none, mix_1, ar_1, mix_ar_0, w_pred_loss")
    parser.add_argument('--weight_pred_loss', type=float, default=0.0)
    parser.add_argument('--ablation_study_F_type', type=str, default="CNN", help="Linear, CNN")
    parser.add_argument('--ablation_study_masking_type', type=str, default="none", help="none, hard, segment")
    parser.add_argument('--ablation_study_masking_tau', type=float, default=0.9)

    parser.add_argument('--parameterization', type=str, default='x_start', help=["noise", "x_start"])
    parser.add_argument('--wandb_project', type=str, default='diffusion', help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity name')

    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--num_vars', type=int, default=8, help='# of variables') # ETT family : 7, Exchange : 8, Electiricity: 321, Traffic:862
    parser.add_argument('--subtract_last', type=int, default=1, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--our_ddpm_clip', type=float, default=100) # 100

    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token') # iTransformer
    parser.add_argument('--use_norm', type=int, default=False, help='use norm and denorm')

    parser.add_argument('--use_dec_inp', type=bool, default=False, help='use dec_inp in training')
    parser.add_argument('--cond_i', type=bool, default=True, help='use dec_inp in training')

    parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')


    args = parser.parse_args()
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
        name='exchange_96_192_informer'
    )

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.seed == -1:
        fix_seed = np.random.randint(2147483647)
    else:
        fix_seed = args.seed

    print('Using seed:', fix_seed)

    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    if args.use_gpu:
        if args.use_multi_gpu:
            args.devices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]
        else:
            torch.cuda.set_device(args.gpu)

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)
            
            exp = Exp(args)  # set experiments
            
            print('>>>>>>>start pretraining : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.pretrain(setting)
            
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            torch.cuda.empty_cache()
            wandb.log({'setting': setting})
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                      args.model,
                                                                                                      args.data,
                                                                                                      args.features,
                                                                                                      args.seq_len,
                                                                                                      args.label_len,
                                                                                                      args.pred_len,
                                                                                                      args.d_model,
                                                                                                      args.n_heads,
                                                                                                      args.e_layers,
                                                                                                      args.d_layers,
                                                                                                      args.d_ff,
                                                                                                      args.factor,
                                                                                                      args.embed,
                                                                                                      args.distil,
                                                                                                      args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
