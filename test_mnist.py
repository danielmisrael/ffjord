import argparse
import os
import time
import numpy as np

import torch
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as tforms
from torchvision.utils import save_image

import lib.layers as layers
import lib.utils as utils
import lib.odenvp as odenvp
import lib.multiscale_parallel as multiscale_parallel

from train_misc import standard_normal_logprob
from train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from train_misc import add_spectral_norm, spectral_norm_power_iteration
from train_misc import create_regularization_fns, get_regularization, append_regularization_to_log
# torch.backends.cudnn.benchmark = True
# SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams']
# parser = argparse.ArgumentParser("Continuous Normalizing Flow")
# parser.add_argument("--data", choices=["mnist", "svhn", "cifar10", 'lsun_church'], type=str, default="mnist")
# parser.add_argument("--dims", type=str, default="8,32,32,8")
# parser.add_argument("--strides", type=str, default="2,2,1,-2,-2")
# parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.')
#
# parser.add_argument("--conv", type=eval, default=True, choices=[True, False])
# parser.add_argument(
#     "--layer_type", type=str, default="ignore",
#     choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
# )
# parser.add_argument("--divergence_fn", type=str, default="approximate", choices=["brute_force", "approximate"])
# parser.add_argument(
#     "--nonlinearity", type=str, default="softplus", choices=["tanh", "relu", "softplus", "elu", "swish"]
# )
# parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
# parser.add_argument('--atol', type=float, default=1e-5)
# parser.add_argument('--rtol', type=float, default=1e-5)
# parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")
#
# parser.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
# parser.add_argument('--test_atol', type=float, default=None)
# parser.add_argument('--test_rtol', type=float, default=None)
#
# parser.add_argument("--imagesize", type=int, default=None)
# parser.add_argument("--alpha", type=float, default=1e-6)
# parser.add_argument('--time_length', type=float, default=1.0)
# parser.add_argument('--train_T', type=eval, default=True)
#
# parser.add_argument("--num_epochs", type=int, default=1000)
# parser.add_argument("--batch_size", type=int, default=200)
# parser.add_argument(
#     "--batch_size_schedule", type=str, default="", help="Increases the batchsize at every given epoch, dash separated."
# )
# parser.add_argument("--test_batch_size", type=int, default=200)
# parser.add_argument("--lr", type=float, default=1e-3)
# parser.add_argument("--warmup_iters", type=float, default=1000)
# parser.add_argument("--weight_decay", type=float, default=0.0)
# parser.add_argument("--spectral_norm_niter", type=int, default=10)
#
# parser.add_argument("--add_noise", type=eval, default=True, choices=[True, False])
# parser.add_argument("--batch_norm", type=eval, default=False, choices=[True, False])
# parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
# parser.add_argument('--autoencode', type=eval, default=False, choices=[True, False])
# parser.add_argument('--rademacher', type=eval, default=True, choices=[True, False])
# parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
# parser.add_argument('--multiscale', type=eval, default=False, choices=[True, False])
# parser.add_argument('--parallel', type=eval, default=False, choices=[True, False])
#
# # Regularizations
# parser.add_argument('--l1int', type=float, default=None, help="int_t ||f||_1")
# parser.add_argument('--l2int', type=float, default=None, help="int_t ||f||_2")
# parser.add_argument('--dl2int', type=float, default=None, help="int_t ||f^T df/dt||_2")
# parser.add_argument('--JFrobint', type=float, default=None, help="int_t ||df/dx||_F")
# parser.add_argument('--JdiagFrobint', type=float, default=None, help="int_t ||df_i/dx_i||_F")
# parser.add_argument('--JoffdiagFrobint', type=float, default=None, help="int_t ||df/dx - df_i/dx_i||_F")
#
# parser.add_argument("--time_penalty", type=float, default=0, help="Regularization on the end_time.")
# parser.add_argument(
#     "--max_grad_norm", type=float, default=1e10,
#     help="Max norm of graidents (default is just stupidly high to avoid any clipping)"
# )
#
# parser.add_argument("--begin_epoch", type=int, default=1)
# parser.add_argument("--resume", type=str, default=None)
# parser.add_argument("--save", type=str, default="experiments/cnf")
# parser.add_argument("--val_freq", type=int, default=1)
# parser.add_argument("--log_freq", type=int, default=10)
#
# args = parser.parse_args()
#
#
# def create_model(args, data_shape, regularization_fns):
#     hidden_dims = tuple(map(int, args.dims.split(",")))
#     strides = tuple(map(int, args.strides.split(",")))
#
#     if args.multiscale:
#         model = odenvp.ODENVP(
#             (args.batch_size, *data_shape),
#             n_blocks=args.num_blocks,
#             intermediate_dims=hidden_dims,
#             nonlinearity=args.nonlinearity,
#             alpha=args.alpha,
#             cnf_kwargs={"T": args.time_length, "train_T": args.train_T, "regularization_fns": regularization_fns},
#         )
#     elif args.parallel:
#         model = multiscale_parallel.MultiscaleParallelCNF(
#             (args.batch_size, *data_shape),
#             n_blocks=args.num_blocks,
#             intermediate_dims=hidden_dims,
#             alpha=args.alpha,
#             time_length=args.time_length,
#         )
#     else:
#         if args.autoencode:
#
#             def build_cnf():
#                 autoencoder_diffeq = layers.AutoencoderDiffEqNet(
#                     hidden_dims=hidden_dims,
#                     input_shape=data_shape,
#                     strides=strides,
#                     conv=args.conv,
#                     layer_type=args.layer_type,
#                     nonlinearity=args.nonlinearity,
#                 )
#                 odefunc = layers.AutoencoderODEfunc(
#                     autoencoder_diffeq=autoencoder_diffeq,
#                     divergence_fn=args.divergence_fn,
#                     residual=args.residual,
#                     rademacher=args.rademacher,
#                 )
#                 cnf = layers.CNF(
#                     odefunc=odefunc,
#                     T=args.time_length,
#                     regularization_fns=regularization_fns,
#                     solver=args.solver,
#                 )
#                 return cnf
#         else:
#
#             def build_cnf():
#                 diffeq = layers.ODEnet(
#                     hidden_dims=hidden_dims,
#                     input_shape=data_shape,
#                     strides=strides,
#                     conv=args.conv,
#                     layer_type=args.layer_type,
#                     nonlinearity=args.nonlinearity,
#                 )
#                 odefunc = layers.ODEfunc(
#                     diffeq=diffeq,
#                     divergence_fn=args.divergence_fn,
#                     residual=args.residual,
#                     rademacher=args.rademacher,
#                 )
#                 cnf = layers.CNF(
#                     odefunc=odefunc,
#                     T=args.time_length,
#                     train_T=args.train_T,
#                     regularization_fns=regularization_fns,
#                     solver=args.solver,
#                 )
#                 return cnf
#
#         chain = [layers.LogitTransform(alpha=args.alpha)] if args.alpha > 0 else [layers.ZeroMeanTransform()]
#         chain = chain + [build_cnf() for _ in range(args.num_blocks)]
#         if args.batch_norm:
#             chain.append(layers.MovingBatchNorm2d(data_shape[0]))
#         model = layers.SequentialFlow(chain)
#     return model

model = torch.load('experiments/cnf/mnist_ffjord_50.pth')
print(model)
