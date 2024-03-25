# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2022/09/29
@Author  :   Hou Jinlin
@Contact :   1252405352@qq.com
'''
import argparse
import networkx as nx
import numpy as np
import dill
import pickle as pkl
import scipy
import scipy.sparse as sp
from torch.utils.data import DataLoader

from utils.preprocess import load_graphs, get_context_pairs, get_evaluation_data
# from utils.minibatch import DynamicsDataset
from utils.utilities import *
from eval.link_prediction import evaluate_classifier
from models.model import GeneDynamics, HeatDiffusion, MutualDynamics
from models.ndcn import NDCN
import torchdiffeq as ode

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import random
import os
import wandb
wandb.init(project="NDCN", entity="houjinlin")
torch.autograd.set_detect_anomaly(True)

def get_gnn_sup_d(adj):
    '''
    Function to get GNN support (normalized adjacency matrix w/ self-connected edges)
    :param adj: original adjacency matrix
    :return: GNN support
    '''
    # ====================
    num_nodes, _ = adj.shape
    adj = adj + np.eye(num_nodes)
    degs = np.sum(adj, axis=1)
    sup = adj # GNN support
    for i in range(num_nodes):
        sup[i, :] /= degs[i]

    return sup

def sparse_to_tuple(sparse_mx):
    '''
    Function to transfer sparse matrix to tuple format
    :param sparse_mx: original sparse matrix
    :return: corresponding tuple format
    '''
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx): # sp.sparse.isspmatrix_coo(mx)
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape
    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


# python train.py --time_steps 6 --dataset uci --gpu 1 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_steps', type=int, nargs='?', default=100,
                        help="total time steps used for train, eval and test")
    # Experimental settings.
    parser.add_argument('--dataset', type=str, nargs='?', default='Enron',
                        help='dataset name')
    parser.add_argument('--gpu', type=int, nargs='?', default=0,
                        help='GPU_ID (0/1 etc.)')
    parser.add_argument('--epochs', type=int, nargs='?', default=2000,
                        help='# epochs')
    parser.add_argument('--val_freq', type=int, nargs='?', default=1,
                        help='Validation frequency (in epochs)')
    parser.add_argument('--test_freq', type=int, nargs='?', default=20,
                        help='Testing frequency (in epochs)')
    parser.add_argument('--batch_size', type=int, nargs='?', default=32,
                        help='Batch size (# nodes)')
    parser.add_argument('--featureless', type=bool, nargs='?', default=True,
                    help='True if one-hot encoding.')
    parser.add_argument("--early_stop", type=int, default=10,
                        help="patient")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed")

    # Tunable hyper-params
    # TODO: Implementation has not been verified, performance may not be good.
    parser.add_argument('--learning_rate', type=float, nargs='?', default=0.01,
                        help='Initial learning rate for self-attention model.')
    parser.add_argument('--dropout', type=float, nargs='?', default=0.1,
                        help='Spatial (structural) attention Dropout (1 - keep probability).')
    # parser.add_argument('--temporal_drop', type=float, nargs='?', default=0.5,
                        # help='Temporal attention Dropout (1 - keep probability).')
    parser.add_argument('--weight_decay', type=float, nargs='?', default=1e-3,
                        help='Initial learning rate for self-attention model.')
    # Architecture params
    # parser.add_argument('--structural_head_config', type=str, nargs='?', default='16,8,8',
    #                     help='Encoder layer config: # attention heads in each GAT layer')
    parser.add_argument('--structural_layer_config', type=str, nargs='?', default='32',
                        help='Encoder layer config: # units in each GAT layer')
    # parser.add_argument('--temporal_head_config', type=str, nargs='?', default='16',
    #                     help='Encoder layer config: # attention heads in each Temporal layer')
    parser.add_argument('--temporal_layer_config', type=str, nargs='?', default='32',
                        help='Encoder layer config: # units in each Temporal layer')
    parser.add_argument('--position_ffn', type=str, nargs='?', default='True',
                        help='Position wise feedforward')
    parser.add_argument('--window', type=int, nargs='?', default=-1,
                        help='Window for temporal attention (default : -1 => full)')
    parser.add_argument('--method', type=str,
                    choices=['dopri5', 'adams', 'explicit_adams', 'fixed_adams','tsit5', 'euler', 'midpoint', 'rk4'],
                    default='euler')  # dopri5
    parser.add_argument('--rtol', type=float, default=0.01,
                    help='optional float64 Tensor specifying an upper bound on relative error, per element of y')
    parser.add_argument('--atol', type=float, default=0.001,
                    help='optional float64 Tensor specifying an upper bound on absolute error, per element of y')
    parser.add_argument('--n', type=int, default=400, help='Number of nodes')
    parser.add_argument('--T', type=float, default=5., help='Terminal Time')
    parser.add_argument('--layout', type=str, choices=['community', 'degree'], default='community')
    parser.add_argument('--network', type=str,
                    choices=['grid', 'random', 'power_law', 'small_world', 'community'], default='grid')
    parser.add_argument('--physical_equation', type=str, default='gene',choices=['gene','heat','mutualistic'])

    args = parser.parse_args()
    wandb.config.update(args)
    print(args)

    if args.gpu >= 0:
        device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        cudnn.deterministic = True
    setup_seed(args.seed)

    
    # Build network # A: Adjacency matrix, L: Laplacian Matrix,  OM: Base Operator
    n = args.n  # e.g nodes number 400
    N = int(np.ceil(np.sqrt(n)))  # grid-layout pixels :20
    
    if args.network == 'grid':
        print("Choose graph: " + args.network)
        A = grid_8_neighbor_graph(N)
        G = nx.from_numpy_array(A.numpy())
    elif args.network == 'random':
        print("Choose graph: " + args.network)
        G = nx.erdos_renyi_graph(n, 0.1, seed=args.seed)
        G = networkx_reorder_nodes(G, args.layout)
        A = torch.FloatTensor(nx.to_numpy_array(G))
    elif args.network == 'power_law':
        print("Choose graph: " + args.network)
        G = nx.barabasi_albert_graph(n, 5, seed=args.seed)
        G = networkx_reorder_nodes(G,  args.layout)
        A = torch.FloatTensor(nx.to_numpy_array(G))
    elif args.network == 'small_world':
        print("Choose graph: " + args.network)
        G = nx.newman_watts_strogatz_graph(400, 5, 0.5, seed=args.seed)
        G = networkx_reorder_nodes(G, args.layout)
        A = torch.FloatTensor(nx.to_numpy_array(G))
    elif args.network == 'community':
        print("Choose graph: " + args.network)
        n1 = int(n/3)
        n2 = int(n/3)
        n3 = int(n/4)
        n4 = n - n1 - n2 -n3
        G = nx.random_partition_graph([n1, n2, n3, n4], .25, .01, seed=args.seed)
        G = networkx_reorder_nodes(G, args.layout)
        A = torch.FloatTensor(nx.to_numpy_array(G))    
    
    D = torch.diag(A.sum(1))
    L = (D - A)
    
    t = torch.linspace(0., args.T, args.time_steps)  # args.time_tick) # 100 vector
    # train_deli = 80
    id_train = list(range(int(args.time_steps * 0.8))) # first 80 % for train
    id_test = list(range(int(args.time_steps * 0.8), args.time_steps)) # last 20 % for test (extrapolation)
    t_train = t[id_train]
    t_test = t[id_test]
    
    # Initial Value
    x0 = torch.zeros(N, N) 
    x0[int(0.05*N):int(0.25*N), int(0.05*N):int(0.25*N)] = 25  # x0[1:5, 1:5] = 25  for N = 20 or n= 400 case
    x0[int(0.45*N):int(0.75*N), int(0.45*N):int(0.75*N)] = 20  # x0[9:15, 9:15] = 20 for N = 20 or n= 400 case
    x0[int(0.05*N):int(0.25*N), int(0.35*N):int(0.65*N)] = 17  # x0[1:5, 7:13] = 17 for N = 20 or n= 400 case
    x0 = x0.view(-1, 1).float() 
    energy = x0.sum()
    
    with torch.no_grad():
        if args.physical_equation == 'gene':
            solution_numerical = ode.odeint(GeneDynamics(A, 1), x0, t, method='dopri5')  
        elif args.physical_equation == 'heat':
            solution_numerical = ode.odeint(HeatDiffusion(L, 1), x0, t, method='dopri5')
        elif args.physical_equation == 'mutualistic':
            solution_numerical = ode.odeint(MutualDynamics(A), x0, t, method='dopri5')
        # print(solution_numerical.shape)

    true_y = solution_numerical.squeeze().t().to(device)  # 100 * 1 * 400  --squeeze--> 100 * 400 -t-> 400 * 100
    true_y0 = x0.to(device)  # 400 * 1
    true_y_train = true_y[:, id_train].to(device)  # 400*80  for train
    true_y_test = true_y[:, id_test].to(device)  # 400*20  for extrapolation prediction

    
    # build dataloader and model
    # dataset = DynamicsDataset(args, G, A, gdv, x0)
    # dataloader = DataLoader(dataset,
    #                         batch_size=args.n,
    #                         shuffle=True,
    #                         num_workers=2,
    #                         collate_fn=DynamicsDataset.collate_fn)
    
    # sup_list = []  # List of GNN support (tensor)
    # col_net = np.zeros((A.shape[0],A.shape[1]))
    # coef_sum = 0.0
    # for i in range(80) :
    #     sup = get_gnn_sup_d(A)
    #     sup_sp = sp.coo_matrix(sup)
    #     sup_sp = sparse_to_tuple(sup_sp)
    #     idxs = torch.LongTensor(sup_sp[0].astype(float)).to(device)
    #     vals = torch.FloatTensor(sup_sp[1]).to(device)
    #     sup_tnr = torch.sparse.FloatTensor(idxs.t(), vals, sup_sp[2]).float().to(device)
    #     sup_list.append(sup_tnr)
    #     coef = 0.9**(args.time_steps-i-1)
    #     col_net += coef*A
    #     coef_sum += coef
    
    # col_net /= coef_sum
    # col_sup = get_gnn_sup_d(col_net)
    # col_sup_sp = sp.coo_matrix(col_sup)
    # col_sup_sp = sparse_to_tuple(col_sup_sp)
    # idxs = torch.LongTensor(col_sup_sp[0].astype(float)).to(device)
    # vals = torch.FloatTensor(col_sup_sp[1]).to(device)
    # col_sup_tnr = torch.sparse.FloatTensor(idxs.t(), vals, col_sup_sp[2]).float().to(device)
    
    
    A = torch.Tensor(A).to(device)
    
    model = NDCN(args,input_size=x0.shape[1], hidden_size=32, A=A, out_size=1,
                 dropout=args.dropout, no_embed=False, no_graph=False, no_control=False,
                 rtol=args.rtol, atol=args.atol, method=args.method).to(device)
    
    # model = DynamicsModel(args, num_feat=x0.shape[1], num_gdv=gdv.shape[1], time_length=args.time_steps).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    feat = torch.Tensor(x0).to(device)
    # feat_list = []
    # for i in range(len(feats)-1):
    #     feat_list.append(feat)
    
    # vt = torch.Tensor([t for t in range(args.time_steps)]).to(device)
    
    # gnd_tnr = torch.FloatTensor(adjs[-2].todense()).to(device)
    
    # in training
    best_epoch_val = 0
    patient = 0
    best_epoch_loss = float("inf")
    best_epoch_relative_loss = float("inf")
    criterion = F.l1_loss  # F.mse_loss(pred_y, true_y)
    
    
    
    
    for epoch in range(args.epochs+1):
        model.train()
        # for idx, feed_dict in enumerate(dataloader):
        #     feed_dict = to_device2(feed_dict, device)
        opt.zero_grad()
        pred_y = model(t_train, feat)
        # print(pred_y.shape)
        pred_y = torch.stack([a for a in pred_y])
        pred_y = pred_y.squeeze().t()
        loss_train = criterion(pred_y, true_y_train)
        relative_loss_train = criterion(pred_y, true_y_train) / true_y_train.mean()
        # print(loss_train)
        loss_train.backward()
        opt.step()
    
        if epoch % args.test_freq == 0:
            with torch.no_grad():
                pred_y = model(t, feat)
                pred_y = torch.stack([a for a in pred_y])
                pred_y = pred_y.squeeze().t()  # odeint(model, true_y0, t)
                loss = criterion(pred_y[:, id_test], true_y_test)
                relative_loss = criterion(pred_y[:, id_test], true_y_test) / true_y_test.mean()

                if loss<best_epoch_loss:
                    best_epoch_loss = loss
                if relative_loss<best_epoch_relative_loss:
                    best_epoch_relative_loss = relative_loss
                
                print('Epoch {:04d}| Train Loss {:.6f}({:.6f} Relative) '
                        '| Test Loss {:.6f}({:.6f} Relative) '
                        .format(epoch, loss_train.item(), relative_loss_train.item(),
                                loss.item(), relative_loss.item()))
                wandb.log({"Epoch": epoch,"Train Loss":loss_train.item()," Train Relative":relative_loss_train.item(),
                            "Test Loss":loss.item(),"Test Relative":relative_loss.item()})
    with torch.no_grad():
        pred_y = model(t, feat)
        pred_y = torch.stack([a for a in pred_y])
        pred_y = pred_y.squeeze().t()  # odeint(model, true_y0, t)
        loss = criterion(pred_y[:, id_test], true_y_test)
        relative_loss = criterion(pred_y[:, id_test], true_y_test) / true_y_test.mean()

        # if loss<best_epoch_loss:
        #     best_epoch_loss = loss
        # if relative_loss<best_epoch_relative_loss:
        #     best_epoch_relative_loss = relative_loss    
            
    print("Last Loss = {:.6f} Last Relative = {:.6f} ".format(loss,relative_loss))
    wandb.log({"Last Loss":loss,"Last Relative":relative_loss})










