# Demonstration of dyngraph2vec
import argparse
import networkx as nx
import numpy as np
# import dill
# import pickle as pkl
import scipy
# from torch.utils.data import DataLoader
import scipy.sparse as sp

from utils.preprocess import load_graphs, get_context_pairs, get_multistep_evaluation_data
# from utils.minibatch import  MyDataset
# from utils.utilities import to_device
from eval.link_prediction import evaluate_classifier
from models.gru_gcn import GRU_GCN
import torch
import torch.optim as optim
# from STGSN.modules import STGSN
# from dyngraph2vec.modules import dyngraph2vec
# from dyngraph2vec.loss import *
# from utils import *
import argparse

import wandb
import random
import os
import torch.backends.cudnn as cudnn
wandb.init(project="GRU_GCN", entity="houjinlin")
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


def inductive_graph(graph_former, graph_later):
    """Create the adj_train so that it includes nodes from (t+1) 
       but only edges from t: this is for the purpose of inductive testing.

    Args:
        graph_former ([type]): [description]
        graph_later ([type]): [description]
    """
    newG = nx.MultiGraph()
    newG.add_nodes_from(graph_later.nodes(data=True))
    newG.add_edges_from(graph_former.edges(data=False))
    return newG

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_steps', type=int, nargs='?', default=16,
                        help="total time steps used for train, eval and test")
    # Experimental settings.
    parser.add_argument('--dataset', type=str, nargs='?', default='Enron',
                        help='dataset name')
    parser.add_argument('--gpu', type=int, nargs='?', default=0,
                        help='GPU_ID (0/1 etc.)')
    parser.add_argument('--epochs', type=int, nargs='?', default=200,
                        help='# epochs')
    parser.add_argument('--val_freq', type=int, nargs='?', default=1,
                        help='Validation frequency (in epochs)')
    parser.add_argument('--test_freq', type=int, nargs='?', default=1,
                        help='Testing frequency (in epochs)')
    parser.add_argument('--batch_size', type=int, nargs='?', default=512,
                        help='Batch size (# nodes)')
    parser.add_argument('--featureless', type=bool, nargs='?', default=True,
                    help='True if one-hot encoding.')
    parser.add_argument("--early_stop", type=int, default=10,
                        help="patient")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed")
    # 1-hot encoding is input as a sparse matrix - hence no scalability issue for large datasets.
    # Tunable hyper-params
    # TODO: Implementation has not been verified, performance may not be good.
    # parser.add_argument('--residual', type=bool, nargs='?', default=True,
    #                     help='Use residual')
    # # Number of negative samples per positive pair.
    # parser.add_argument('--neg_sample_size', type=int, nargs='?', default=10,
    #                     help='# negative samples per positive')
    # # Walk length for random walk sampling.
    # parser.add_argument('--walk_len', type=int, nargs='?', default=20,
    #                     help='Walk length for random walk sampling')
    # # Weight for negative samples in the binary cross-entropy loss function.
    # parser.add_argument('--neg_weight', type=float, nargs='?', default=1.0,
    #                     help='Weightage for negative samples')
    parser.add_argument('--learning_rate', type=float, nargs='?', default=1e-4,
                        help='Initial learning rate for self-attention model.')
    # parser.add_argument('--spatial_drop', type=float, nargs='?', default=0.2,
    #                     help='Spatial (structural) attention Dropout (1 - keep probability).')
    # parser.add_argument('--temporal_drop', type=float, nargs='?', default=0.5,
                        # help='Temporal attention Dropout (1 - keep probability).')
    parser.add_argument('--weight_decay', type=float, nargs='?', default=1e-4,
                        help='Initial learning rate for self-attention model.')
    # Architecture params
    # parser.add_argument('--structural_head_config', type=str, nargs='?', default='16,8,8',
    #                     help='Encoder layer config: # attention heads in each GAT layer')
    parser.add_argument('--structural_layer_config', type=str, nargs='?', default='256',
                        help='Encoder layer config: # units in each GAT layer')
    # parser.add_argument('--temporal_head_config', type=str, nargs='?', default='16',
    #                     help='Encoder layer config: # attention heads in each Temporal layer')
    parser.add_argument('--temporal_layer_config', type=str, nargs='?', default='256',
                        help='Encoder layer config: # units in each Temporal layer')
    parser.add_argument('--position_ffn', type=str, nargs='?', default='True',
                        help='Position wise feedforward')
    parser.add_argument('--window', type=int, nargs='?', default=-1,
                        help='Window for temporal attention (default : -1 => full)')
    parser.add_argument('--tasktype', type=str, default="multisteps",choices=['siglestep','multisteps','data_scarce'])
    parser.add_argument('--scare_snapshot', type=str, default='')
    args = parser.parse_args()
    print(args)
    wandb.config.update(args)
    

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
    
    #graphs, feats, adjs = load_graphs(args.dataset)
    graphs, adjs = load_graphs(args.dataset, args.time_steps)
    if args.featureless == True:
        feats = [scipy.sparse.identity(adjs[args.time_steps - 1].shape[0]).tocsr()[range(0, x.shape[0]), :] for x in adjs if
             x.shape[0] <= adjs[args.time_steps - 1].shape[0]]

    assert args.time_steps <= len(adjs), "Time steps is illegal"

    # context_pairs_train = get_context_pairs(graphs, adjs)

    # Load evaluation data for link prediction.
    train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, \
    test_edges_pos, test_edges_neg = get_multistep_evaluation_data(graphs)
    print("No. Train1: Pos={}, Neg={} | No. Val1: Pos={}, Neg={} | No. Test1: Pos={}, Neg={}".format(
        len(train_edges_pos[0]), len(train_edges_neg[0]), len(val_edges_pos[0]), len(val_edges_neg[0]),
        len(test_edges_pos[0]), len(test_edges_neg[0])))
    print("No. Train2: Pos={}, Neg={} | No. Val2: Pos={}, Neg={} | No. Test2: Pos={}, Neg={}".format(
        len(train_edges_pos[1]), len(train_edges_neg[1]), len(val_edges_pos[1]), len(val_edges_neg[1]),
        len(test_edges_pos[1]), len(test_edges_neg[1])))
    print("No. Train3: Pos={}, Neg={} | No. Val3: Pos={}, Neg={} | No. Test3: Pos={}, Neg={}".format(
        len(train_edges_pos[2]), len(train_edges_neg[2]), len(val_edges_pos[2]), len(val_edges_neg[2]),
        len(test_edges_pos[2]), len(test_edges_neg[2])))
    print("No. Train4: Pos={}, Neg={} | No. Val4: Pos={}, Neg={} | No. Test4: Pos={}, Neg={}".format(
        len(train_edges_pos[3]), len(train_edges_neg[3]), len(val_edges_pos[3]), len(val_edges_neg[3]),
        len(test_edges_pos[3]), len(test_edges_neg[3])))
    print("No. Train5: Pos={}, Neg={} | No. Val5: Pos={}, Neg={} | No. Test5: Pos={}, Neg={}".format(
        len(train_edges_pos[4]), len(train_edges_neg[4]), len(val_edges_pos[4]), len(val_edges_neg[4]),
        len(test_edges_pos[4]), len(test_edges_neg[4])))

    # Create the adj_train so that it includes nodes from (t+1) but only edges from t: this is for the purpose of
    # inductive testing.
    new_G = inductive_graph(graphs[args.time_steps-2], graphs[args.time_steps-1])
    graphs[args.time_steps-1] = new_G
    adjs[args.time_steps-1] = nx.adjacency_matrix(new_G)

    # build dataloader and model
    # dataset = MyDataset(args, graphs, feats, adjs, context_pairs_train)
    # dataloader = DataLoader(dataset, 
    #                         batch_size=args.batch_size, 
    #                         shuffle=True, 
    #                         num_workers=0, 
    #                         collate_fn=MyDataset.collate_fn)
    #dataloader = NodeMinibatchIterator(args, graphs, feats, adjs, context_pairs_train, device) 
    # model = DySAT(args, feats[0].shape[1], args.time_steps).to(device)
    # structural = [feats[0].shape[1], int(args.structural_layer_config),int(args.structural_layer_config),int(args.temporal_layer_config)]
    # structural = [feats[0].shape[1], 256, 256, feats[0].shape[1]]
    model = GRU_GCN(int(feats[0].shape[1]), 256, int(feats[0].shape[1])).to(device)
    # model = DySAT(args, 1, args.time_steps).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # in training
    best_epoch_val = 0
    best_epoch_ap = [0,0,0,0,0]
    best_epoch_test = [0,0,0,0,0]
    patient = 0
    
    feat = np.array(feats[0].todense())
    rowsum = np.array(feat.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    feat = r_mat_inv.dot(feat)
    feat = torch.Tensor(feat)
    feat_list = list()
    feat_list.append(feat)
    for i in range(len(feats)-5):
        feat_list.append(feat)
    # for adj in adjs[:-1]:
    #     degs = np.sum(adj, axis=1)
    #     x = np.array(degs)
    #     x = x.reshape(-1,1)
    #     x = torch.Tensor(x).to(device)
    #     feat_list.append(x)
    feats = torch.stack([f for f in feat_list]).transpose(0,1).to(device)
    # feats = torch.stack([f for f in feat_list]).to(device)
    sup_list = []  # List of GNN support (tensor)
    # col_net = np.zeros((adjs[0].shape[0],adjs[0].shape[1]))
    # coef_sum = 0.0
    # for i,adj in enumerate(adjs[:-1]) :
    #     sup = get_gnn_sup_d(adj)
    #     sup_sp = sp.coo_matrix(sup)
    #     sup_sp = sparse_to_tuple(sup_sp)
    #     idxs = torch.LongTensor(sup_sp[0].astype(float)).to(device)
    #     vals = torch.FloatTensor(sup_sp[1]).to(device)
    #     sup_tnr = torch.sparse.FloatTensor(idxs.t(), vals, sup_sp[2]).float().to(device)
    #     sup_list.append(sup_tnr)
        # coef = 0.9**(args.time_steps-i-1)
        # col_net += coef*adj
        # coef_sum += coef
    sups = torch.stack([torch.Tensor(sp.coo_matrix(a, dtype=np.float32).todense()) for a in adjs[:-5]]).transpose(0,1).to(device)
    # col_net /= coef_sum
    # col_sup = get_gnn_sup_d(col_net)
    # col_sup_sp = sp.coo_matrix(col_sup)
    # col_sup_sp = sparse_to_tuple(col_sup_sp)
    # idxs = torch.LongTensor(col_sup_sp[0].astype(float)).to(device)
    # vals = torch.FloatTensor(col_sup_sp[1]).to(device)
    # col_sup_tnr = torch.sparse.FloatTensor(idxs.t(), vals, col_sup_sp[2]).float().to(device)
    
    gnd_tnr = torch.FloatTensor(adjs[-6].todense()).to(device)
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = []
        # for idx, feed_dict in enumerate(dataloader):
        #     feed_dict = to_device(feed_dict, device)
        #     opt.zero_grad()
        #     loss = model.get_loss(feed_dict)
        #     loss.backward()
        #     opt.step()
        #     epoch_loss.append(loss.item())

        opt.zero_grad()
        loss = model.get_loss(sups, feats, gnd_tnr)
        loss.backward()
        opt.step()
        epoch_loss.append(loss.item())
        
        
        
        model.eval()
        emb = model(sups, feats).detach().cpu().numpy()
        val_results1, test_results1, _, _,test_ap1 = evaluate_classifier(train_edges_pos[0],
                                                              train_edges_neg[0],
                                                              val_edges_pos[0],
                                                              val_edges_neg[0],
                                                              test_edges_pos[0],
                                                              test_edges_neg[0],
                                                              emb,
                                                              emb)
        epoch_auc_val1 = val_results1["HAD"][1]
        epoch_auc_test1 = test_results1["HAD"][1]

        val_results2, test_results2, _, _,test_ap2 = evaluate_classifier(train_edges_pos[1],
                                                              train_edges_neg[1],
                                                              val_edges_pos[1],
                                                              val_edges_neg[1],
                                                              test_edges_pos[1],
                                                              test_edges_neg[1],
                                                              emb,
                                                              emb)
        epoch_auc_val2 = val_results2["HAD"][1]
        epoch_auc_test2 = test_results2["HAD"][1]
        
        val_results3, test_results3, _, _,test_ap3 = evaluate_classifier(train_edges_pos[2],
                                                              train_edges_neg[2],
                                                              val_edges_pos[2],
                                                              val_edges_neg[2],
                                                              test_edges_pos[2],
                                                              test_edges_neg[2],
                                                              emb,
                                                              emb)
        epoch_auc_val3 = val_results3["HAD"][1]
        epoch_auc_test3 = test_results3["HAD"][1]
        
        val_results4, test_results4, _, _,test_ap4 = evaluate_classifier(train_edges_pos[3],
                                                              train_edges_neg[3],
                                                              val_edges_pos[3],
                                                              val_edges_neg[3],
                                                              test_edges_pos[3],
                                                              test_edges_neg[3],
                                                              emb,
                                                              emb)
        epoch_auc_val4 = val_results4["HAD"][1]
        epoch_auc_test4 = test_results4["HAD"][1]
        
        val_results5, test_results5, _, _,test_ap5 = evaluate_classifier(train_edges_pos[4],
                                                              train_edges_neg[4],
                                                              val_edges_pos[4],
                                                              val_edges_neg[4],
                                                              test_edges_pos[4],
                                                              test_edges_neg[4],
                                                              emb,
                                                              emb)
        epoch_auc_val5 = val_results5["HAD"][1]
        epoch_auc_test5 = test_results5["HAD"][1]

        if epoch_auc_val1 > best_epoch_val:
            best_epoch_val = epoch_auc_val1
            best_epoch_test = [epoch_auc_test1,epoch_auc_test2,epoch_auc_test3,epoch_auc_test4,epoch_auc_test5]
            best_epoch_ap = [test_ap1,test_ap2,test_ap3,test_ap4,test_ap5]
            torch.save(model.state_dict(), "./model_checkpoints/model_{}.pt".format(args.dataset))
            patient = 0
        else:
            patient += 1
            if patient > args.early_stop:
                break

        print("Epoch {:<3},  Loss = {:.3f}, Val1 AUC {:.4f} Test1 AUC {:.4f}\n Val2 AUC {:.4f} Test2 AUC {:.4f}\n Val3 AUC {:.4f} Test3 AUC {:.4f}\n Val4 AUC {:.4f} Test4 AUC {:.4f}\n Val5 AUC {:.4f} Test5 AUC {:.4f}".format(epoch,
                                                                                   np.mean(epoch_loss),
                                                                                   epoch_auc_val1,
                                                                                   epoch_auc_test1,
                                                                                   epoch_auc_val2,
                                                                                   epoch_auc_test2,
                                                                                   epoch_auc_val3,
                                                                                   epoch_auc_test3,
                                                                                   epoch_auc_val4,
                                                                                   epoch_auc_test4,
                                                                                   epoch_auc_val5,
                                                                                   epoch_auc_test5
                                                                                   ))
        wandb.log({"Epoch": epoch,"loss":np.mean(epoch_loss),
                   "Val1 AUC":epoch_auc_val1,"Test1 AUC":epoch_auc_test1,
                   "Val2 AUC":epoch_auc_val2,"Test2 AUC":epoch_auc_test2,
                   "Val3 AUC":epoch_auc_val3,"Test3 AUC":epoch_auc_test3,
                   "Val4 AUC":epoch_auc_val4,"Test4 AUC":epoch_auc_test4,
                   "Val5 AUC":epoch_auc_val5,"Test5 AUC":epoch_auc_test5,"Test AP1":test_ap1,"Test AP2":test_ap2,"Test AP3":test_ap3,"Test AP4":test_ap4,"Test AP5":test_ap5})
                                                       
    # Test Best Model
    # model.load_state_dict(torch.load("./model_checkpoints/model.pt"))
    # model.eval()
    # emb = model(sup_list).detach().cpu().numpy()
    # val_results, test_results, _, _ = evaluate_classifier(train_edges_pos,
    #                                                     train_edges_neg,
    #                                                     val_edges_pos, 
    #                                                     val_edges_neg, 
    #                                                     test_edges_pos,
    #                                                     test_edges_neg, 
    #                                                     emb, 
    #                                                     emb)
    # auc_val = val_results["HAD"][1]
    # auc_test = test_results["HAD"][1]
    print("Best Test1 AUC = {:.4f} | Best Test2 AUC = {:.4f} | Best Test3 AUC = {:.4f} | Best Test4 AUC = {:.4f} | Best Test5 AUC = {:.4f} |".format(best_epoch_test[0],
                                                                                                                                                     best_epoch_test[1],
                                                                                                                                                     best_epoch_test[2],
                                                                                                                                                     best_epoch_test[3],
                                                                                                                                                     best_epoch_test[4]))
    wandb.log({"Best Test1 AUC":best_epoch_test[0],
               "Best Test2 AUC":best_epoch_test[1],
               "Best Test3 AUC":best_epoch_test[2],
               "Best Test4 AUC":best_epoch_test[3],
               "Best Test5 AUC":best_epoch_test[4],"Best AP1":best_epoch_ap[0],"Best AP2":best_epoch_ap[1],"Best AP3":best_epoch_ap[2],"Best AP4":best_epoch_ap[3],"Best AP5":best_epoch_ap[4]})


