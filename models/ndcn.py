import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq as ode
from utils import *

class ODEFunc(nn.Module):  # A kind of ODECell in the view of RNN
    def __init__(self, hidden_size, A, dropout=0.0, no_graph=False, no_control=False):
        super(ODEFunc, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.A = A  # N_node * N_node
        # self.nfe = 0
        self.wt = nn.Linear(hidden_size, hidden_size)
        self.no_graph = no_graph
        self.no_control = no_control

    def forward(self, t, x):  # How to use t?
        """
        :param t:  end time tick, if t is not used, it is an autonomous system
        :param x:  initial value   N_node * N_dim   400 * hidden_size
        :return:
        """
        # self.nfe += 1
        if not self.no_graph:
            if hasattr(self.A, 'is_sparse') and self.A.is_sparse:
                x = torch.sparse.mm(self.A, x)
            else:
                x = torch.mm(self.A, x)
        if not self.no_control:
            x = self.wt(x)
        x = self.dropout_layer(x)
        # x = torch.tanh(x)
        x = F.relu(x)
        # !!!!! Not use relu seems doesn't  matter!!!!!! in theory. Converge faster !!! Better than tanh??
        # x = torch.sigmoid(x)
        return x


class ODEBlock(nn.Module):
    def __init__(self, odefunc, rtol=.01, atol=.001, method='dopri5', adjoint=False, terminal=False): #vt,         :param vt:
        """
        :param odefunc: X' = f(X, t, G, W)
        :param rtol: optional float64 Tensor specifying an upper bound on relative error,
            per element of `y`.
        :param atol: optional float64 Tensor specifying an upper bound on absolute error,
            per element of `y`.
        :param method:
            'explicit_adams': AdamsBashforth,
            'fixed_adams': AdamsBashforthMoulton,
            'adams': VariableCoefficientAdamsBashforth,
            'tsit5': Tsit5Solver,
            'dopri5': Dopri5Solver,
            'euler': Euler,
            'midpoint': Midpoint,
            'rk4': RK4,
        """

        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        # self.integration_time_vector = vt  # time vector
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.adjoint = adjoint
        self.terminal = terminal

    def forward(self, vt, x):
        integration_time_vector = vt.type_as(x)
        if self.adjoint:
            out = ode.odeint_adjoint(self.odefunc, x, integration_time_vector,
                                     rtol=self.rtol, atol=self.atol, method=self.method)
        else:
            out = ode.odeint(self.odefunc, x, integration_time_vector,
                             rtol=self.rtol, atol=self.atol, method=self.method)
        # return out[-1]
        return out   # 100 * 400 * 10


class ODEBlock2(nn.Module):
    def __init__(self, odefunc, vt, rtol=.01, atol=.001, method='dopri5', adjoint=False, terminal=False): #vt,         :param vt:
        """
        :param odefunc: X' = f(X, t, G, W)
        :param rtol: optional float64 Tensor specifying an upper bound on relative error,
            per element of `y`.
        :param atol: optional float64 Tensor specifying an upper bound on absolute error,
            per element of `y`.
        :param method:
            'explicit_adams': AdamsBashforth,
            'fixed_adams': AdamsBashforthMoulton,
            'adams': VariableCoefficientAdamsBashforth,
            'tsit5': Tsit5Solver,
            'dopri5': Dopri5Solver,
            'euler': Euler,
            'midpoint': Midpoint,
            'rk4': RK4,
        """

        super(ODEBlock2, self).__init__()
        self.odefunc = odefunc
        self.integration_time_vector = vt  # time vector
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.adjoint = adjoint
        self.terminal = terminal

    def forward(self,   x):
        integration_time_vector = self.integration_time_vector.type_as(x)
        if self.adjoint:
            out = ode.odeint_adjoint(self.odefunc, x, integration_time_vector,
                                     rtol=self.rtol, atol=self.atol, method=self.method)
        else:
            out = ode.odeint(self.odefunc, x, integration_time_vector,
                             rtol=self.rtol, atol=self.atol, method=self.method)
        # return out[-1]
        return out[-1] if self.terminal else out  # 100 * 400 * 10


class NDCN(nn.Module):  # myModel
    def __init__(self, args, input_size, hidden_size, A, out_size,  dropout=0.0,
                 no_embed=False, no_graph=False, no_control=False,
                 rtol=.01, atol=.001, method='dopri5'):
        super(NDCN, self).__init__()
        self.args = args 
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.A = A  # N_node * N_node
        self.num_classes = out_size

        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

        self.no_embed = no_embed
        self.no_graph = no_graph
        self.no_control = no_control

        self.rtol = rtol
        self.atol = atol
        self.method = method

        self.input_layer = nn.Sequential(nn.Linear(input_size, hidden_size, bias=True), nn.Tanh(),
                               nn.Linear(hidden_size, hidden_size, bias=True))
        self.neural_dynamic_layer = ODEBlock(
                ODEFunc(hidden_size, A, dropout=dropout, no_graph=no_graph, no_control=no_control),  # OM
                rtol= rtol, atol= atol, method= method)  # t is like  continuous depth
        self.output_layer = nn.Linear(hidden_size, out_size, bias=False)

    def forward(self, vt, x):  # How to use t?
        """
        :param t:  end time tick, if t is not used, it is an autonomous system
        :param x:  initial value   N_node * N_dim   400 * hidden_size
        :return:
        """
        if not self.no_embed:
            x = self.input_layer(x)
        hvx = self.neural_dynamic_layer(vt, x)
        emb=list()
        for out in hvx[-6:]:
        # for out in hvx:
            out = self.output_layer(out)
            emb.append(out)
        return emb
    
    def get_loss(self, vt,feat, gnd_tnr):
        # graphs = feed_dict["graphs"]
        # run gnn
        # graphs = graphs[:-1]
        # gnd = torch_geometric.utils.to_scipy_sparse_matrix(graphs[-1].edge_index).todense() 
        # final_emb = self.forward(graphs)
        
        final_emb = self.forward(vt,feat)
        if self.args.tasktype =='multisteps':
            final_emb = final_emb[-6]
        else:
            final_emb = final_emb[-2]
        # if self.args.multisteps_pre == "True":   
        #     final_emb = final_emb[-6]
        # else:
        #     final_emb = final_emb[-2]
        self.graph_loss = 0
        loss = torch.norm((final_emb - gnd_tnr), p='fro')**2
        self.graph_loss += loss
        return self.graph_loss  