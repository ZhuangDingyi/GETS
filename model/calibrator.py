from typing import Sequence
import numpy as np
import scipy
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression
import copy
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
import dgl
import dgl.nn as dglnn
from model.GETS import GETS


def fit_calibration(temp_model, eval, g, features, labels, masks, epochs, patience):
    train_idx = masks[1]
    val_idx = masks[0]
    vlss_mn = float('Inf')
    with torch.no_grad():
        logits = temp_model.model(g, features)
        model_dict = temp_model.state_dict()
        parameters = {k: v for k,v in model_dict.items() if k.split(".")[0] != "model"}
    for epoch in range(epochs):
        temp_model.optimizer.zero_grad()
        temp_model.train()
        # Post-hoc calibration set the classifier to the evaluation mode
        temp_model.model.eval()
        assert not temp_model.model.training
        ret = eval(logits)
        loss_load = None
        if isinstance(ret, tuple):
            calibrated, loss_load, _ = ret
        else:
            calibrated = ret
        loss = F.cross_entropy(calibrated[train_idx], labels[train_idx])
        if loss_load is not None:
            loss += loss_load
        loss.backward()
        temp_model.optimizer.step()

        with torch.no_grad():
            temp_model.eval()
            ret = eval(logits)
            loss_load = None
            if isinstance(ret, tuple):
                calibrated, loss_load, _ = ret
            else:
                calibrated = ret
            val_loss = F.cross_entropy(calibrated[val_idx], labels[val_idx])
            flag = False
            if val_loss <= vlss_mn:
                flag = True
                with torch.no_grad():
                    logits = temp_model.model(g, features)
                    model_dict = temp_model.state_dict()
                    parameters = {k: v for k,v in model_dict.items() if k.split(".")[0] != "model"}
                state_dict_early_model = copy.deepcopy(parameters)
                vlss_mn = np.min((val_loss.cpu().numpy(), vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step >= patience:
                    break
        if isinstance(ret, tuple):
            print("Epoch {:05d} | Loss(calibration) {:.4f} | Loss(load) {:.4f} |{}"
                  .format(epoch + 1, val_loss.item(), loss_load.item(), "*" if flag else ""))
    model_dict.update(state_dict_early_model)
    temp_model.load_state_dict(model_dict)

class ETS(nn.Module):
    def __init__(self, model, num_classes, device, conf):
        super().__init__()
        self.model = model
        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.zeros(1))
        self.weight3 = nn.Parameter(torch.zeros(1))
        self.num_classes = num_classes
        self.temp_model = TS(model, device, conf)
        self.device = device
        self.conf = conf
    def forward(self, g, features):
        logits = self.model(g, features)
        temp = self.temp_model.temperature_scale(logits)
        p = self.w1 * F.softmax(logits / temp, dim=1) + self.w2 * F.softmax(logits, dim=1) + self.w3 * 1/self.num_classes
        return torch.log(p)

    def fit(self, g, features, labels, masks):
        self.to(self.device)
        self.temp_model.fit(g, features, labels, masks)
        torch.cuda.empty_cache()
        logits = self.model(g, features)[masks[1]]
        label = labels[masks[1]]
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.unsqueeze(-1), 1)
        temp = self.temp_model.temperature.cpu().detach().numpy()
        w = self.ensemble_scaling(logits.cpu().detach().numpy(), one_hot.cpu().detach().numpy(), temp)
        self.w1, self.w2, self.w3 = w[0], w[1], w[2]
        return self

    def ensemble_scaling(self, logit, label, t):
        """
        Official ETS implementation from Mix-n-Match: Ensemble and Compositional Methods for Uncertainty Calibration in Deep Learning
        Code taken from (https://github.com/zhang64-llnl/Mix-n-Match-Calibration)
        Use the scipy optimization because PyTorch does not have constrained optimization.
        """
        p1 = np.exp(logit)/np.sum(np.exp(logit),1)[:,None]
        logit = logit/t
        p0 = np.exp(logit)/np.sum(np.exp(logit),1)[:,None]
        p2 = np.ones_like(p0)/self.num_classes
        

        bnds_w = ((0.0, 1.0),(0.0, 1.0),(0.0, 1.0),)
        def my_constraint_fun(x): return np.sum(x)-1
        constraints = { "type":"eq", "fun":my_constraint_fun,}
        w = scipy.optimize.minimize(ETS.ll_w, (1.0, 0.0, 0.0), args = (p0,p1,p2,label), method='SLSQP', constraints = constraints, bounds=bnds_w, tol=1e-12, options={'disp': False})
        w = w.x
        return w

    @staticmethod
    def ll_w(w, *args):
    ## find optimal weight coefficients with Cros-Entropy loss function
        p0, p1, p2, label = args
        p = (w[0]*p0+w[1]*p1+w[2]*p2)
        N = p.shape[0]
        ce = -np.sum(label*np.log(p))/N
        return ce       
    
class TS(nn.Module):
    def __init__(self, model, device, conf):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))
        self.device = device
        self.conf = conf
    def forward(self, g, features):
        logits = self.model(g, features)
        temperature = self.temperature_scale(logits)
        return logits / temperature

    def temperature_scale(self, logits):
        """
        Expand temperature to match the size of logits
        """
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return temperature

    def fit(self, g, features, labels, masks):
        self.to(self.device)
        def eval(logits):
            temperature = self.temperature_scale(logits)
            calibrated = logits / temperature
            return calibrated
        
        self.train_param = [self.temperature]
        self.optimizer = optim.Adam(self.train_param, lr=self.conf.calibration["cal_lr"], weight_decay=self.conf.calibration["cal_weight_decay"])
        fit_calibration(self, eval, g, features, labels, masks, self.conf.calibration["epochs"], self.conf.calibration["patience"])
        return self

class VS(nn.Module):
    def __init__(self, model, num_classes, device, conf):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(num_classes))
        self.bias = nn.Parameter(torch.ones(num_classes))
        self.device = device
        self.conf = conf
    def forward(self, x, edge_index):
        logits = self.model(x, edge_index)
        temperature = self.vector_scale(logits)
        return logits * temperature + self.bias

    def vector_scale(self, logits):
        """
        Expand temperature to match the size of logits
        """
        temperature = self.temperature.unsqueeze(0).expand(logits.size(0), logits.size(1))
        return temperature

    def fit(self, g, features, labels, masks):
        self.to(self.device)
        def eval(logits):
            temperature = self.vector_scale(logits)
            calibrated = logits * temperature + self.bias
            return calibrated

        self.train_param = [self.temperature]
        self.optimizer = optim.Adam(self.train_param, lr=self.conf.calibration["cal_lr"], weight_decay=self.conf.calibration["cal_weight_decay"])
        fit_calibration(self, eval, g, features, labels, masks, self.conf.calibration["epochs"], self.conf.calibration["patience"])
        return self

class GCN_pure(torch.nn.Module):
    def __init__(self, in_channels, num_classes, num_hidden, drop_rate, num_layers):
        super().__init__()
        self.drop_rate = drop_rate
        self.feature_list = [in_channels, num_hidden, num_classes]
        for _ in range(num_layers-2):
            self.feature_list.insert(-1, num_hidden)
        layer_list = []

        for i in range(len(self.feature_list)-1):
            layer_list.append(["conv"+str(i+1), dglnn.GraphConv(self.feature_list[i], self.feature_list[i+1])])
        
        self.layer_list = torch.nn.ModuleDict(layer_list)

    def forward(self, features, g):
        x = features
        for i in range(len(self.feature_list)-1):
            x = self.layer_list["conv"+str(i+1)](g, x)
            if i < len(self.feature_list)-2:
                x = F.relu(x)
                x = F.dropout(x, self.drop_rate, self.training)
        return x
    
class GCN(torch.nn.Module):
    def __init__(self, in_channels, num_classes, num_hidden, drop_rate, num_layers):
        super().__init__()
        self.drop_rate = drop_rate
        self.feature_list = [in_channels, num_hidden, num_classes]
        for _ in range(num_layers-2):
            self.feature_list.insert(-1, num_hidden)
        layer_list = []

        for i in range(len(self.feature_list)-1):
            layer_list.append(["conv"+str(i+1), dglnn.GraphConv(self.feature_list[i], self.feature_list[i+1])])
        
        self.layer_list = torch.nn.ModuleDict(layer_list)

    def forward(self, features, g):
        x = features
        for i in range(len(self.feature_list)-1):
            x = self.layer_list["conv"+str(i+1)](g, x)
            if i < len(self.feature_list)-2:
                x = F.relu(x)
                x = F.dropout(x, self.drop_rate, self.training)
        return x
    
class CaGCN(nn.Module):
    def __init__(self, model, num_class, device, conf):
        super().__init__()
        self.model = model
        self.cagcn = GCN(num_class, 1, 16, drop_rate=conf.calibration["cal_dropout"], num_layers=2)
        self.device = device
        self.conf = conf

    def forward(self, g, features):
        logits = self.model(g, features)
        temperature = self.graph_temperature_scale(logits, g)
        return logits * F.softplus(temperature)

    def graph_temperature_scale(self, logits, g):
        """
        Perform graph temperature scaling on logits
        """
        temperature = self.cagcn(logits, g)
        return temperature

    def fit(self, g, features, labels, masks):
        self.to(self.device)
        def eval(logits):
            temperature = self.graph_temperature_scale(logits, g)
            calibrated = logits * F.softplus(temperature)
            return calibrated

        self.train_param = self.cagcn.parameters()
        self.optimizer = optim.Adam(self.train_param, lr=self.conf.calibration["cal_lr"], weight_decay=self.conf.calibration["cal_weight_decay"])
        fit_calibration(self, eval, g, features, labels, masks, self.conf.calibration["epochs"], self.conf.calibration["patience"])
        return self

class CaGCN_GETS(nn.Module):
    def __init__(self, model, feature_dim, num_class, device, conf):
        super().__init__()
        self.model = model
        self.device = device

        self.learner = GETS(
            num_classses=num_class,
            hidden_dim=conf.calibration["hidden_dim"],
            dropout_rate=conf.calibration["cal_dropout"],
            num_layer=conf.calibration["cal_num_layer"],
            expert_select=conf.calibration["expert_select"],
            expert_configs=conf.calibration["expert_configs"],
            feature_dim=feature_dim,
            feature_hidden_dim=conf.calibration["feature_hidden_dim"],
            degree_hidden_dim=conf.calibration["degree_hidden_dim"],
            noisy_gating=conf.calibration["noisy_gating"],
            coef=conf.calibration["coef"],
            device=device,
            backbone=conf.calibration['backbone']
        )
        self.conf = conf
        
    def forward(self, g, features):
        logits = self.model(g, features)
        return self.learner(g, logits, features)
    
    def fit(self, g, features, labels, masks):
        self.to(self.device)
        def eval(logits):
            return self.learner(g, logits, features)

        self.train_param = self.parameters()
        self.optimizer = optim.Adam(self.train_param, lr=self.conf.calibration["cal_lr"], weight_decay=self.conf.calibration["cal_weight_decay"])
        fit_calibration(self, eval, g, features, labels, masks, self.conf.calibration["epochs"], self.conf.calibration["patience"])
        return self

    
from typing import Union, Optional
from torch_geometric.typing import OptPairTensor, Adj, OptTensor
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree

def shortest_path_length(edge_index, mask, max_hop, device):
    """
    Return the shortest path length to the mask for every node
    """
    dist_to_train = torch.ones_like(mask, dtype=torch.long, device=device) * torch.iinfo(torch.long).max
    seen_mask = torch.clone(mask).to(device)
    for hop in range(max_hop):
        current_hop = torch.nonzero(mask).to(device)
        dist_to_train[mask] = hop
        next_hop = torch.zeros_like(mask, dtype=torch.bool, device=device)
        for node in current_hop:
            node_mask = edge_index[0,:]==node
            nbrs = edge_index[1,node_mask]
            next_hop[nbrs] = True
        hop += 1
        # mask for the next hop shouldn't be seen before
        mask = torch.logical_and(next_hop, ~seen_mask)
        seen_mask[next_hop] = True
    return dist_to_train   

class CalibAttentionLayer(MessagePassing):
    _alpha: OptTensor

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            edge_index: Adj,
            num_nodes: int,
            train_mask: Tensor,
            dist_to_train: Tensor = None,
            heads: int = 8,
            negative_slope: float = 0.2,
            bias: float = 1,
            self_loops: bool = True,
            fill_value: Union[float, Tensor, str] = 'mean',
            bfs_depth=2,
            device='cpu',
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.fill_value = fill_value
        self.edge_index = edge_index
        self.num_nodes = num_nodes

        self.temp_lin = Linear(in_channels, heads,
                               bias=False, weight_initializer='glorot')

        # The learnable clustering coefficient for training node and their neighbors
        self.conf_coef = Parameter(torch.zeros([]))
        self.bias = Parameter(torch.ones(1) * bias)
        self.train_a = Parameter(torch.ones(1))
        self.dist1_a = Parameter(torch.ones(1))

        # Compute the distances to the nearest training node of each node
        train_mask_indices_tensor = torch.from_numpy(train_mask).to(device)
        train_mask_tensor = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        train_mask_tensor.scatter_(0, train_mask_indices_tensor, True)
        dist_to_train = dist_to_train if dist_to_train is not None else shortest_path_length(edge_index, train_mask_tensor, bfs_depth, device)
        self.register_buffer('dist_to_train', dist_to_train)

        self.reset_parameters()
        if self_loops:
            # We only want to add self-loops for nodes that appear both as
            # source and target nodes:
            self.edge_index, _ = remove_self_loops(
                self.edge_index, None)
            self.edge_index, _ = add_self_loops(
                self.edge_index, None, fill_value=self.fill_value,
                num_nodes=num_nodes)

    def reset_parameters(self):
        self.temp_lin.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor]):
        N, H = self.num_nodes, self.heads

        # Individual Temperature
        normalized_x = x - torch.min(x, 1, keepdim=True)[0]
        normalized_x /= torch.max(x, 1, keepdim=True)[0] - \
                        torch.min(x, 1, keepdim=True)[0]

        # t_delta for individual nodes
        # x_sorted_scalar: [N, 1]
        x_sorted = torch.sort(normalized_x, -1)[0]
        temp = self.temp_lin(x_sorted)

        # Next, we assign spatial coefficient
        # a_cluster:[N]
        a_cluster = torch.ones(N, dtype=torch.float32, device=x[0].device)
        a_cluster[self.dist_to_train == 0] = self.train_a
        a_cluster[self.dist_to_train == 1] = self.dist1_a


        # For confidence smoothing
        conf = F.softmax(x, dim=1).amax(-1)
        deg = degree(self.edge_index[0, :], self.num_nodes)
        deg_inverse = 1 / deg
        deg_inverse[deg_inverse == float('inf')] = 0

        out = self.propagate(self.edge_index,
                             temp=temp.view(N, H) * a_cluster.unsqueeze(-1),
                             alpha=x / a_cluster.unsqueeze(-1),
                             conf=conf)
        sim, dconf = out[:, :-1], out[:, -1:]
        out = F.softplus(sim + self.conf_coef * dconf * deg_inverse.unsqueeze(-1))
        out = out.mean(dim=1) + self.bias 
        return out.unsqueeze(1)

    def message(
            self,
            temp_j: Tensor,
            alpha_j: Tensor,
            alpha_i: OptTensor,
            conf_i: Tensor,
            conf_j: Tensor,
            index: Tensor,
            ptr: OptTensor,
            size_i: Optional[int]) -> Tensor:
        """
        alpha_i, alpha_j: [E, H]
        temp_j: [E, H]
        """
        if alpha_i is None:
            print("alphai is none")
        alpha = (alpha_j * alpha_i).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        # Agreement smoothing + Confidence smoothing
        return torch.cat([
            (temp_j * alpha.unsqueeze(-1).expand_as(temp_j)),
            (conf_i - conf_j).unsqueeze(-1)], -1)

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}{self.out_channels}, heads={self.heads}')

    
class GATS(nn.Module):
    def __init__(self, model, g, num_class, train_mask, device, conf):
        super().__init__()
        self.model = model
        src_nodes, dst_nodes = g.edges()
        self.edge_index = torch.stack([src_nodes, dst_nodes], dim=0)
        self.num_nodes = g.num_nodes()
        self.conf = conf
        self.cagat = CalibAttentionLayer(in_channels=num_class,
                                         out_channels=1,
                                         edge_index=self.edge_index,
                                         num_nodes=self.num_nodes,
                                         train_mask=train_mask,
                                         dist_to_train=conf.calibration["dist_to_train"],
                                         heads=conf.calibration["heads"],
                                         bias=conf.calibration["bias"],
                                         device = device)
        self.device = device
        
    def forward(self, g, features):
        logits = self.model(g, features)
        temperature = self.graph_temperature_scale(logits)
        return logits / temperature

    def graph_temperature_scale(self, logits):
        """
        Perform graph temperature scaling on logits
        """
        temperature = self.cagat(logits).view(self.num_nodes, -1)
        return temperature.expand(self.num_nodes, logits.size(1))

    def fit(self, g, features, labels, masks):
        self.to(self.device)
        def eval(logits):
            temperature = self.graph_temperature_scale(logits)
            calibrated = logits / temperature
            return calibrated

        self.train_param = self.cagat.parameters()
        self.optimizer = optim.Adam(self.train_param, lr=self.conf.calibration["cal_lr"], weight_decay=self.conf.calibration["cal_weight_decay"])
        fit_calibration(self, eval, g, features, labels, masks, self.conf.calibration["epochs"], self.conf.calibration["patience"])
        return self