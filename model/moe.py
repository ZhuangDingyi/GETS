# Note:
# Adapted form https://raw.githubusercontent.com/davidmrau/mixture-of-experts/master/moe.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from torch.distributions.normal import Normal
import numpy as np
import networkx as nx

class GCN_MoE(torch.nn.Module):
    def __init__(self,
                 num_classes, 
                 hidden_dim, 
                 dropout_rate, 
                 num_layers,
                 device,
                 expert_config,
                 feature_dim,
                 feature_hidden_dim,
                 degree_hidden_dim):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.expert_config = expert_config
        self.device = device

        in_channels = 0
        if "logits" in expert_config:
            in_channels += num_classes
        if "features" in expert_config:
            self.proj_feature = nn.Linear(feature_dim, feature_hidden_dim)
            in_channels += feature_hidden_dim
        if "degrees" in expert_config:
            in_channels += degree_hidden_dim
        for _ in range(num_layers-2):
            self.feature_list.insert(-1, hidden_dim)
        self.feature_list = [in_channels, hidden_dim, num_classes]

        layer_list = []
        for i in range(len(self.feature_list)-1):
            layer_list.append(["conv"+str(i+1), dglnn.GraphConv(self.feature_list[i], self.feature_list[i+1])])
        
        self.layer_list = torch.nn.ModuleDict(layer_list)

        self.degree_dim = degree_hidden_dim

    def forward(self, g, logits, features):
        inputs = []
        if "logits" in self.expert_config:
            inputs.append(logits)
        if "features" in self.expert_config:
            features = self.proj_feature(features)
            inputs.append(features)
        if "degrees" in self.expert_config:
            # only compute once
            if not hasattr(self, "degrees"):
                degrees = g.in_degrees() + g.out_degrees()
                max_degree = degrees.max() + 1
                self.degree_embdder = nn.Embedding(num_embeddings=max_degree, embedding_dim=self.degree_dim).to(self.device)
                self.degrees= degrees.unsqueeze(-1)
            degree_embeds = self.degree_embdder(self.degrees.squeeze(-1))
            inputs.append(degree_embeds)
        x = torch.concat(inputs,dim=-1)
        for i in range(len(self.feature_list)-1):
            x = self.layer_list["conv"+str(i+1)](g, x)
            if i < len(self.feature_list)-2:
                x = F.relu(x)
                x = F.dropout(x, self.dropout_rate, self.training)
        return x
    
class GAT_MoE(torch.nn.Module):
    def __init__(self,
                 num_classes,  
                 hidden_dim, 
                 dropout_rate, 
                 num_layers,
                 device,
                 expert_config,
                 feature_dim,
                 feature_hidden_dim,
                 degree_hidden_dim,
                 num_heads=2):  
        super().__init__()
        self.dropout_rate = dropout_rate
        self.expert_config = expert_config
        self.device = device
        self.num_heads = num_heads

        in_channels = 0
        if "logits" in expert_config:
            in_channels += num_classes 
        if "features" in expert_config:
            self.proj_feature = nn.Linear(feature_dim, feature_hidden_dim)
            in_channels += feature_hidden_dim
        if "degrees" in expert_config:
            in_channels += degree_hidden_dim
        self.feature_list = [in_channels] + [hidden_dim] * (num_layers - 1)
        layer_list = []
        for i in range(len(self.feature_list) - 1):
            layer_list.append(
                ("conv" + str(i + 1), 
                 dglnn.GATConv(self.feature_list[i], self.feature_list[i + 1] // num_heads, num_heads=num_heads))
            )

        self.layer_list = nn.ModuleDict(layer_list)
        self.degree_dim = degree_hidden_dim
        self.final_proj = nn.Linear(hidden_dim , num_classes)

    def forward(self, g, logits, features):
        inputs = []
        if "logits" in self.expert_config:
            inputs.append(logits)
        if "features" in self.expert_config:
            features = self.proj_feature(features)
            inputs.append(features)
        if "degrees" in self.expert_config:
            if not hasattr(self, "degrees"):
                degrees = g.in_degrees() + g.out_degrees()
                max_degree = degrees.max().item() + 1
                self.degree_embdder = nn.Embedding(num_embeddings=max_degree, embedding_dim=self.degree_dim).to(self.device)
                self.degrees = degrees.unsqueeze(-1)
            degree_embeds = self.degree_embdder(self.degrees.squeeze(-1))
            inputs.append(degree_embeds)
        x = torch.cat(inputs, dim=-1)
        for i in range(len(self.feature_list) - 1):
            x = self.layer_list["conv" + str(i + 1)](g, x)
            x = x.flatten(start_dim=2)              
            if i < len(self.feature_list) - 2:
                x = F.relu(x)
                x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.final_proj(x.view(x.size(0),-1))

        return x
    

class GIN_MoE(torch.nn.Module):
    def __init__(self,
                 num_classes, 
                 hidden_dim, 
                 dropout_rate, 
                 num_layers,
                 device,
                 expert_config,
                 feature_dim,
                 feature_hidden_dim,
                 degree_hidden_dim):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.expert_config = expert_config
        self.device = device

        in_channels = 0
        if "logits" in expert_config:
            in_channels += num_classes
        if "features" in expert_config:
            self.proj_feature = nn.Linear(feature_dim, feature_hidden_dim)
            in_channels += feature_hidden_dim
        if "degrees" in expert_config:
            in_channels += degree_hidden_dim

        self.feature_list = [in_channels, hidden_dim, num_classes]
        for _ in range(num_layers - 2):
            self.feature_list.insert(-1, hidden_dim)

        layer_list = []
        for i in range(len(self.feature_list) - 1):
            layer_list.append(["conv" + str(i + 1), dglnn.GINConv(
                nn.Sequential(
                    nn.Linear(self.feature_list[i], self.feature_list[i+1]),
                    nn.ReLU(),
                    nn.Linear(self.feature_list[i+1], self.feature_list[i+1])
                )
            )])

        self.layer_list = torch.nn.ModuleDict(layer_list)
        self.degree_dim = degree_hidden_dim

    def forward(self, g, logits, features):
        inputs = []
        if "logits" in self.expert_config:
            inputs.append(logits)
        if "features" in self.expert_config:
            features = self.proj_feature(features)
            inputs.append(features)
        if "degrees" in self.expert_config:
            # only compute once
            if not hasattr(self, "degrees"):
                degrees = g.in_degrees() + g.out_degrees()
                max_degree = degrees.max() + 1
                self.degree_embdder = nn.Embedding(num_embeddings=max_degree, embedding_dim=self.degree_dim).to(self.device)
                self.degrees = degrees.unsqueeze(-1)
            degree_embeds = self.degree_embdder(self.degrees.squeeze(-1))
            inputs.append(degree_embeds)

        x = torch.concat(inputs, dim=-1)
        for i in range(len(self.feature_list) - 1):
            x = self.layer_list["conv" + str(i + 1)](g, x)
            if i < len(self.feature_list) - 2:
                x = F.relu(x)
                x = F.dropout(x, self.dropout_rate, training=self.training)
        return x
class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self,
                 num_classses,
                 hidden_dim,
                 dropout_rate,
                 num_layer,
                 expert_select,
                 expert_configs,
                 feature_dim,
                 feature_hidden_dim,
                 degree_hidden_dim,
                 noisy_gating,
                 coef,
                 device,
                 backbone='gcn'):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = len(expert_configs)
        self.k = expert_select # an integer - how many experts to use for each batch element
        self.loss_coef = coef
        self.device = device
        self.backbone = backbone
        # self.k_list = k_list
        # instantiate experts
        # self.cagcn = GCN(num_class, 1, 16, drop_rate=dropout_rate, num_layers=2)
        self.proj_feature = nn.Linear(feature_dim, feature_hidden_dim)
        if backbone == 'gcn':
            self.experts = nn.ModuleList([
                GCN_MoE(
                    num_classes=num_classses, 
                    hidden_dim=hidden_dim,
                    dropout_rate=dropout_rate,
                    num_layers=num_layer,
                    device=device,
                    expert_config=expert_configs[i],
                    feature_dim=feature_dim,
                    feature_hidden_dim=feature_hidden_dim,
                    degree_hidden_dim=degree_hidden_dim,
                ) for i in range(self.num_experts)])
        elif backbone == 'gat':
            self.experts = nn.ModuleList([
                GAT_MoE(
                    num_classes=num_classses, 
                    hidden_dim=hidden_dim,
                    dropout_rate=dropout_rate,
                    num_layers=num_layer,
                    device=device,
                    expert_config=expert_configs[i],
                    feature_dim=feature_dim,
                    feature_hidden_dim=feature_hidden_dim,
                    degree_hidden_dim=degree_hidden_dim,
                ) for i in range(self.num_experts)])
        elif backbone =='gin':
            self.experts = nn.ModuleList([
                GIN_MoE(
                    num_classes=num_classses, 
                    hidden_dim=hidden_dim,
                    dropout_rate=dropout_rate,
                    num_layers=num_layer,
                    device=device,
                    expert_config=expert_configs[i],
                    feature_dim=feature_dim,
                    feature_hidden_dim=feature_hidden_dim,
                    degree_hidden_dim=degree_hidden_dim,
                ) for i in range(self.num_experts)])
        else:
            raise NotImplementedError
        self.w_gate = nn.Parameter(torch.zeros(feature_hidden_dim+num_classses, self.num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(feature_hidden_dim+num_classses, self.num_experts), requires_grad=True)
        self.topo_val = None
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob
    
    
    def noisy_top_k_gating(self, x,  train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate # size:(nums_node,nums_expert)
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k+1, self.num_experts), dim=1) 
        top_k_logits = top_logits[:, :self.k] # size:(batch_size,self.k)
        top_k_indices = top_indices[:, :self.k] # size:(batch_size,self.k)
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)  # size:(batch_size,num_experts)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load  
    
    def forward(self, g, logits, features):
        features_trans = self.proj_feature(features)
        gating_input = torch.cat([features_trans, logits], dim=1)
        node_gates, load = self.noisy_top_k_gating(gating_input, self.training) # N, |E|
        importance = node_gates.sum(0) 
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= self.loss_coef

        expert_outputs = []
        for i in range(self.num_experts):
            expert_i_output = self.experts[i](g, logits, features)
            expert_outputs.append(expert_i_output)
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # print(expert_outputs.shape)
        # print(node_gates.shape)

        temperature = (expert_outputs * node_gates.unsqueeze(-1)).sum(dim=1)
        calibrated = logits * F.softplus(temperature)
        return calibrated, loss, node_gates