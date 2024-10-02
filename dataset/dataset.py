import torch
from dgl import AddSelfLoop, DGLGraph
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset, RedditDataset, CoraFullDataset, AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset, CoauthorCSDataset, CoauthorPhysicsDataset
from ogb.nodeproppred import DglNodePropPredDataset
import numpy as np
import networkx as nx

class Dataset:
    def __init__(self, ds_name, n_runs=1):
        self.ds_name = ds_name
        self.n_runs = n_runs
        self.device = torch.device('cuda')
        data = load_dataset(ds_name)
        self.g, self.features, self.labels, self.num_classes = self._prepare_data(data)
        self.train_idxs, self.val_idxs, self.test_idxs = self._split_data(data)
        print(f"Dataset: {ds_name} | #Nodes: {self.g.number_of_nodes()} | #Edges: {self.g.number_of_edges()} | #Classes: {self.num_classes} |#Features: {self.features.shape[1]}")
    
    def _prepare_data(self, data):
        if self.ds_name in ["cora", "citeseer", "pubmed", "reddit"]:
            g = data[0]
            g = g.int().to(self.device)

            features = g.ndata["feat"]

            labels = g.ndata["label"]

        elif self.ds_name in ["ogbn-arxiv"]:
            g, labels = data[0]
            g = g.int().to(self.device)
            srcs, dsts = g.all_edges()
            g.add_edges(dsts, srcs)
            g = g.remove_self_loop().add_self_loop()
            g.create_formats_()

            features = g.ndata["feat"]

            labels = labels[:, 0].to(self.device)

        elif self.ds_name  in ["cora-full", "computers", "photo", "cs", "physics"]:
            g = data[0]
            features = g.ndata["feat"]
            labels = g.ndata["label"]

            # Find the largest connected component
            nx_g = g.to_networkx()
            nx_g = nx_g.to_undirected()
            largest_cc = max(nx.connected_components(nx_g), key=len)
            subgraph = nx_g.subgraph(largest_cc).copy()
            g = DGLGraph(subgraph).to(self.device)

            features = features[list(largest_cc)].to(self.device)
            labels = labels[list(largest_cc)].to(self.device)
    
        return g, features, labels, data.num_classes
    
    def _split_data(self, data):
        train_idxs = []
        val_idxs = []
        test_idxs = []
        if self.ds_name in ["cora", "citeseer", "pubmed", "reddit"]:
            # train_idx, val_idx, test_idx = self.g.ndata["train_mask"], self.g.ndata["val_mask"], self.g.ndata["test_mask"]
            idx = np.array(range(len(self.labels)))
            np.random.shuffle(idx)
            split_res = np.split(idx, [int(0.2 * len(idx)), int(0.3 * len(idx))])
            train_idx, val_idx, test_idx = split_res[0], split_res[1], split_res[2]
            for _ in range(self.n_runs):
                train_idxs.append(train_idx)
                val_idxs.append(val_idx)
                test_idxs.append(test_idx)
        elif self.ds_name in ["ogbn-arxiv"]:
            # splitted_idx = data.get_idx_split()
            # train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
            idx = np.array(range(len(self.labels)))
            np.random.shuffle(idx)
            split_res = np.split(idx, [int(0.2 * len(idx)), int(0.3 * len(idx))])
            train_idx, val_idx, test_idx = split_res[0], split_res[1], split_res[2]
            for _ in range(self.n_runs):
                train_idxs.append(train_idx)
                val_idxs.append(val_idx)
                test_idxs.append(test_idx)
        elif self.ds_name  in ["cora-full", "computers", "photo", "cs", "physics"]:
            idx = np.array(range(len(self.labels)))
            np.random.shuffle(idx)
            split_res = np.split(idx, [int(0.2 * len(idx)), int(0.3 * len(idx))])
            train_idx, val_idx, test_idx = split_res[0], split_res[1], split_res[2]
 
            for _ in range(self.n_runs):
                train_idxs.append(np.array(train_idx))
                val_idxs.append(np.array(val_idx))
                test_idxs.append(np.array(test_idx))
        
        return train_idxs, val_idxs, test_idxs
        
def load_dataset(ds_name):
    if ds_name== "cora":
        data = CoraGraphDataset(transform=AddSelfLoop())
    elif ds_name == "citeseer":
        data = CiteseerGraphDataset(transform=AddSelfLoop())
    elif ds_name == "pubmed":
        data = PubmedGraphDataset(transform=AddSelfLoop())
    elif ds_name == "reddit":
        data = RedditDataset(transform=AddSelfLoop())
    elif ds_name == "cora-full":
        data = CoraFullDataset(transform=AddSelfLoop())
    elif ds_name == "computers":
        data = AmazonCoBuyComputerDataset()
    elif ds_name == "photo":
        data = AmazonCoBuyPhotoDataset()
    elif ds_name == "cs":
        data = CoauthorCSDataset()
    elif ds_name == "physics":
        data = CoauthorPhysicsDataset()
    elif ds_name == "ogbn-arxiv":
        data = DglNodePropPredDataset(name="ogbn-arxiv")
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")
    return data