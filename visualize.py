import argparse
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch
import os
import random
from dataset.dataset import Dataset

num_node = 1000
figsize = (28, 20)
lengend_size = 30
predefined_colors = [
    "#7DC0A7",
    "#ED936B",
    "#919FC7",
    "#DA8EC0",
    "#B0D767",
    "#F9DA56"
]

predefined_experts = [
    "logits",
    "features",
    "degrees",
    "logits + features",
    "features + degrees",
    "logits + degrees"
]

def bfs_subgraph(G, start_node, num_nodes):
    bfs_edges = list(nx.bfs_edges(G, start_node))
    nodes = set([start_node])
    for u, v in bfs_edges:
        if len(nodes) >= num_nodes:
            break
        nodes.add(u)
        nodes.add(v)
    return G.subgraph(nodes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora", help="Choose from: [cora, citeseer, pubmed, cora-full, computers, photo, cs, physics, ogbn-arxiv]")
    parser.add_argument("--gpu", type=int, default=0, help="Use which gpu")
    parser.add_argument("--no", type=int, default=1)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    dataset = Dataset(ds_name=args.dataset, n_runs=1)
    G = dataset.g.cpu().to_networkx().to_undirected()
    G.remove_edges_from(nx.selfloop_edges(G))
    node_gates = torch.load(f"output/MoE/{args.dataset}/node_gates.pt")

    start_node = random.choice(list(G.nodes()))
    subgraph = bfs_subgraph(G, start_node, num_node)
    subgraph_nodes = list(subgraph.nodes())
    subgraph_gates = torch.argmax(node_gates[subgraph_nodes], dim=1).tolist()
    subgraph_colors = [predefined_colors[g] for g in subgraph_gates]

    plt.figure(figsize=figsize)
    pos = nx.spring_layout(subgraph)
    nx.draw(subgraph, pos, node_color=subgraph_colors, edge_color='gray', node_size=500, font_size=10)

    legend_elements = [Patch(facecolor=color, edgecolor='gray', label=predefined_experts[i]) for i, color in enumerate(predefined_colors)]
    # plt.legend(handles=legend_elements, loc='upper right', prop={'size': lengend_size})

    plt.savefig(f"output/MoE/{args.dataset}/subgraph_vis_{args.no}.png")