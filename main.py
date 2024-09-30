import argparse
from utils.utils import load_conf, set_seed
from dataset.dataset import Dataset
from exp.solver import Solver
from exp.expManager import ExpManager
import os


if __name__ == "__main__":
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # set_seed(42)
    set_seed(3407)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",type=str, default="ogbn-arxiv", help="Choose from: [cora, citeseer, pubmed, cora-full, computers, photo, cs, physics, ogbn-arxiv]")
    parser.add_argument("--gpu", type=int, default=1, help="Use which gpu")
    parser.add_argument('--n_runs', type=int, default=10)
    # store true or false
    # parser.add_argument('--ablation', action='store_true', help='Whether to make it ablation studies')
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    # print(f"Using GPU: {args.gpu}")

    conf = load_conf(dataset=args.dataset)

    dataset = Dataset(ds_name=args.dataset, n_runs=args.n_runs)

    solver = Solver(conf, dataset)

    exp = ExpManager(solver)
    exp.run(n_runs=args.n_runs)