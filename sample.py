import dgl
import torch
import torch.nn.functional as F
import os
from data import load_dataset, preprocess, load_datasets_nc
from eval_utils import Evaluator
from setup_utils import set_seed
import networkx as nx

def main(args):
    dataset = args.dataset

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')

    if dataset in ['cora', 'citeseer', 'amazon_photo', 'amazon_computer']:
        g_real = load_dataset(dataset)
    else:
        g_real = load_datasets_nc(dataset)
    X_one_hot_3d_real, s_real, y_real, E_one_hot_real,\
        X_marginal, s_marginal, y_marginal, E_marginal, X_cond_s_marginals, X_cond_y_marginals, y_cond_s_marginals, p_values = preprocess(g_real)
    s_one_hot_real = F.one_hot(s_real)
    if y_real is not None:
        Y_one_hot_3d_real = F.one_hot(y_real)
    else:
        Y_one_hot_3d_real = None
    evaluator = Evaluator(dataset,
                          os.getcwd(),
                          g_real,
                          X_one_hot_3d_real,
                          s_one_hot_real,
                          Y_one_hot_3d_real)
    if y_real is not None:
        y_marginal = y_marginal.to(device)
        y_cond_s_marginals = y_cond_s_marginals.to(device)
    X_marginal = X_marginal.to(device)
    s_marginal = s_marginal.to(device)   
    E_marginal = E_marginal.to(device)
    X_cond_s_marginals = X_cond_s_marginals.to(device)
    num_nodes = s_real.size(0)


    # Set seed for better reproducibility.
    set_seed()
    evaluator.summary()

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to the model.")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        choices=["cora", "citeseer", "amazon_photo", "amazon_computer", "german", "pokec_n"])
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to generate.")
    parser.add_argument("--gpu", type=int, default=0, required=False,  choices=[0, 1, 2, 3])
    args = parser.parse_args()

    main(args)
