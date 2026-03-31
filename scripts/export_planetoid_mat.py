#!/usr/bin/env python3

from argparse import ArgumentParser
import inspect
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.io import savemat
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import remove_self_loops, to_scipy_sparse_matrix


DATASET_NAMES = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
}


def patch_torch_load_for_legacy_torch():
    if "weights_only" in inspect.signature(torch.load).parameters:
        return

    original_torch_load = torch.load

    def compat_torch_load(*args, **kwargs):
        kwargs.pop("weights_only", None)
        return original_torch_load(*args, **kwargs)

    torch.load = compat_torch_load


def parse_args():
    parser = ArgumentParser(
        description="Export a Planetoid dataset to the .mat format expected by FairGen."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        choices=sorted(DATASET_NAMES),
        help="Planetoid dataset to export.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("dataset") / "pyg",
        help="Directory used by PyG to cache the raw dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output .mat path. Defaults to data/<dataset>/<dataset>.mat.",
    )
    parser.add_argument(
        "--binary-sensitive-class",
        type=int,
        default=None,
        help="If set, Label becomes 1 for this class and 0 otherwise. By default Label == Class.",
    )
    return parser.parse_args()


def build_adjacency(data):
    edge_index, _ = remove_self_loops(data.edge_index)
    adjacency = to_scipy_sparse_matrix(edge_index, num_nodes=data.num_nodes).tocsr()
    adjacency.data = np.ones_like(adjacency.data, dtype=np.int8)
    adjacency = adjacency.maximum(adjacency.transpose()).tocsr()
    adjacency.eliminate_zeros()
    return adjacency


def build_sensitive_labels(labels, binary_sensitive_class):
    if binary_sensitive_class is None:
        return labels.reshape(-1, 1)
    return (labels == binary_sensitive_class).astype(np.int64).reshape(-1, 1)


def main():
    args = parse_args()
    patch_torch_load_for_legacy_torch()
    dataset_name = DATASET_NAMES[args.dataset]
    output_path = args.output or Path("data") / args.dataset / f"{args.dataset}.mat"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args.root.mkdir(parents=True, exist_ok=True)

    dataset = Planetoid(root=str(args.root), name=dataset_name)
    data = dataset[0]

    adjacency = build_adjacency(data)
    features = sparse.csr_matrix(data.x.cpu().numpy())
    labels = data.y.cpu().numpy().astype(np.int64)
    sensitive = build_sensitive_labels(labels, args.binary_sensitive_class)

    payload = {
        "Network": adjacency,
        "Attributes": features,
        "Class": labels.reshape(-1, 1),
        "Label": sensitive,
        "TrainMask": data.train_mask.cpu().numpy().astype(np.int8).reshape(-1, 1),
        "ValMask": data.val_mask.cpu().numpy().astype(np.int8).reshape(-1, 1),
        "TestMask": data.test_mask.cpu().numpy().astype(np.int8).reshape(-1, 1),
    }
    savemat(output_path, payload, do_compression=True)

    print(f"Saved {output_path}")
    print(f"Nodes: {data.num_nodes}")
    print(f"Edges: {adjacency.nnz // 2}")
    print(f"Features: {features.shape[1]}")
    print(f"Classes: {labels.min()}..{labels.max()}")
    if args.binary_sensitive_class is None:
        print("Label mode: same_as_class")
    else:
        print(f"Label mode: binary_class_{args.binary_sensitive_class}")


if __name__ == "__main__":
    main()
