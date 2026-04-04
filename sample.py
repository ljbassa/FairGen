import argparse
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from scipy.io import loadmat

try:
    from torch_geometric.data import Data
except ImportError:
    Data = None


PLANETOID_STYLE_DATASETS = {"cora", "citeseer", "amazon_photo", "amazon_computer"}


def default_mat_path(dataset_name):
    return Path("data") / dataset_name / f"{dataset_name}.mat"


def load_real_node_payload(dataset_name, mat_path):
    mat_payload = loadmat(mat_path)

    if "Attributes" not in mat_payload:
        raise KeyError(f"'Attributes' is missing in {mat_path}")

    x = mat_payload["Attributes"]
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float32)
    x[x != 0] = 1.0

    non_full_zero_feat_mask = x.sum(axis=0) != 0
    x = x[:, non_full_zero_feat_mask]

    non_full_one_feat_mask = x.sum(axis=0) != x.shape[0]
    x = x[:, non_full_one_feat_mask]

    class_key = "Class" if "Class" in mat_payload else "Label"
    y = np.asarray(mat_payload[class_key]).reshape(-1).astype(np.int64)

    if dataset_name in PLANETOID_STYLE_DATASETS:
        sens = y.copy()
    else:
        sens_key = "Label" if "Label" in mat_payload else class_key
        sens = np.asarray(mat_payload[sens_key]).reshape(-1).astype(np.int64)

    return x, y, sens


def load_node_id_mapping(mapping_path):
    with open(mapping_path, "rb") as f:
        mapping_payload = pickle.load(f)

    node_value = mapping_payload.get("value")
    if node_value is None:
        raise KeyError(f"'value' mapping is missing in {mapping_path}")

    return {int(node_id): int(orig_id) for node_id, orig_id in node_value.items()}


def read_generated_edges(graph_path):
    edge_graph = nx.read_edgelist(
        graph_path,
        nodetype=int,
        data=(("weight", float),),
        create_using=nx.Graph(),
    )
    return sorted(
        (
            (int(min(u, v)), int(max(u, v)))
            for u, v in edge_graph.edges()
            if int(u) != int(v)
        ),
        key=lambda edge: (edge[0], edge[1]),
    )


def build_generated_graph(dataset_name, graph_path, mapping_path, mat_path):
    x_real, y_real, sens_real = load_real_node_payload(dataset_name, mat_path)
    node_id_mapping = load_node_id_mapping(mapping_path)
    generated_edges = read_generated_edges(graph_path)

    num_nodes = len(node_id_mapping)
    graph = nx.Graph()

    x_local = np.zeros((num_nodes, x_real.shape[1]), dtype=np.float32)
    y_local = np.zeros(num_nodes, dtype=np.int64)
    sens_local = np.zeros(num_nodes, dtype=np.int64)
    orig_ids = np.zeros(num_nodes, dtype=np.int64)

    for node_id in range(num_nodes):
        orig_id = node_id_mapping.get(node_id, node_id)
        if orig_id < 0 or orig_id >= x_real.shape[0]:
            raise IndexError(
                f"orig_id {orig_id} for generated node {node_id} is out of bounds "
                f"for dataset {dataset_name}"
            )

        x_local[node_id] = x_real[orig_id]
        y_local[node_id] = y_real[orig_id]
        sens_local[node_id] = sens_real[orig_id]
        orig_ids[node_id] = orig_id

        graph.add_node(
            node_id,
            orig_id=int(orig_id),
            x=x_local[node_id],
            y=int(y_local[node_id]),
            sens=int(sens_local[node_id]),
        )

    graph.add_edges_from(generated_edges)
    graph.graph["dataset"] = dataset_name
    graph.graph["num_features"] = int(x_local.shape[1])
    graph.graph["source_graph_path"] = str(Path(graph_path).resolve())

    return graph, x_local, y_local, sens_local, orig_ids


def build_pyg_data(x_local, y_local, sens_local, orig_ids, generated_edges):
    if Data is None:
        raise ImportError(
            "torch_geometric is required for --save_pt_path but is not installed "
            "in the active environment."
        )

    if generated_edges:
        edge_index = torch.tensor(generated_edges, dtype=torch.long).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(
        x=torch.from_numpy(x_local).float(),
        edge_index=edge_index,
        orig_id=torch.from_numpy(orig_ids).long(),
        y=torch.from_numpy(y_local).long(),
        sens=torch.from_numpy(sens_local).long(),
    )


def export_generated_graph(dataset_name, graph_path, mapping_path, save_pkl_dir=None, save_pt_path=None,
                           sample_idx=0, mat_path=None):
    if mat_path is None:
        mat_path = default_mat_path(dataset_name)
    graph, x_local, y_local, sens_local, orig_ids = build_generated_graph(
        dataset_name=dataset_name,
        graph_path=graph_path,
        mapping_path=mapping_path,
        mat_path=mat_path,
    )
    generated_edges = sorted((int(u), int(v)) for u, v in graph.edges())

    if save_pkl_dir is not None:
        save_dir = Path(save_pkl_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"sample_{sample_idx:03d}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Saved pkl] {save_path}")

    if save_pt_path is not None:
        save_path = Path(save_pt_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        pyg_graph = build_pyg_data(
            x_local=x_local,
            y_local=y_local,
            sens_local=sens_local,
            orig_ids=orig_ids,
            generated_edges=generated_edges,
        )
        torch.save([pyg_graph], save_path)
        print(f"[Saved pt] {save_path} (1 graph)")

    return graph


def default_graph_path(dataset_name):
    return Path("data") / dataset_name / f"{dataset_name}_output_edgelist_0_2.txt"


def default_mapping_path(dataset_name):
    return Path("data") / dataset_name / "graph.pickle"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert FairGen generated edge lists to the FairWire graph format."
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        choices=["cora", "citeseer", "amazon_photo", "amazon_computer", "german", "pokec_n"],
    )
    parser.add_argument(
        "--graph_path",
        type=str,
        default=None,
        help="Path to the generated FairGen edge list. Defaults to data/<dataset>/<dataset>_output_edgelist_0_2.txt",
    )
    parser.add_argument(
        "--mapping_path",
        type=str,
        default=None,
        help="Path to FairGen graph.pickle. Defaults to data/<dataset>/graph.pickle",
    )
    parser.add_argument(
        "--mat_path",
        type=str,
        default=None,
        help="Path to the FairGen dataset .mat file. Defaults to data/<dataset>/<dataset>.mat",
    )
    parser.add_argument(
        "--save_pkl_dir",
        type=str,
        default="generated_pkls",
        help="Directory where sample_000.pkl-style files will be stored.",
    )
    parser.add_argument(
        "--save_pt_path",
        type=str,
        default=None,
        help="Optional path for a FairWire-compatible list[PyG Data] .pt file.",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Index used in the exported sample_<idx>.pkl filename.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    graph_path = Path(args.graph_path) if args.graph_path is not None else default_graph_path(args.dataset)
    mapping_path = Path(args.mapping_path) if args.mapping_path is not None else default_mapping_path(args.dataset)
    mat_path = Path(args.mat_path) if args.mat_path is not None else default_mat_path(args.dataset)

    export_generated_graph(
        dataset_name=args.dataset,
        graph_path=graph_path,
        mapping_path=mapping_path,
        save_pkl_dir=args.save_pkl_dir,
        save_pt_path=args.save_pt_path,
        sample_idx=args.sample_idx,
        mat_path=mat_path,
    )


if __name__ == "__main__":
    main()
