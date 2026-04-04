#!/usr/bin/env python3

import argparse
import importlib.util
import inspect
import pickle
from pathlib import Path
from typing import List, Optional

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.io import loadmat
import torch

from sample import default_mapping_path, default_mat_path, export_generated_graph, load_real_node_payload


def load_fairwire_eval_module():
    fairwire_root = Path(__file__).resolve().parent.parent / "FairWire"
    module_path = fairwire_root / "evaluate_generated_graphs.py"
    if not module_path.exists():
        raise FileNotFoundError(f"FairWire evaluator not found: {module_path}")

    spec = importlib.util.spec_from_file_location("fairwire_evaluate_generated_graphs", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import FairWire evaluator from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_torch_load(path: Path):
    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    return torch.load(path, **load_kwargs)


def load_saved_graphs_safe(graph_path: Path) -> List[object]:
    obj = safe_torch_load(graph_path)
    graphs = obj if isinstance(obj, list) else [obj]
    if not graphs:
        raise ValueError(f"No graphs found in {graph_path}")
    return graphs


def build_reference_graph(dataset_name: str, mat_path: Path) -> nx.Graph:
    x, y, sens = load_real_node_payload(dataset_name, mat_path)

    mat_payload = loadmat(mat_path)
    if "Network" not in mat_payload:
        raise KeyError(f"'Network' is missing in {mat_path}")

    adjacency = mat_payload["Network"]
    if sparse.issparse(adjacency):
        adjacency = adjacency.tocsr()
    else:
        adjacency = sparse.csr_matrix(adjacency)

    adjacency = adjacency.maximum(adjacency.transpose()).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    graph = nx.from_scipy_sparse_array(adjacency, create_using=nx.Graph())
    for node_id in graph.nodes():
        graph.nodes[node_id]["orig_id"] = int(node_id)
        graph.nodes[node_id]["x"] = x[node_id]
        graph.nodes[node_id]["y"] = int(y[node_id])
        graph.nodes[node_id]["sens"] = int(sens[node_id])

    graph.graph["dataset"] = dataset_name
    graph.graph["num_features"] = int(x.shape[1])
    graph.graph["source_mat_path"] = str(mat_path.resolve())
    return graph


def default_reference_graph_path(dataset_name: str) -> Path:
    return Path(__file__).resolve().parent / "graphs" / f"{dataset_name}_feat.pkl"


def ensure_reference_graph_path(dataset_name: str, mat_path: Path, explicit_path: Optional[str]) -> Path:
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"reference graph not found: {path}")
        return path

    local_path = default_reference_graph_path(dataset_name)
    if local_path.exists():
        return local_path

    fairwire_path = Path(__file__).resolve().parent.parent / "FairWire" / "graphs" / f"{dataset_name}_feat.pkl"
    if fairwire_path.exists():
        return fairwire_path

    local_path.parent.mkdir(parents=True, exist_ok=True)
    graph = build_reference_graph(dataset_name=dataset_name, mat_path=mat_path)
    with open(local_path, "wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[Saved reference graph] {local_path}")
    return local_path


def ensure_graph_pt_path(args) -> Path:
    graph_path = Path(args.graph_path)
    if graph_path.suffix == ".pt":
        if not graph_path.exists():
            raise FileNotFoundError(f"graph pt not found: {graph_path}")
        return graph_path

    if graph_path.suffix != ".txt":
        raise ValueError("graph_path must be a FairGen edge list (*.txt) or a saved PyG graph file (*.pt)")

    converted_pt_path = Path(args.converted_pt_path) if args.converted_pt_path else graph_path.with_suffix(".pyg.pt")
    if converted_pt_path.exists() and not args.force_rebuild_graph_pt:
        return converted_pt_path

    mapping_path = Path(args.mapping_path) if args.mapping_path else default_mapping_path(args.dataset)
    mat_path = Path(args.mat_path) if args.mat_path else default_mat_path(args.dataset)

    export_generated_graph(
        dataset_name=args.dataset,
        graph_path=graph_path,
        mapping_path=mapping_path,
        save_pkl_dir=None,
        save_pt_path=converted_pt_path,
        sample_idx=0,
        mat_path=mat_path,
    )
    return converted_pt_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate FairGen-generated graphs with the same LP/fairness metrics used in FairWire."
    )
    parser.add_argument("--graph_path", type=str, required=True,
                        help="Path to FairGen edge list (*.txt) or exported PyG graph file (*.pt).")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["cora", "citeseer", "amazon_photo", "amazon_computer", "german", "pokec_n"])
    parser.add_argument("--graph_index", type=int, default=None,
                        help="Optional: evaluate only one graph from the saved list.")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--mapping_path", type=str, default=None,
                        help="Used only when graph_path is *.txt. Defaults to data/<dataset>/graph.pickle.")
    parser.add_argument("--mat_path", type=str, default=None,
                        help="Used for reference graph creation and *.txt conversion. Defaults to data/<dataset>/<dataset>.mat.")
    parser.add_argument("--converted_pt_path", type=str, default=None,
                        help="Used only when graph_path is *.txt. Defaults to <graph_path>.pyg.pt.")
    parser.add_argument("--force_rebuild_graph_pt", action="store_true",
                        help="Rebuild the converted *.pyg.pt even if it already exists.")
    parser.add_argument("--reference_graph_path", type=str, default=None,
                        help="Optional explicit path to reference graphs/<dataset>_feat.pkl.")

    parser.add_argument("--sensitive_attr", type=str, default="y")
    parser.add_argument("--sensitive_value", type=int, default=3)
    parser.add_argument("--edge_sensitive_mode", type=str, default="either", choices=["either", "both"])

    parser.add_argument("--max_pos_edges", type=int, default=20000)
    parser.add_argument("--neg_ratio", type=float, default=1.0)

    parser.add_argument("--lp_model", type=str, default="gcn", choices=["gcn", "sage", "gat"])
    parser.add_argument("--lp_num_layers", type=int, default=2)
    parser.add_argument("--lp_hidden_dim", type=int, default=128)
    parser.add_argument("--lp_out_dim", type=int, default=64)
    parser.add_argument("--lp_dropout", type=float, default=0.1)
    parser.add_argument("--lp_lr", type=float, default=1e-2)
    parser.add_argument("--lp_weight_decay", type=float, default=0.0)
    parser.add_argument("--lp_epochs", type=int, default=300)
    parser.add_argument("--lp_patience", type=int, default=30)
    parser.add_argument("--lp_batch_size", type=int, default=16384)
    parser.add_argument("--lp_test_ratio", type=float, default=0.2)
    parser.add_argument("--lp_val_ratio", type=float, default=0.2)
    parser.add_argument("--gat_heads", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--lp_search", action="store_true")
    parser.add_argument("--lp_search_hidden_dims", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--lp_search_lrs", type=float, nargs="+", default=[1e-2, 3e-3])
    parser.add_argument("--lp_search_dropouts", type=float, nargs="+", default=[0.0, 0.1, 0.2])
    parser.add_argument("--lp_search_num_layers", type=int, nargs="+", default=[1, 2])

    parser.add_argument("--out_per_graph_csv", type=str, default=None)
    parser.add_argument("--out_summary_csv", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    fairwire_eval = load_fairwire_eval_module()
    fairwire_eval.set_seed(args.seed)

    mat_path = Path(args.mat_path) if args.mat_path else default_mat_path(args.dataset)
    graph_pt_path = ensure_graph_pt_path(args)
    reference_graph_path = ensure_reference_graph_path(
        dataset_name=args.dataset,
        mat_path=mat_path,
        explicit_path=args.reference_graph_path,
    )

    stem = graph_pt_path.name[:-3] if graph_pt_path.name.endswith(".pt") else graph_pt_path.name
    out_dir = graph_pt_path.parent
    if args.out_per_graph_csv is None:
        args.out_per_graph_csv = str(out_dir / f"{stem}.overlap_lp_gae_per_graph.csv")
    if args.out_summary_csv is None:
        args.out_summary_csv = str(out_dir / f"{stem}.overlap_lp_gae_summary.csv")

    graphs = load_saved_graphs_safe(graph_pt_path)
    total_loaded = len(graphs)
    if args.graph_index is not None:
        if args.graph_index < 0 or args.graph_index >= total_loaded:
            raise IndexError(f"graph_index={args.graph_index} out of range for {total_loaded} graphs")
        graphs = [graphs[args.graph_index]]

    with open(reference_graph_path, "rb") as f:
        g_ref = pickle.load(f)

    reference_pairs, reference_labels = fairwire_eval.build_fixed_eval_pairs(
        g_ref,
        max_pos_edges=args.max_pos_edges,
        neg_ratio=args.neg_ratio,
        seed=args.seed,
    )
    reference_node_sensitive = fairwire_eval.build_reference_node_sensitive_map(
        g_ref,
        sensitive_attr=args.sensitive_attr,
        sensitive_value=args.sensitive_value,
    )

    per_graph_rows = []
    overlap_raw_rows = []
    lp_raw_rows = []

    for i, data in enumerate(graphs):
        original_idx = args.graph_index if args.graph_index is not None else i
        row = {
            "graph_idx": float(original_idx),
            "lp/model": args.lp_model,
            "lp_protocol": "gae_train_test",
        }

        overlap_metrics, overlap_raw = fairwire_eval.edge_overlap_on_fixed_pairs(
            data=data,
            reference_pairs=reference_pairs,
            reference_labels=reference_labels,
            reference_node_sensitive=reference_node_sensitive,
            edge_sensitive_mode=args.edge_sensitive_mode,
        )
        row.update(overlap_metrics)
        overlap_raw_rows.append(overlap_raw)

        try:
            split = fairwire_eval.build_generated_graph_train_test_split(
                data=data,
                test_ratio=args.lp_test_ratio,
                seed=args.seed + int(original_idx),
            )
            model, meta = fairwire_eval.train_lp_for_generated_graph(
                data=data,
                split=split,
                args=args,
                seed=args.seed + 100 * int(original_idx),
            )
            row["lp/best_val_auc"] = float(meta.get("best_val_auc", float("nan")))
            row["lp/best_train_loss"] = float(meta.get("best_train_loss", float("nan")))
            row["lp/best_epoch"] = float(meta.get("best_epoch", float("nan")))
            row["lp/best_num_layers"] = float(meta.get("num_layers", float("nan")))
            row["lp/best_hidden_dim"] = float(meta.get("hidden_dim", float("nan")))
            row["lp/best_dropout"] = float(meta.get("dropout", float("nan")))
            row["lp/best_lr"] = float(meta.get("lr", float("nan")))
            row["lp/train_num_pos"] = float(split["train_pos"].size(1))
            row["lp/test_num_pos"] = float(split["test_pos"].size(1))
            row["lp/test_num_neg"] = float(split["test_neg"].size(1))

            lp_metrics, lp_raw = fairwire_eval.evaluate_lp_on_generated_test_pairs(
                data=data,
                model=model,
                train_mp_edge_index=split["train_mp_edge_index"],
                test_pos=split["test_pos"],
                test_neg=split["test_neg"],
                sensitive_attr=args.sensitive_attr,
                sensitive_value=args.sensitive_value,
                edge_sensitive_mode=args.edge_sensitive_mode,
                threshold=args.threshold,
                device=args.device,
            )
            row.update(lp_metrics)
            lp_raw_rows.append(lp_raw)
        except Exception as exc:
            row["lp/error"] = str(exc)
            for key in [
                "lp/auc",
                "lp/score_sp_gap",
                "lp/score_sp_abs_gap",
                "lp/sp_gap",
                "lp/sp_abs_gap",
                "lp/eo_gap",
                "lp/eo_abs_gap",
                "lp/score_mean_sensitive",
                "lp/score_mean_nonsensitive",
                "lp/hard_rate_sensitive",
                "lp/hard_rate_nonsensitive",
                "lp/num_eval_pairs",
                "lp/best_val_auc",
                "lp/best_train_loss",
                "lp/best_epoch",
                "lp/best_num_layers",
                "lp/best_hidden_dim",
                "lp/best_dropout",
                "lp/best_lr",
                "lp/train_num_pos",
                "lp/test_num_pos",
                "lp/test_num_neg",
            ]:
                row[key] = float("nan")

        fairwire_eval.add_compat_metric_aliases(row)
        per_graph_rows.append(row)
        fairwire_eval.write_csv(per_graph_rows, Path(args.out_per_graph_csv))

    summary = {
        "num_loaded_graphs": float(total_loaded),
        "num_evaluated_graphs": float(len(per_graph_rows)),
        "reference_num_pairs": float(len(reference_pairs)),
        "reference_pos_pairs": float(reference_labels.sum()),
        "reference_neg_pairs": float((reference_labels == 0).sum()),
        "lp/model": args.lp_model,
        "lp_protocol": "gae_train_test",
        "lp_search": float(bool(args.lp_search)),
        "lp_test_ratio": float(args.lp_test_ratio),
        "lp_val_ratio": float(args.lp_val_ratio),
    }

    numeric_keys = []
    seen = set()
    for row in per_graph_rows:
        for key, value in row.items():
            if isinstance(value, (int, float, np.floating)) and key not in seen:
                seen.add(key)
                numeric_keys.append(key)

    for key in numeric_keys:
        values = [float(row[key]) for row in per_graph_rows if key in row]
        summary[f"{key}_mean"] = fairwire_eval.safe_mean(values)
        summary[f"{key}_std"] = fairwire_eval.safe_std(values)

    overlap_mean_scores, overlap_valid = fairwire_eval.ensemble_mean_scores(overlap_raw_rows, len(reference_pairs))
    if overlap_valid.sum() > 0:
        summary["ensemble_overlap/auc"] = fairwire_eval.safe_auc(
            reference_labels[overlap_valid],
            overlap_mean_scores[overlap_valid],
        )
        summary["ensemble_overlap/coverage_pairs"] = float(overlap_valid.sum())

    if lp_raw_rows:
        all_lp_labels = np.concatenate([row["labels"] for row in lp_raw_rows], axis=0)
        all_lp_scores = np.concatenate([row["scores"] for row in lp_raw_rows], axis=0)
        all_lp_sens = np.concatenate([row["sens_mask"] for row in lp_raw_rows], axis=0)
        aggregate_lp = fairwire_eval.compute_binary_and_score_fairness(
            probs=all_lp_scores,
            labels=all_lp_labels,
            sens_mask=all_lp_sens,
            threshold=args.threshold,
        )
        for key, value in aggregate_lp.items():
            summary[f"aggregate_lp/{key}"] = value
        summary["aggregate_lp/num_graphs"] = float(len(lp_raw_rows))

    fairwire_eval.add_compat_metric_aliases(summary)
    if "aggregate_lp/score_sp_gap" in summary and "aggregate_fair_gap" not in summary:
        summary["aggregate_fair_gap"] = float(summary["aggregate_lp/score_sp_gap"])
    if "aggregate_lp/score_sp_abs_gap" in summary and "aggregate_fair_abs_gap" not in summary:
        summary["aggregate_fair_abs_gap"] = float(summary["aggregate_lp/score_sp_abs_gap"])
    if "ensemble_overlap/auc" in summary and "ensemble_value/linkpred_auc" not in summary:
        summary["ensemble_value/linkpred_auc"] = float(summary["ensemble_overlap/auc"])

    fairwire_eval.write_csv([summary], Path(args.out_summary_csv))
    print(f"[Saved per-graph csv] {args.out_per_graph_csv}")
    print(f"[Saved summary csv] {args.out_summary_csv}")
    print(f"[Reference graph] {reference_graph_path}")
    print(f"[Evaluated graph pt] {graph_pt_path}")


if __name__ == "__main__":
    main()
