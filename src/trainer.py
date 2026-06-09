import os
import torch

from sklearn.metrics import f1_score, precision_score, recall_score
from torch_geometric.utils.mask import mask_to_index
from tqdm import tqdm
from src.support.utils import compute_auc, merge_masks
from src.support.utils_graph import k_hop_subgraph


def train(model, all_data, new_nodes, old_nodes, optimizer, criterion, scheduler, run, strategy, directory, reduction_factor, n_epochs=200, kwargs=None):
    n_split = max(1, int(reduction_factor * n_epochs))
    data_splits = strategy.sample(n_split, all_data, new_nodes["train"], old_nodes["train"], kwargs)
    progress_bar = tqdm(range(n_epochs))
    for epoch in progress_bar:
        model.train()
        optimizer.zero_grad()
        train_data = data_splits[int(epoch%n_split)][0]
        out = model(train_data.x_dict, train_data.edge_index_dict)
        loss = criterion(out[train_data.target_type], train_data[train_data.target_type].y)
        loss.backward()
        optimizer.step()
        if scheduler is not None: scheduler.step()
        progress_bar.set_description(f"Epoch: {epoch + 1:03d}, Train Loss: {loss:.3f}")

    return model


def evaluate(model, all_data, new_nodes, old_nodes, run, directory):
    test_data, _, test_mask = k_hop_subgraph(all_data, merge_masks([new_nodes["test"], old_nodes["test"]])[all_data.target_type], 2, False)
    model.eval()
    with torch.no_grad():
        out_dict = model(test_data.x_dict, test_data.edge_index_dict)
        out_target_test = out_dict[test_data.target_type][test_mask]
        y_target_test = test_data[test_data.target_type].y[test_mask]
        pred = out_target_test.argmax(dim=-1)
        pred_prob = torch.nn.functional.softmax(out_target_test, dim=-1)
        y_true_cpu = y_target_test.cpu().numpy()
        pred_cpu = pred.cpu().numpy()
        pred_prob_cpu = pred_prob.cpu().detach().numpy()
        f1_micro = f1_score(y_true_cpu, pred_cpu, average="micro")
        f1_macro = f1_score(y_true_cpu, pred_cpu, average="macro")
        auc = compute_auc(y_true_cpu, pred_prob_cpu)
        precision = precision_score(y_true_cpu, pred_cpu, average=None, zero_division=0)
        recall = recall_score(y_true_cpu, pred_cpu, average=None, zero_division=0)
        return f1_micro, f1_macro, auc, precision, recall
