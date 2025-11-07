import os
import torch

from sklearn.metrics import f1_score, precision_score, recall_score
from torch_geometric.utils.mask import mask_to_index
from tqdm import tqdm
from src.support.utils import compute_auc
from src.support.utils_graph import k_hop_subgraph


def train(model, all_data, new_nodes, old_nodes, optimizer, criterion, scheduler, run, strategy, directory, reduction_factor, n_epochs=200, kwargs=None):
    n_split = max(1, int(reduction_factor * n_epochs))
    data_splits = strategy.sample(n_split, all_data, new_nodes, old_nodes, kwargs)
    progress_bar = tqdm(range(n_epochs))
    for epoch in progress_bar:
        model.train()
        optimizer.zero_grad()
        data = data_splits[int(epoch%n_split)][0]
        train_data = k_hop_subgraph(data, mask_to_index(data[data.target_type].train_mask), 2)[0].to(data.device)
        out = model(train_data.x_dict, train_data.edge_index_dict)
        loss = criterion(out[data.target_type], train_data[data.target_type].y)
        loss.backward()
        optimizer.step()
        if scheduler is not None: scheduler.step()
        f1_micro, f1_macro, auc, precision, recall = evaluate(model, data, run, directory)
        progress_bar.set_description(f"Epoch: {epoch + 1:03d}, Train Loss: {loss:.3f}, Val f1_micro: {f1_micro:.3f}, Val f1_macro: {f1_macro:.3f}, Val AUC: {auc:.3f}, Precision: [{' '.join('{:.5f}'.format(x) for x in precision)}], Recall: [{' '.join('{:.5f}'.format(x) for x in recall)}]")

    return model


def evaluate(model, data, run, directory):
    model.eval()
    with torch.no_grad():
        test_data = k_hop_subgraph(data, mask_to_index(~data[data.target_type].train_mask), 2)[0].to(data.device)
        out = model(test_data.x_dict, test_data.edge_index_dict)
        pred = out[data.target_type].argmax(dim=-1)
        pred_prob = torch.nn.functional.softmax(model(test_data.x_dict, test_data.edge_index_dict)[data.target_type], -1)
        f1_micro = f1_score(test_data[data.target_type].y.cpu(), pred.cpu(), average="micro")
        f1_macro = f1_score(test_data[data.target_type].y.cpu(), pred.cpu(), average="macro")
        auc = compute_auc(test_data[data.target_type].y.cpu().numpy(), pred_prob.cpu().detach().numpy())
        precision = precision_score(test_data[data.target_type].y.cpu(), pred.cpu(), average=None, zero_division=0)
        recall = recall_score(test_data[data.target_type].y.cpu(), pred.cpu(), average=None, zero_division=0)
        """
        # Save embeddings for validation set
        val_embeddings = embeddings[data.target_type].cpu().numpy()
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, f"embeddings_{run}.npy"), val_embeddings)
        """
        return f1_micro, f1_macro, auc, precision, recall

