import os
import torch
import numpy as np

from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
from src.support.utils import compute_auc


def train(model, all_data, new_nodes, new_edges, old_nodes, old_edges, optimizer, criterion, scheduler, target_type, run, strategy, directory, n_epochs=200, kwargs=None):
    data_splits = strategy.sample(n_epochs, all_data, new_nodes, new_edges, old_nodes, old_edges, target_type, kwargs)
    progress_bar = tqdm(range(n_epochs))
    for epoch in progress_bar:
        model.train()
        optimizer.zero_grad()
        data = data_splits[epoch]
        out, _ = model(data.x_dict, data.edge_index_dict)
        mask = data[target_type].train_mask
        loss = criterion(out[target_type][mask], data[target_type].y[mask])
        loss.backward()
        optimizer.step()
        scheduler.step()
        f1_micro, f1_macro, auc, precision, recall = evaluate(model, data, target_type, run, directory)
        progress_bar.set_description(f"Epoch: {epoch + 1:03d}, Train Loss: {loss:.3f}, Val f1_micro: {f1_micro:.3f}, Val f1_macro: {f1_macro:.3f}, Val AUC: {auc:.3f}, Precision: [{' '.join('{:.5f}'.format(x) for x in precision)}], Recall: [{' '.join('{:.5f}'.format(x) for x in recall)}]")

    return model


def evaluate(model, data, target_type, run, directory):
    model.eval()
    with torch.no_grad():
        mask = data[target_type].test_mask
        out, embeddings = model(data.x_dict, data.edge_index_dict)
        pred = out[target_type][mask].argmax(dim=-1)
        pred_prob = torch.nn.functional.softmax(model(data.x_dict, data.edge_index_dict)[0][target_type][mask], -1)
        f1_micro = f1_score(data[target_type].y[mask].cpu(), pred.cpu(), average="micro")
        f1_macro = f1_score(data[target_type].y[mask].cpu(), pred.cpu(), average="macro")
        auc = compute_auc(data[target_type].y[mask].cpu().numpy(), pred_prob.cpu().detach().numpy())
        precision = precision_score(data[target_type].y[mask].cpu(), pred.cpu(), average=None, zero_division=0)
        recall = recall_score(data[target_type].y[mask].cpu(), pred.cpu(), average=None, zero_division=0)
        # Save embeddings for validation set
        val_embeddings = embeddings[target_type].cpu().numpy()
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, f"embeddings_{run}.npy"), val_embeddings)
        return f1_micro, f1_macro, auc, precision, recall

