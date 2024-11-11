import os
import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from src.data_utils import extract_heterodata_sub


# Model training
def train_node_classifier(model, data, K_new_nodes, K_new_edges, K_old_nodes, K_old_edges, optimizer, criterion, target_type, run, strategy, directory, n_epochs=200):
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        mask_strategy = None #TODO build_mask(strategy, epoch, K_new_nodes, K_new_edges, K_old_nodes, K_old_edges)
        data = extract_heterodata_sub(data, mask_strategy)
        out, _ = model(data.x_dict, data.edge_index_dict)
        mask = data[target_type].train_mask
        loss = criterion(out[target_type][mask], data[target_type].y[mask])
        loss.backward()
        optimizer.step()
        f1_micro, f1_macro, f1_weigh, auc = eval_node_classifier(model, data, target_type, run, directory)
        if epoch % 20 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val f1_micro: {f1_micro:.3f}')
    return model


def eval_node_classifier(model, data, target_type, run, directory):
    model.eval()
    with torch.no_grad():
        out, embeddings = model(data.x_dict, data.edge_index_dict)
        pred = out[target_type].argmax(dim=-1)
        #pred = model(data.x_dict, data.edge_index_dict)['author'].argmax(dim=-1)
        pred_prob = torch.nn.functional.softmax(model(data.x_dict, data.edge_index_dict)[target_type], -1)
        mask = data[target_type].val_mask
        #correct = (pred[mask] == data[target_type].y[mask]).sum()
        #acc = int(correct) / int(mask.sum())
        f1_micro = f1_score(data[target_type].y.cpu(), pred.cpu(), average='micro')
        f1_macro = f1_score(data[target_type].y.cpu(), pred.cpu(), average='macro')
        f1_weigh = f1_score(data[target_type].y.cpu(), pred.cpu(), average='weighted')
        #servono predicted probabilities, no. of classes
        auc = roc_auc_score(data[target_type].y.cpu(), pred_prob.cpu().detach().numpy(), average='macro', multi_class='ovo')

        # Save embeddings for validation set
        val_embeddings = embeddings[target_type].cpu().numpy()
        np.save(os.path.join(directory, 'embeddings_'+str(run)+'.npy'), val_embeddings)

        return f1_micro, f1_macro, f1_weigh, auc