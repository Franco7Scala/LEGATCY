import os

from src.utils import build_new_heterodata

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import Linear, to_hetero
from torch_geometric.nn.conv import GATv2Conv
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.explain import Explainer, CaptumExplainer

from utils import processing_results, compute_weights, build_new_heterodata

fname_base = 'DyHANE/'
base = '2019-2021/' #'2019-2022/'
new = '2022/'

#Define target node type
target_type = 'author'

#Build new tran-val-test set as I U B.
#I per me è la nuova conoscenza (da "Influenced node set"). Sono i nodi in qualche modo affected dal cambiamento (i nuovi + quelli cui si attaccano + eventualm. qualcos'altro)
#B è la vecchia conoscenza fissata (da "memory Buffer"). Sono i nodi che al timestamp precedente sono ritenuti rilevanti e memorizzati.
#Al tempo t=0, I=V e B è vuoto
I = pickle.load(open(os.path.join(fname_base,new,'I.pkl'), 'rb'))
B = pickle.load(open(os.path.join(fname_base,base,'B.pkl'), 'rb'))
data = build_new_heterodata(I,B) #data ha già train_mask, val_mask, test_mask


#Questa GAT ha due layer. è pensata solo per la classificazione, perché out_channels = num_classes
class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, dropout=0):
        super().__init__()
        self.conv1 = GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, dropout=dropout)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATv2Conv((-1, -1), out_channels, add_self_loops=False, dropout=dropout)
        self.lin2 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x.relu()) #self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x.relu()) #self.lin2(x)
        return x


#GAT pensata sempre per la classificazione, ma salva anche gli embeddings prima dell'ultimo layer
class GAT_enhanced(torch.nn.Module):
    def __init__(self, hidden_channels=128, out_channels=2, dropout=0, num_layers=2):
        super().__init__()
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        # First layer
        self.convs.append(GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, dropout=dropout))
        self.lins.append(Linear(-1, hidden_channels))

        # Intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, dropout=dropout))
            self.lins.append(Linear(-1, hidden_channels))

        # Last layer
        self.convs.append(GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, dropout=dropout))
        self.lins.append(Linear(-1, hidden_channels))

        self.final_conv = GATv2Conv((-1, -1), out_channels, add_self_loops=False, dropout=dropout)  #final GAT layer for classification
        self.final_lin = Linear(-1, out_channels)  # final linear layer for classification

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index) + self.lins[i](x.relu())
            x = x.relu()

        # Save embeddings before the final layer
        self.embeddings = self.convs[-1](x, edge_index) + self.lins[-1](x.relu())

        # Last layer (no ReLU after the last convolution)
        x = self.final_conv(self.embeddings, edge_index) + self.final_lin(self.embeddings.relu())
        return x, self.embeddings


# Model training

def train_node_classifier(model, data, optimizer, criterion, n_epochs=200, target_type='author'):
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        mask = data[target_type].train_mask
        loss = criterion(out[target_type][mask], data[target_type].y[mask])
        loss.backward()
        optimizer.step()
        pred = out[target_type].argmax(dim=1) ## Use the class with highest probability.
        f1_micro, f1_macro, f1_weigh, auc = eval_node_classifier(model, data)
        if epoch % 20 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val f1_micro: {f1_micro:.3f}')
    return model


def eval_node_classifier(model, data):
    model.eval()
    with torch.no_grad():
        pred = model(data.x_dict, data.edge_index_dict)['author'].argmax(dim=-1)
        pred_prob = torch.nn.functional.softmax(model(data.x_dict, data.edge_index_dict)['author'], -1)
        mask = data['author'].val_mask
        correct = (pred[mask] == data['author'].y[mask]).sum()
        #acc = int(correct) / int(mask.sum())
        f1_micro = f1_score(data['author'].y.cpu(), pred.cpu(), average='micro')
        f1_macro = f1_score(data['author'].y.cpu(), pred.cpu(), average='macro')
        f1_weigh = f1_score(data['author'].y.cpu(), pred.cpu(), average='weighted')
        #servono predicted probabilities, 8 classes
        auc = roc_auc_score(data['author'].y.cpu(), pred_prob.cpu().detach().numpy(), average='macro', multi_class='ovo')
        return f1_micro, f1_macro, f1_weigh, auc


#Alternative to training: load saved model
def load_model(hidden_channels,out_channels,dropout,data,weigths_filename):
    model = GAT(hidden_channels, out_channels,dropout)
    model = to_hetero(model, data.metadata(), aggr='sum')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, model = data.to(device), model.to(device)
    with torch.no_grad():
        model.eval()
        model(data.x_dict, data.edge_index_dict)
        model.train()
    model.load_state_dict(torch.load(weigths_filename))
    return model



num_run = 5
l_micro = []
l_macro = []
l_weigh = []
l_auc = []
for run in range(num_run):
    print('##### RUN '+str((run+1))+' #####')

    model = None
    model = GAT(hidden_channels=64, out_channels=8, dropout=0.4) #num_classes
    model = to_hetero(model, data.metadata(), aggr='sum')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, model = data.to(device), model.to(device)

    #TRAIN THE MODEL
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
    weights = compute_weights(data[target_type].y).float().to(device) #torch.tensor([1.5, 1.5, 1, 1, 1, 1, 1.5, 1]).float().to(device)
    criterion = nn.CrossEntropyLoss(weights)
    model = train_node_classifier(model, data, optimizer, criterion, n_epochs=500)
    torch.save(model.state_dict(), fname_base+base+'base_model.pth')

    ### alternative to train: LOAD SAVED MODEL
    #model = load_model(64,8,0.4,data, os.path.join(fname_base,base, 'model.pth'))


    f1_micro, f1_macro, f1_weigh, auc = eval_node_classifier(model, data)
    #print(f'Test Eval: {test_ev:.3f}')
    print(f'f1-micro: {f1_micro:.3f}, f1-macro: {f1_macro:.3f}, f1-weighted: {f1_weigh:.3f}, roc-auc: {auc:.3f}')

    l_micro.append(f1_micro)
    l_macro.append(f1_macro)
    l_weigh.append(f1_weigh)
    l_auc.append(auc)

df = pd.DataFrame(columns=['F1_micro', 'F1_macro', 'F1_weighted', 'ROC-AUC'])
df['F1_micro'] = l_micro
df['F1_macro'] = l_macro
df['F1_weighted'] = l_weigh
df['ROC-AUC'] = l_auc
df_ok = processing_results(df)
df_ok.to_excel(os.path.join(fname_base,base,'results.xlsx'), index=False)


################# SONO ARRIVATA QUI ################


#Da qui in poi, vecchia strategia di Liliana per aggiornare il buffer

#Explainer for new memory buffer B
explainer = Explainer(
    model=model,
    algorithm=CaptumExplainer('IntegratedGradients'),  # InputXGradient
    explanation_type='phenomenon',  # model's beahviour (model) vs individual predictions (phenomenon)
    node_mask_type='attributes',
    # attributes #'CaptumExplainer' only supports 'node_mask_type' None or 'attributes' (got 'object')
    edge_mask_type=None,
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='probs',  # log_probs, raw
    ),
)
with torch.no_grad():
    explanation = explainer(data.x_dict, data.edge_index_dict, target=data.y_dict["author"], index=torch.arange(13116))  #target=data.y if explanation_type=='phenomenon' else None
# print(f'Generated explanations in {explanation.available_explanations}')
# path = fname_base+'feature_importance.png'
# explanation.visualize_feature_importance(path, top_k=10)
# print(f"Feature importance plot has been saved to '{path}'")
B = {}
B_size = 768
topk = 64
n_types = ['author', 'paper', 'institution']
percs = [0] * len(n_types)
#type importance, dato dalla media dei topk per ogni tipo
for i, nt in enumerate(n_types):
    values,indices=torch.topk(explanation.node_mask_dict[nt].sum(-1),k=topk)
    percs[i] = values.mean().item()
scaling_factor = B_size / sum(percs)
rates = [round(val * scaling_factor) for val in percs]
# top nodes for each type
for i, nt in enumerate(n_types):
    values,indices=torch.topk(explanation.node_mask_dict[nt].sum(-1),k=rates[i])
    B[nt+'s'] = indices.cpu().numpy().tolist()
with open(fname_base+base+'B_'+str(B_size)+'.pkl', 'wb') as fp:
    pickle.dump(B, fp)
    print('dictionary saved successfully to file')
#print(B)
