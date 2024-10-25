import os
import pickle
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import Linear, to_hetero
from torch_geometric.explain import Explainer, CaptumExplainer

from src import utils, trainer
from src.utils import build_masks
from utils import set_random_seed, processing_results, compute_weights, build_new_heterodata
from trainer import train_node_classifier, eval_node_classifier


path_technique = "DyHANE/"
base = "2019-2021/" #"2019-2022/"
new = "2022/"
target_type = "author"
num_classes = 8

#Build new tran-val-test set as I U B.
#I per me è la nuova conoscenza (da "Influenced node set"). Sono i nodi in qualche modo affected dal cambiamento (i nuovi + quelli cui si attaccano + eventualm. qualcos"altro)
#B è la vecchia conoscenza fissata (da "memory Buffer"). Sono i nodi che al timestamp precedente sono ritenuti rilevanti e memorizzati.
#Al tempo t=0, I=V e B è vuoto
new_knowledge = pickle.load(open(os.path.join(path_technique, new, "I.pkl"), "rb"))
previous_knowledge = pickle.load(open(os.path.join(path_technique, base, "B.pkl"), "rb"))
data = build_new_heterodata(new_knowledge, previous_knowledge) #data ha già train_mask, val_mask, test_mask


l_micro = []
l_macro = []
l_weigh = []
l_auc = []
for run in range(len(utils.training_seeds)):
    print(f"Performing run n {run} on {len(utils.training_seeds)}...")
    set_random_seed(utils.training_seeds[run])

    model = None
    model = GAT(hidden_channels=64, out_channels=num_classes, dropout=0.4)
    model = to_hetero(model, data.metadata(), aggr="sum")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, model = data.to(device), model.to(device)
    masks = build_masks(strategy="RS2")

    #TRAIN THE MODEL
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
    weights = compute_weights(data[target_type].y).float().to(device) #torch.tensor([1.5, 1.5, 1, 1, 1, 1, 1.5, 1]).float().to(device)
    criterion = nn.CrossEntropyLoss(weights)
    model = train_node_classifier(model, data, masks, optimizer, criterion, n_epochs=500, target_type)
    torch.save(model.state_dict(), path_technique + base + "base_model.pth")

    ### alternative to train: LOAD SAVED MODEL
    #model = load_model(64,8,0.4,data, os.path.join(fname_base,base, "model.pth"))


    f1_micro, f1_macro, f1_weigh, auc = eval_node_classifier(model, data, target_type)
    #print(f"Test Eval: {test_ev:.3f}")
    print(f"f1-micro: {f1_micro:.3f}, f1-macro: {f1_macro:.3f}, f1-weighted: {f1_weigh:.3f}, roc-auc: {auc:.3f}")

    l_micro.append(f1_micro)
    l_macro.append(f1_macro)
    l_weigh.append(f1_weigh)
    l_auc.append(auc)

df = pd.DataFrame(columns=["F1_micro", "F1_macro", "F1_weighted", "ROC-AUC"])
df["F1_micro"] = l_micro
df["F1_macro"] = l_macro
df["F1_weighted"] = l_weigh
df["ROC-AUC"] = l_auc
df_ok = processing_results(df)
df_ok.to_excel(os.path.join(path_technique, base, "results.xlsx"), index=False)



