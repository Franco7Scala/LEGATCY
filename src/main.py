import os
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import to_hetero

from src import utils
from src.models.GAT_enhanced import GAT_enhanced
from src.utils import set_random_seed, training_seeds, processing_results, compute_weights
from src.data_utils import get_target_type
from src.data_loader import build_heterodata, get_knowledge
from src.trainer import train_node_classifier, eval_node_classifier


dataset_name = "openalex_subset"
no_snapshot = 1 #current snapshot
print("Building heterodata...")
data = build_heterodata(dataset_name=dataset_name, no_snapshot=no_snapshot)
print("heterodata object correctly build!")
print(data)
K_new_nodes, K_new_edges = get_knowledge(dataset_name=dataset_name, no_snapshot=no_snapshot, new=True)
K_old_nodes, K_old_edges = get_knowledge(dataset_name=dataset_name, no_snapshot=no_snapshot, new=False)
target_type = get_target_type(dataset_name)
num_classes = len(torch.unique(data[target_type].y))
print(f"Number of classes: {num_classes}")
strategy = "RS2"
output_dir = os.path.join("data", dataset_name, f"snapshot_{no_snapshot}", "processed_data")

l_micro = []
l_macro = []
l_weigh = []
l_auc = []
for run in range(len(training_seeds)):
    print(f"Performing run n {run} on {len(training_seeds)}...")
    set_random_seed(training_seeds[run])

    model = GAT_enhanced(hidden_channels=64, out_channels=num_classes, dropout=0.4, num_layers=3)
    model = to_hetero(model, data.metadata(), aggr="sum")
    device = utils.get_device()
    data, model = data.to(device), model.to(device)

    #TRAIN THE MODEL
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
    weights = compute_weights(data[target_type].y).float().to(device) #torch.tensor([1.5, 1.5, 1, 1, 1, 1, 1.5, 1]).float().to(device)
    criterion = nn.CrossEntropyLoss(weights)
    model = train_node_classifier(model, data, K_new_nodes, K_new_edges, K_old_nodes, K_old_edges, optimizer, criterion, target_type, run, strategy, directory=output_dir, n_epochs=500)
    torch.save(model.state_dict(), os.path.join(output_dir, "model_"+str(run)+".pth"))


    f1_micro, f1_macro, f1_weigh, auc = eval_node_classifier(model, data, target_type, run, directory=output_dir)
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
df_ok.to_excel(os.path.join(output_dir, "results.xlsx"), index=False)



