import os
import pandas
import torch

from src import utils
from src.al_techniques.margin_al_technique import MarginALTechnique
from src.models.GAT import GAT
from src.sampling_strategies.active_ers2 import ActiveERS2
from src.utils import set_random_seed, training_seeds, processing_results, compute_weights, cprint, Color, count_n_snapshots
from src.data.data_utils import get_target_type
from src.data.graph_loader import build_heterodata, get_knowledge
from src.trainer import train, evaluate
from torch_geometric.nn import to_hetero


dataset_name = "openalex"
n_epochs = 1
k = 10
min_lr = 1e-4
training_strategy = ActiveERS2
sampling_technique = MarginALTechnique  # RandomALTechnique LCSALTechnique EntropyALTechnique MarginALTechnique

# TODO sistemare quel bug che fa scoppiare il training
# TODO cacciare un po di stampe superflue (per Liliana che programma come uno scimpanzé)
# TODO aggistare GT alex
# TODO manca train-val-test mask in data

############################################################################################


device = utils.get_device()
n_snapshot = count_n_snapshots(dataset_name)

for snapshot in range(n_snapshot):
    cprint(f"Working on snapshot n.{snapshot}...", Color.EXPERIMENT_CONFIG_INFO)
    cprint(f"Building dataset...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
    data = build_heterodata(dataset_name=dataset_name, no_snapshot=snapshot).to(device)

    new_nodes, new_edges = get_knowledge(dataset_name=dataset_name, no_snapshot=snapshot, new=True)
    old_nodes, old_edges = get_knowledge(dataset_name=dataset_name, no_snapshot=snapshot, new=False)
    target_type = get_target_type(dataset_name)
    num_classes = len(torch.unique(data[target_type].y))

    cprint(f"Number of classes: {num_classes}", Color.EXPERIMENT_STATUS_LOW_PRIORITY)
    output_dir = os.path.join("data", dataset_name, f"snapshot_{snapshot}", "processed_data")

    l_micro = []
    l_macro = []
    l_auc = []
    for run in range(len(training_seeds)):
        cprint(f"Performing run n {run} on {len(training_seeds)}...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
        set_random_seed(training_seeds[run])

        cprint(f"Building model...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
        model = GAT(hidden_channels=64, out_channels=num_classes, dropout=0.4, num_layers=3)
        model = to_hetero(model, data.metadata(), aggr="sum").to(device)

        cprint(f"Training model...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
        criterion = torch.nn.CrossEntropyLoss(compute_weights(data[target_type].y).float().to(device))

        n_old = sum([len(old_nodes[val]) for val in old_nodes.keys()])
        n_new = sum([len(new_nodes[val]) for val in new_nodes.keys()])
        t_max = max(1, int(n_new + (n_old / n_epochs)) * n_epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, t_max, eta_min=min_lr)
        strategy = training_strategy(sampling_technique(model), k)

        model = train(model, data, new_nodes, new_edges, old_nodes, old_edges, optimizer, criterion, scheduler, target_type, run, strategy, directory=output_dir, n_epochs=n_epochs)
        torch.save(model.state_dict(), os.path.join(output_dir, f"model_{run}.pth"))

        f1_micro, f1_macro, auc = evaluate(model, data, target_type, run, directory=output_dir)
        cprint(f"f1-micro: {f1_micro:.3f}, f1-macro: {f1_macro:.3f}, roc-auc: {auc:.3f}", Color.EXPERIMENT_OUTPUT)

        l_micro.append(f1_micro)
        l_macro.append(f1_macro)
        l_auc.append(auc)

    cprint(f"Saving results in '{output_dir}/results.xlsx'...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
    data_frame = pandas.DataFrame(columns=["F1_micro", "F1_macro", "ROC-AUC"])
    data_frame["F1_micro"] = l_micro
    data_frame["F1_macro"] = l_macro
    data_frame["ROC-AUC"] = l_auc
    processing_results(data_frame).to_excel(os.path.join(output_dir, "results.xlsx"), index=False)
    cprint(f"Completed snapshot n.{snapshot}!", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)

cprint(f"Completed!", Color.OTHER)
