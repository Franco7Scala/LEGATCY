import os
import pandas
import torch

from data_preprocessing.mumin import load_mumin_heterodata
from src import utils
from src.al_techniques.entropy_al_technique import EntropyALTechnique
from src.al_techniques.lcs_al_technique import LCSALTechnique
from src.al_techniques.margin_al_technique import MarginALTechnique
from src.al_techniques.random_al_technique import RandomALTechnique
from src.models.GAT_enhanced import GAT_enhanced
from src.sampling_strategies.active_ers2 import ActiveERS2
from src.utils import set_random_seed, training_seeds, processing_results, compute_weights, cprint, Color
from src.data_utils import get_target_type
from src.data_loader import build_heterodata, get_knowledge
from src.trainer import train_node_classifier, eval_node_classifier
from torch_geometric.nn import to_hetero

from trainer import train_node_classifier_single_snapshot

dataset_name = "mumin"
n_epochs = 5
k = 10
min_lr = 1e-4
sampling_technique = MarginALTechnique  # RandomALTechnique LCSALTechnique EntropyALTechnique MarginALTechnique
training_strategy = ActiveERS2

############################################################################################


device = utils.get_device()

cprint(f"Building dataset...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
data = load_mumin_heterodata()

target_type = get_target_type(dataset_name)
num_classes = len(torch.unique(data[target_type].y))

output_dir = os.path.join("...")


l_micro = []
l_macro = []
l_weigh = []
l_auc = []
for run in range(len(training_seeds)):
    cprint(f"Performing run n {run} on {len(training_seeds)}...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
    set_random_seed(training_seeds[run])

    cprint(f"Building model...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
    model = GAT_enhanced(hidden_channels=64, out_channels=num_classes, dropout=0.4, num_layers=3)
    model = to_hetero(model, data.metadata(), aggr="sum").to(device)

    cprint(f"Training model...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
    criterion = torch.nn.CrossEntropyLoss(compute_weights(data[target_type].y).float().to(device))

    t_max = max(1, int(n_new + (n_old / n_epochs)) * n_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, t_max, eta_min=min_lr)
    strategy = training_strategy(sampling_technique(model), k)

    model = train_node_classifier_single_snapshot(model, data, optimizer, criterion, scheduler, target_type, run, strategy, directory=output_dir, n_epochs=n_epochs)
    torch.save(model.state_dict(), os.path.join(output_dir, "model_" + str(run) + ".pth"))

    f1_micro, f1_macro, f1_weigh, auc = eval_node_classifier(model, data, target_type, run, directory=output_dir)
    cprint(f"f1-micro: {f1_micro:.3f}, f1-macro: {f1_macro:.3f}, f1-weighted: {f1_weigh:.3f}, roc-auc: {auc:.3f}", Color.EXPERIMENT_OUTPUT)

    l_micro.append(f1_micro)
    l_macro.append(f1_macro)
    l_weigh.append(f1_weigh)
    l_auc.append(auc)

cprint(f"Saving results in '{output_dir}/results.xlsx'...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
data_frame = pandas.DataFrame(columns=["F1_micro", "F1_macro", "F1_weighted", "ROC-AUC"])
data_frame["F1_micro"] = l_micro
data_frame["F1_macro"] = l_macro
data_frame["F1_weighted"] = l_weigh
data_frame["ROC-AUC"] = l_auc
processing_results(data_frame).to_excel(os.path.join(output_dir, "results.xlsx"), index=False)
cprint(f"Completed!", Color.OTHER)
