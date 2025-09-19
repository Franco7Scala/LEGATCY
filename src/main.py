import os
import sys
import pandas
import torch
import warnings

from torch_geometric.explain import Explainer, CaptumExplainer

from models.HeteroGAT import HeteroGAT
from src.sampling_strategies.other_full_retraining import FullRetraining
from src.sampling_strategies.other_online_training import OnlineTraining
from src.sampling_strategies.active_ers2 import ActiveERS2
from src.al_techniques.random_al_technique import RandomALTechnique
from src.al_techniques.margin_al_technique import MarginALTechnique
from src.al_techniques.lcs_al_technique import LCSALTechnique
from src.al_techniques.entropy_al_technique import EntropyALTechnique
from src.support import utils
from src.models.GAT import GAT
from src.support.focal_loss import FocalLoss
from src.support.utils import set_random_seed, training_seeds, processing_results, cprint, Color, count_n_snapshots, get_base_dir, get_time_in_millis, get_class_distribution, Kwargs, compute_weights
from src.data.data_utils import get_target_type
from src.data.graph_loader import build_heterodata, get_knowledge
from src.trainer import train, evaluate
from torch_geometric.nn import to_hetero


dataset_name = "openalex"
n_epochs = 1
max_lr = 0.005
min_lr = 1e-4
focal_gamma = 2
focal_alpha = 5
training_strategy = ActiveERS2 #FullRetraining  # FullRetraining OnlineTraining ActiveERS2
sampling_technique = MarginALTechnique#None           # used only with ActiveERS2, RandomALTechnique LCSALTechnique EntropyALTechnique MarginALTechnique
k = 500#None                            # used only with ActiveERS2, it identifies the amount of data to keep from the old nodes


############################################################################################


kwargs = Kwargs()
warnings.filterwarnings("ignore")
device = utils.get_device()
n_snapshot = count_n_snapshots(dataset_name)
root_dir = os.path.join(get_base_dir(), dataset_name, "results", str(get_time_in_millis()))
os.makedirs(root_dir, exist_ok=True)
cprint(f"Saving results in '{root_dir}'", Color.EXPERIMENT_CONFIG_INFO)

std_out = sys.stdout
sys.stdout = open(os.path.join(root_dir, "log_file.log"), "w")

cprint(f"Experiment config:\n"
       f"- Dataset: {dataset_name}\n"
       f"- n epochs: {n_epochs}\n"
       f"- k: {k}\n"
       f"- max lr: {max_lr}\n"
       f"- min lr: {min_lr}\n"
       f"- focal gamma: {focal_gamma}\n"
       f"- n snapshot: {n_snapshot}\n"
       f"- Training strategy: {training_strategy.__name__}\n"
       f"- Sampling technique: {'None' if sampling_technique is None else sampling_technique.__name__}\n"
       f"- Device: {device}\n", Color.EXPERIMENT_CONFIG_INFO)

for snapshot in range(n_snapshot):
    cprint(f"Working on snapshot n.{snapshot}...", Color.EXPERIMENT_CONFIG_INFO)
    subgraphs_dir = os.path.join(get_base_dir(), dataset_name, "subgraphs", f"snapshot_{snapshot}")
    os.makedirs(subgraphs_dir, exist_ok=True)
    kwargs.subgraphs_dir = subgraphs_dir

    cprint(f"Initializing training strategy...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
    strategy = training_strategy()

    cprint(f"Building dataset...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
    data = build_heterodata(dataset_name=dataset_name, no_snapshot=snapshot).to(device)

    if snapshot != 0 and strategy.needs_previous_data:
        kwargs.old_data = build_heterodata(dataset_name=dataset_name, no_snapshot=snapshot - 1).to(device)

    else:
        kwargs.old_data = None

    new_nodes, new_edges = get_knowledge(dataset_name=dataset_name, no_snapshot=snapshot, new=True)
    old_nodes, old_edges = get_knowledge(dataset_name=dataset_name, no_snapshot=snapshot, new=False)
    target_type = get_target_type(dataset_name)
    num_classes = len(torch.unique(data[target_type].y))

    class_distribution = get_class_distribution(data, target_type)
    for i, count in enumerate(class_distribution):
        print(f"Samples' number of class {i}: {count.item()}")

    print()

    cprint(f"Number of classes: {num_classes}", Color.EXPERIMENT_STATUS_LOW_PRIORITY)
    previous_output_dir = os.path.join(root_dir, f"snapshot_{snapshot-1}")
    output_dir = os.path.join(root_dir, f"snapshot_{snapshot}")
    os.makedirs(output_dir, exist_ok=True)

    l_micro = []
    l_macro = []
    l_auc = []
    l_times = []
    l_precision = []
    l_recall = []
    for run in range(len(training_seeds)):
        start_time = get_time_in_millis()
        cprint(f"Performing run n {run + 1} on {len(training_seeds)}...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
        set_random_seed(training_seeds[run])

        cprint(f"Building model...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)

        # OLD version using to_hetero wrapper on the GAT model
        #model = GAT(hidden_channels=64, out_channels=num_classes, dropout=0.4, num_layers=3)
        #model = to_hetero(model, data.metadata(), aggr="sum").to(device)
        #out, emb = model(data.x_dict, data.edge_index_dict)

        #NEW version using directly HeteroGAT with HeteroConv
        model = HeteroGAT(
            metadata=data.metadata(),
            hidden_channels=64,
            out_channels=num_classes,
            dropout=0.4,
            num_layers=3
        ).to(device)
        out_dict, emb_dict = model(data.x_dict, data.edge_index_dict)

        if snapshot != 0 and strategy.needs_previous_model:
            cprint(f"Loading model from previous snapshot...", Color.EXPERIMENT_STATUS_LOW_PRIORITY)
            model.load_state_dict(torch.load(os.path.join(previous_output_dir, f"model_{run}.pth")))
            kwargs.old_model = model

        cprint(f"Training model...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
        optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, weight_decay=0.001)
        #criterion = FocalLoss(num_classes=num_classes, gamma=focal_gamma, alpha=focal_alpha, reduction="mean")
        criterion = torch.nn.CrossEntropyLoss(compute_weights(data[target_type].y).float().to(device))

        n_old = sum([len(old_nodes[val]) for val in old_nodes.keys()])
        n_new = sum([len(new_nodes[val]) for val in new_nodes.keys()])
        t_max = max(1, int(n_new + (n_old / n_epochs)) * n_epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, t_max, eta_min=min_lr)
        if sampling_technique is not None:
            strategy.al_technique = sampling_technique(model)

        if k is not None:
            strategy.k = k

        model = train(model, data, new_nodes, new_edges, old_nodes, old_edges, optimizer, criterion, scheduler, target_type, run, strategy, directory=output_dir, n_epochs=n_epochs, kwargs=kwargs)
        torch.save(model.state_dict(), os.path.join(output_dir, f"model_{run}.pth"))

        f1_micro, f1_macro, auc, precision, recall = evaluate(model, data, target_type, run, directory=output_dir)
        elapsed_time = get_time_in_millis() - start_time
        cprint(f"f1-micro: {f1_micro:.3f}, f1-macro: {f1_macro:.3f}, roc-auc: {auc:.3f}, precision: {[{' '.join('{:.5f}'.format(x) for x in precision)}]}, recall: {[{' '.join('{:.5f}'.format(x) for x in recall)}]}, time: {elapsed_time}", Color.EXPERIMENT_OUTPUT)

        l_micro.append(f1_micro)
        l_macro.append(f1_macro)
        l_auc.append(auc)
        l_precision.append(precision)
        l_recall.append(recall)
        l_times.append(elapsed_time)

    cprint(f"Saving results in '{output_dir}/results.xlsx'...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
    data_frame = pandas.DataFrame(columns=["F1_micro", "F1_macro", "ROC-AUC", "time"])
    data_frame["F1_micro"] = l_micro
    data_frame["F1_macro"] = l_macro
    data_frame["ROC-AUC"] = l_auc
    data_frame["time"] = l_times
    processing_results(data_frame).to_excel(os.path.join(output_dir, "results.xlsx"), index=False)
    cprint(f"Completed snapshot n.{snapshot}!", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)

cprint(f"Completed!", Color.OTHER)

sys.stdout = std_out
cprint(f"Completed!", Color.OTHER)
