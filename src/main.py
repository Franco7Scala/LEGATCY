import os
import sys
import pandas
import torch
import warnings

from torch_geometric.nn import to_hetero
from torch_geometric.explain import Explainer, CaptumExplainer
from data.dataset_loader import load_dataset
from models.HeteroGAT import HeteroGAT
from sampling_strategies.voting_strategy import VotingStrategy
from sampling_strategies.other_dyhane import DyHANE
from sampling_strategies.other_full_retraining import FullRetraining
from sampling_strategies.other_online_training import OnlineTraining
from sampling_strategies.active_ers2 import ActiveERS2
from al_techniques.random_al_technique import RandomALTechnique
from al_techniques.margin_al_technique import MarginALTechnique
from al_techniques.lcs_al_technique import LCSALTechnique
from al_techniques.entropy_al_technique import EntropyALTechnique
from support import utils
from support.arguments import parse_arguments
from support.utils import str2bool, set_random_seed, training_seeds, processing_results, cprint, Color, get_base_dir, get_time_in_millis, Kwargs, compute_weights, print_samples_count
from support.utils_data import create_nodes_dict_empty
from trainer import train, evaluate


args = parse_arguments()

# experiment setting parameters
debug = str2bool(args.debug)
dataset_name = args.dataset_name
n_snapshot = args.n_snapshot
times_fist_snapshot = args.times_first_snapshot
metapaths_enabled = str2bool(args.metapaths_enabled)
subgraph_hops = args.subgraph_hops
training_strategy = getattr(sys.modules[__name__], args.training_strategy)
sampling_technique = getattr(sys.modules[__name__], args.sampling_technique)
k = args.k
reduction_factor = args.reduction_factor
results_dir = args.result_directory

# training parameters
n_epochs = args.n_epochs
max_lr = args.max_lr
min_lr = args.min_lr

# model parameters
num_layers = args.num_layers
hidden_channels = args.hidden_channels
dropout = args.dropout


############################################################################################


kwargs = Kwargs()
warnings.filterwarnings("ignore")
device = utils.get_device()
if results_dir is None:
    root_dir = os.path.join(get_base_dir(), dataset_name, "results_debug" if debug else "results", str(get_time_in_millis()))

else:
    root_dir = results_dir

os.makedirs(root_dir, exist_ok=True)
cprint(f"Saving results in '{root_dir}'", Color.EXPERIMENT_CONFIG_INFO)

if debug:
    n_epochs = 1
    training_seeds = training_seeds[:2]

cprint(f"Experiment config:\n"
       f"- Dataset: {dataset_name}\n"
       f"- metapaths: {metapaths_enabled}\n"
       f"- n epochs: {n_epochs}\n"
       f"- k: {k}\n"
       f"- max lr: {max_lr}\n"
       f"- min lr: {min_lr}\n"
       f"- num layers: {num_layers}\n"
       f"- hidden channels: {hidden_channels}\n"
       f"- dropout: {dropout}\n"
       #f"- focal gamma: {focal_gamma}\n"
       f"- n snapshot: {n_snapshot}\n"
       f"- Training strategy: {training_strategy.__name__}\n"
       f"- Sampling technique: {'None' if sampling_technique is None else sampling_technique.__name__}\n"
       f"- Device: {device}\n", Color.EXPERIMENT_CONFIG_INFO)

cprint(f"Loading dataset...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
data, target_type, snapshot_masks = load_dataset(dataset_name=dataset_name, metapaths_enabled=metapaths_enabled, n_snapshot=n_snapshot, times_fist_snapshot=times_fist_snapshot, k=subgraph_hops, device=device)
data = data.to(device)

for idx_snapshot, snapshot in enumerate(snapshot_masks):
    cprint(f"Working on snapshot n.{idx_snapshot + 1}...", Color.EXPERIMENT_CONFIG_INFO)
    if results_dir is None:
        subgraphs_dir = os.path.join(get_base_dir(), dataset_name, "subgraphs", f"snapshot_{idx_snapshot}")

    else:
        subgraphs_dir = os.path.join(root_dir, "subgraphs", f"snapshot_{idx_snapshot}")

    os.makedirs(subgraphs_dir, exist_ok=True)
    kwargs.subgraphs_dir = subgraphs_dir

    cprint(f"Initializing training strategy...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
    strategy = training_strategy()
    new_nodes = snapshot_masks[idx_snapshot]
    if idx_snapshot != 0:
        old_nodes = snapshot_masks[idx_snapshot - 1]

    else:
        old_nodes = {"test": create_nodes_dict_empty(data, dtype=torch.tensor), "train": create_nodes_dict_empty(data, dtype=torch.tensor)}

    num_classes = len(torch.unique(data[target_type].y))
    cprint(f"Number of classes: {num_classes}", Color.EXPERIMENT_STATUS_LOW_PRIORITY)

    cprint(f"Number of new samples per class:", Color.EXPERIMENT_STATUS_LOW_PRIORITY)
    print_samples_count(new_nodes["train"])
    cprint(f"Number of old samples per class:", Color.EXPERIMENT_STATUS_LOW_PRIORITY)
    print_samples_count(old_nodes["train"])

    previous_output_dir = os.path.join(root_dir, f"snapshot_{idx_snapshot-1}")
    output_dir = os.path.join(root_dir, f"snapshot_{idx_snapshot}")
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
        model = HeteroGAT(
            metadata=data.metadata(),
            target_type=target_type,
            hidden_channels=hidden_channels,
            out_channels=num_classes,
            dropout=dropout,
            num_layers=num_layers
        ).to(device)

        if idx_snapshot != 0 and strategy.needs_previous_model:
            cprint(f"Loading model from previous snapshot...", Color.EXPERIMENT_STATUS_LOW_PRIORITY)
            model.load_state_dict(torch.load(os.path.join(previous_output_dir, f"model_{run}.pth")))
            kwargs.old_model = model

        cprint(f"Training model...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
        optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, weight_decay=0.001)
        #criterion = FocalLoss(num_classes=num_classes, gamma=focal_gamma, alpha=focal_alpha, reduction="mean")
        criterion = torch.nn.CrossEntropyLoss(compute_weights(data[target_type].y).float().to(device))

        t_max = n_epochs
        scheduler = None #torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, t_max, eta_min=min_lr)
        if sampling_technique is not None:
            strategy.al_technique = sampling_technique(model)

        if k is not None:
            strategy.k = k

        model = train(model, data, new_nodes, old_nodes, optimizer, criterion, scheduler, run, strategy, directory=output_dir, reduction_factor=reduction_factor, n_epochs=n_epochs, kwargs=kwargs)
        torch.save(model.state_dict(), os.path.join(output_dir, f"model_{run}.pth"))
        f1_micro, f1_macro, auc, precision, recall = evaluate(model, data, new_nodes, old_nodes, run, directory=output_dir)
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
    cprint(f"Completed snapshot n.{idx_snapshot + 1}!", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)

cprint(f"Completed!", Color.OTHER)
