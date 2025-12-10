#!/bin/bash
gpus=(0 1 2 3 4 5 6)

# experiment setting parameters
debug="True"
dataset_name="imdb"
n_snapshot=4
times_fist_snapshot=3
metapaths_enabled="False"
subgraph_hops=2
training_strategies=(DyHANE ActiveERS2 FullRetraining OnlineTraining VotingStrategy)
sampling_techniques=(RandomALTechnique LCSALTechnique EntropyALTechnique MarginALTechnique)
k=500
reduction_factor=0.5

# training parameters
n_epochs=200
max_lr=0.01
min_lr=0.001

# model parameters
num_layers=3
hidden_channels=64
dropout=0.3


source /home/scala/.virtualenvs/GNN_ContinualLearning/bin/activate
export PYTHONPATH=$PYTHONPATH:/projects/GNN_ContinualLearning/

gpu_id_index=${gpus[0]}
for i in "${!training_strategies[@]}"; do
  current_training_strategy="${training_strategies[$i]}"
  if [ "$current_training_strategy" = "ActiveERS2" ]; then
    for j in "${!sampling_techniques[@]}"; do
      path_results="./results/debug_${debug}/${dataset_name}/n_snap_${n_snapshot}_n_first_${times_fist_snapshot}_k_${k}_meta_${metapaths_enabled}_rfactor_${reduction_factor//./}/${current_training_strategy}/${sampling_techniques[$j]}"
      mkdir -p "$path_results"
      echo "Running Experiment in debug mode turned ${debug} with technique $current_training_strategy and sampling policy $sampling_techniques on dataset: $dataset_name, n snapshots value: $n_snapshot, times first snapshot: times_fist_snapshot, k: $k, metapaths enabled: $metapaths_enabled, reduction factor: $reduction_factor on GPU: $gpu_id_index"
      echo "Saving results in: $path_results"
      CUDA_VISIBLE_DEVICES=$gpu_id_index python src/main.py --result-directory "$path_results" --debug "$debug" --dataset-name "$dataset_name" --n-snapshot "$n_snapshot" --times-first-snapshot "$times_fist_snapshot" --metapaths-enabled "$metapaths_enabled" --training-strategy "$current_training_strategy" --sampling-technique "${sampling_techniques[$j]}" --k "$k" --reduction-factor "$reduction_factor" > "$path_results/log_$(date +%s%3N).log" 2>&1 &
      sleep 1
      gpu_id_index=${gpus[$(( ((i + j) + 1) % ${#gpus[@]} ))]}
    done
  else
    path_results="./results/debug_${debug}/${dataset_name}/n_snap_${n_snapshot}_n_first_${times_fist_snapshot}_k_${k}_meta_${metapaths_enabled}_rfactor_${reduction_factor//./}/${current_training_strategy}"
    mkdir -p "$path_results"
    echo "Running Experiment in debug mode turned ${debug} with technique $current_training_strategy and sampling policy $sampling_techniques on dataset: $dataset_name, n snapshots value: $n_snapshot, times first snapshot: times_fist_snapshot, k: $k, metapaths enabled: $metapaths_enabled, reduction factor: $reduction_factor on GPU: $gpu_id_index"
    echo "Saving results in: $path_results"
    CUDA_VISIBLE_DEVICES=$gpu_id_index python src/main.py --result-directory "$path_results" --debug "$debug" --dataset-name "$dataset_name" --n-snapshot "$n_snapshot" --times-first-snapshot "$times_fist_snapshot" --metapaths-enabled "$metapaths_enabled" --training-strategy "$current_training_strategy" --k "$k" --reduction-factor "$reduction_factor" > "$path_results/log_$(date +%s%3N).log" 2>&1 &
    sleep 1
    gpu_id_index=${gpus[$(( ((i + j) + 1) % ${#gpus[@]} ))]}
  fi
done

wait

echo "Completed!"
