#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/projects/GNNContinualLearning


# experiment setting parameters
debug="off"
dataset_name="dblp"
n_snapshot=4
times_fist_snapshot=3
metapaths_enabled="False"
training_strategies=(ActiveERS2 VotingStrategy DyHANE)
sampling_techniques=(RandomALTechnique LCSALTechnique)
k_values=(25 100 150)
reduction_factor=0.5


path_results="/home/jovyan/projects/GNNContinualLearning/results/k/$dataset_name/"
mkdir -p "$path_results"
echo "Saving results in: $path_results"

for k in "${k_values[@]}"; do
  for i in "${!training_strategies[@]}"; do
    current_training_strategy="${training_strategies[$i]}"
    if [ "$current_training_strategy" = "ActiveERS2" ]; then
      for j in "${!sampling_techniques[@]}"; do
        echo "Running Experiment in debug mode turned ${debug} with technique $current_training_strategy and sampling policy ${sampling_techniques[$j]} on dataset: $dataset_name, n snapshots value: $n_snapshot, times first snapshot: $times_fist_snapshot, k: $k, metapaths enabled: $metapaths_enabled, reduction factor: $reduction_factor"
        python src/main.py --result-directory "$path_results" --debug "$debug" --dataset-name "$dataset_name" --n-snapshot "$n_snapshot" --times-first-snapshot "$times_fist_snapshot" --metapaths-enabled "$metapaths_enabled" --training-strategy "$current_training_strategy" --sampling-technique "${sampling_techniques[$j]}" --k "$k" --reduction-factor "$reduction_factor" > "$path_results/log_$(date +%s%3N).log" 2>&1
      done
    else
      echo "Running Experiment in debug mode turned ${debug} with technique $current_training_strategy and sampling policy None on dataset: $dataset_name, n snapshots value: $n_snapshot, times first snapshot: $times_fist_snapshot, k: $k, metapaths enabled: $metapaths_enabled, reduction factor: $reduction_factor"
      python src/main.py --result-directory "$path_results" --debug "$debug" --dataset-name "$dataset_name" --n-snapshot "$n_snapshot" --times-first-snapshot "$times_fist_snapshot" --metapaths-enabled "$metapaths_enabled" --training-strategy "$current_training_strategy" --k "$k" --reduction-factor "$reduction_factor" > "$path_results/log_$(date +%s%3N).log" 2>&1
    fi
  done
done

message="Experiments Continual Learning k variation completed!"
message=${message// /%20}
token="7531410690:AAERJ_0H8THYS098xpSMvzVfPrflMr3iaW8"
chat_ids=(255950847 496539491)
for chat_id in "${chat_ids[@]}"; do
    url="https://api.telegram.org/bot${token}/sendMessage?chat_id=${chat_id}&text=${message}"
    curl -s -o /dev/null "$url"
done

echo "All experiments completed!"
